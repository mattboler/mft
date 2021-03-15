#include "mft/feature_tracker.hpp"

namespace mft {

FeatureTrackerParams buildTrackerParams(std::string path)
{
    FeatureTrackerParams p;

    auto n = YAML::LoadFile(path);

    p.num_features = n["num_features"].as<int>();
    p.use_clahe = n["use_clahe"].as<bool>();
    p.minimum_distance = n["minimum_distance"].as<size_t>();
    p.use_subpix = n["use_subpix"].as<bool>();
    p.window_size = n["window_size"].as<size_t>();
    p.num_levels = n["num_levels"].as<size_t>();
    p.ransac_pix_threshold = n["ransac_pix_threshold"].as<double>();
    p.ransac_confidence = n["ransac_confidence"].as<double>();

    return p;
}

Camera buildCamera(std::string path) 
{
    Camera c;
    
    auto n = YAML::LoadFile(path);

    auto s = n["size"];
    c.width = s["width"].as<size_t>();
    c.height = s["height"].as<size_t>();

    auto i = n["intrinsics"];
    c.fx = i["fx"].as<double>();
    c.fy = i["fy"].as<double>();
    c.cx = i["cx"].as<double>();
    c.cy = i["cy"].as<double>();
    c.K = (cv::Mat1d(3, 3) <<
        c.fx, 0, c.cx,
        0, c.fy, c.cy,
        0, 0, 1);

    auto d = n["distortion"];
    c.D = (cv::Mat1d(1, 5) << 
        d["k1"].as<double>(),
        d["k2"].as<double>(),
        d["p1"].as<double>(),
        d["p2"].as<double>(),
        d["k3"].as<double>());
    
    return c;
}

FeatureTracker::FeatureTracker()
{

}

FeatureTracker::FeatureTracker(
    FeatureTrackerParams params,
    Camera cam)
{
    // set params
    this->params_ = params;
    // set cam
    this->cam_ = cam;
}

Frame
FeatureTracker::buildFirstFrame(
    const cv::Mat& img)
{
    auto f = Frame();
    img.copyTo(f.img);

    if (params_.use_clahe) {
        auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(f.img, f.img);
    }
    
    extractFeatures_(f);

    return f;
}

Frame FeatureTracker::buildNextFrame(
    const cv::Mat& img,
    const Frame& prev_frame)
{
    auto f = Frame();
    img.copyTo(f.img);

    if (params_.use_clahe) {
        auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(f.img, f.img);
    }

    // Track previous features if possible
    // Update ages, fill in velocities etc
    trackFeatures_(
        f,
        prev_frame);
    
    // Detect new features
    // Assign new features ids, ages, etc
    extractFeatures_(f);

    return f;
}

cv::Mat
FeatureTracker::annotateFrame(
    const Frame& f)
{
    cv::Mat img;
    f.img.copyTo(img);

    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    for (const auto& pt : f.points) {
        cv::circle(
            img,
            pt,
            3,
            cv::Scalar(0, 0, 255),
            -1);
    }

    return img;
}

void
FeatureTracker::extractFeatures_(
    Frame& f)
{   
    /**
     * 1. Extract features
     * 2. Assign IDs, ages, and velocies to new features
     */

    if (params_.num_features <= f.points.size()) { // Do we have enough already?
        return;
    }

    if (f.points.size() != 0) { // Need to mask off previous points
        detectFeaturesMask_(f);
    } else {
        detectFeaturesNoMask_(f);
    }
}

void
FeatureTracker::detectFeaturesMask_(
    Frame& f)
{
    std::vector<cv::Point2f> new_points;

    cv::Mat mask = buildMask_(
        f.img,
        f.points,
        this->params_.minimum_distance);
    
    // Build data for new frame
    // get features
    cv::goodFeaturesToTrack(
        f.img,
        new_points,
        params_.num_features - f.points.size(),
        0.01,
        this->params_.minimum_distance,
        mask);

    // undistort features
    auto new_points_und = undistortAndNormalizePoints_(new_points);
    
    // assign ids
    std::vector<uint64_t> new_ids;
    new_ids.reserve(new_points.size());
    for (size_t i = 0; i < new_points.size(); ++i) {
        new_ids.push_back(this->id_counter_++);
    }
    // assign ages
    std::vector<uint64_t> new_ages(new_points.size(), 0);
    // assign velocities (negative means first sighting of this feature)
    std::vector<double> new_velocities(new_points.size(), -1.0);

    // Merge old and new frame data
    f.ids.insert(
        f.ids.end(), new_ids.begin(), new_ids.end());
    f.points.insert(
        f.points.end(), new_points.begin(), new_points.end());
    f.points_und.insert(
        f.points_und.end(), new_points_und.begin(), new_points_und.end());
    f.ages.insert(
        f.ages.end(), new_ages.begin(), new_ages.end());
    f.velocities.insert(
        f.velocities.end(), new_velocities.begin(), new_velocities.end());
}

cv::Mat
FeatureTracker::buildMask_(
    const cv::Mat& img,
    const std::vector<cv::Point2f>& pts,
    const int radius)
{
    cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(255));

    for (const auto& pt : pts) {
        cv::circle(mask, pt, radius, 0, -1);
    }

    return mask;
}

void
FeatureTracker::detectFeaturesNoMask_(
    Frame& f)
{
    std::vector<cv::Point2f> new_points;
    
    // Build data for new frame
    // get features
    cv::goodFeaturesToTrack(
        f.img,
        new_points,
        params_.num_features - f.points.size(),
        0.01,
        params_.minimum_distance);
    // undistort features
    auto new_points_und = undistortAndNormalizePoints_(new_points);
    
    // assign ids
    std::vector<uint64_t> new_ids;
    new_ids.reserve(new_points.size());
    for (size_t i = 0; i < new_points.size(); ++i) {
        new_ids.push_back(this->id_counter_++);
    }
    // assign ages
    std::vector<uint64_t> new_ages(new_points.size(), 0);
    // assign velocities
    std::vector<double> new_velocities(new_points.size(), 0.0);
    // Merge old and new frame data
    f.ids.insert(
        f.ids.end(), new_ids.begin(), new_ids.end());
    f.points.insert(
        f.points.end(), new_points.begin(), new_points.end());
    f.points_und.insert(
        f.points_und.end(), new_points_und.begin(), new_points_und.end());
    f.ages.insert(
        f.ages.end(), new_ages.begin(), new_ages.end());
    f.velocities.insert(
        f.velocities.end(), new_velocities.begin(), new_velocities.end());
}

void
FeatureTracker::trackFeatures_(
    Frame& next_frame,
    const Frame& prev_frame)
{
    // Extract previous info
    auto prev_img = prev_frame.img;
    auto ids = prev_frame.ids;
    auto ages = prev_frame.ages;
    auto prev_points = prev_frame.points;

    // Extract incoming info
    auto next_img = next_frame.img;

    std::vector<cv::Point2f> new_points;
    std::vector<uchar> pyrlk_status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
       prev_img,
       next_img,
       prev_points,
       new_points,
       pyrlk_status,
       err,
       cv::Size(params_.window_size, params_.window_size),
       params_.num_levels);

    // Filter out points that moved too much
    for (int i = 0; i < new_points.size(); ++i) {
        if(pyrlk_status[i] && cv::norm(new_points[i] - prev_frame.points[i]) > 25) {
            pyrlk_status[i] = 0;
        }
    }

    // Filter out failed tracking points
    filterByMask(ids, pyrlk_status);
    filterByMask(prev_points, pyrlk_status);
    filterByMask(new_points, pyrlk_status);
    filterByMask(ages, pyrlk_status);
    
    if (ids.size() < 8) { // Need at least 8 points to estimate FMAT
    /**
     * TODO: Return here with "empty" frame instead of erroring out 
     */
        throw "Too few points for ransac!";
    }

    // Increment ages while we're at it
    for (int i = 0; i < ages.size(); ++i) {
        ages[i]++;
    }
    // create velocities too
    std::vector<double> new_velocities(new_points.size());
    for (int i = 0; i < new_points.size(); ++i) {
        new_velocities[i] = cv::norm(new_points[i] - prev_points[i]);
    }

    // Use pixel coordinates for ransac so pix_thresh makes sense
    auto new_points_und = undistortPoints_(new_points);
    auto prev_points_und = undistortPoints_(prev_points);

    // Reject more outliers with FMAT
    std::vector<uchar> fmat_status;

    cv::findFundamentalMat(
        prev_points_und,
        new_points_und,
        cv::FM_RANSAC,
        this->params_.ransac_pix_threshold,
        this->params_.ransac_confidence,
        fmat_status);

    filterByMask(ids, fmat_status);
    filterByMask(new_points, fmat_status);
    filterByMask(ages, fmat_status);
    filterByMask(new_velocities, fmat_status);

    // Assemble frame
    next_frame.ids = ids;
    next_frame.points = new_points;
    next_frame.points_und = undistortAndNormalizePoints_(new_points);
    next_frame.ages = ages;
    next_frame.velocities = new_velocities;
}

std::vector<cv::Point2f>
FeatureTracker::undistortPoints_(
    const std::vector<cv::Point2f>& pts)
{
    std::vector<cv::Point2f> out_vec;

    cv::undistortPoints(
        pts,
        out_vec,
        cam_.K,
        cam_.D,
        cv::noArray(),
        cam_.K);
    
    return out_vec;
}

std::vector<cv::Point2f>
FeatureTracker::undistortAndNormalizePoints_(
    const std::vector<cv::Point2f>& pts)
{
    std::vector<cv::Point2f> out_vec;

    cv::undistortPoints(
        pts,
        out_vec,
        cam_.K,
        cam_.D);
    
    return out_vec;
}

} // namespace mft