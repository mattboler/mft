#include "mft/feature_tracker.hpp"

namespace mft {

FeatureTrackerParams buildTrackerParams(std::string path)
{
    FeatureTrackerParams p;

    auto n = YAML::LoadFile(path);

    p.num_features = n["num_features"].as<double>();
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
    f.img = img;

    if (params_.use_clahe) {
        auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(f.img, f.img);
    }
    
    extractFeatures(f);

    return f;
}

Frame FeatureTracker::buildNextFrame(
    const cv::Mat& img,
    const Frame& prev_frame)
{
    auto f = Frame();
    f.img = img;

    if (params_.use_clahe) {
        auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(f.img, f.img);
    }

    // Track previous features if possible
    // Update ages, fill in velocities etc
    trackFeatures(
        f,
        prev_frame);
    
    // Detect new features
    // Assign new features ids, ages, etc
    extractFeatures(f);

    return f;
}

void
FeatureTracker::extractFeatures(
    Frame& f)
{   
    /**
     * 1. Extract features
     * 2. Assign IDs, ages, and velocies to new features
     * 3.
     */

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
        this->params_.num_features - f.points.size(),
        0.01,
        this->params_.minimum_distance,
        mask);
    // undistort features
    auto new_points_und = undistortAndNormalizePoints(new_points);
    
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
buildMask_(
    const cv::Mat& img,
    const std::vector<cv::Point2f>& pts,
    const int radius)
{
    cv::Mat mask(img.rows, img.cols, CV_8UC1, cv::Scalar(255));

    for (const auto& pt : pts) {
        cv::circle(mask, pt, radius, 0, -1);
    }
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
        this->params_.num_features - f.points.size(),
        0.01,
        this->params_.minimum_distance);
    // undistort features
    auto new_points_und = undistortAndNormalizePoints(new_points);
    
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
FeatureTracker::trackFeatures(
    Frame& next_frame,
    const Frame& prev_frame)
{
    std::vector<cv::Point2f> new_points;
    std::vector<uchar> status;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        prev_frame.img,
        next_frame.img,
        prev_frame.points,
        new_points,
        status,
        err,
        cv::Size(this->params_.window_size, this->params_.window_size),
        this->params_.num_levels);
    
    for (int i = 0; i < new_points.size(); ++i) {
        if(status[i] && cv::norm(new_points[i] - prev_frame.points[i]) > 25) {
            status[i] = 0;
        }
    }

    // Filter out failed tracking points
    auto ids = prev_frame.ids;
    filterByMask(ids, status);

    auto prev_points = prev_frame.points;
    filterByMask(prev_points, status);

    auto ages = prev_frame.ages;
    filterByMask(ages, status);
    for (int i = 0; i < ages.size(); ++i) {
        ages[i]++;
    }

    std::vector<double> new_velocities(new_points.size());
    for (int i = 0; i < new_points.size(); ++i) {
        new_velocities[i] = cv::norm(new_points[i] - prev_points[i]);
    }

    filterByMask(new_points, status);
    // Use pixel coordinates for ransac so pix_thresh makes sense
    auto new_points_und = undistortPoints(new_points);
    auto prev_points_und = undistortPoints(prev_points);

    // Reject more outliers with FMAT
    status.clear();
    cv::findFundamentalMat(
        prev_points_und,
        new_points_und,
        cv::FM_RANSAC,
        this->params_.ransac_pix_threshold,
        this->params_.ransac_confidence,
        status);
    
    filterByMask(ids, status);
    filterByMask(new_points, status);
    filterByMask(new_points_und, status);
    filterByMask(ages, status);
    filterByMask(new_velocities, status);


    // Assemble frame
    next_frame.ids = ids;
    next_frame.points = new_points;
    next_frame.points_und = undistortAndNormalizePoints(new_points);
    next_frame.ages = ages;
    next_frame.velocities = new_velocities;
}

std::vector<cv::Point2f>
FeatureTracker::undistortPoints(
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
FeatureTracker::undistortAndNormalizePoints(
    const std::vector<cv::Point2f>& pts)
{
    std::vector<cv::Point2f> out_vec;

    cv::undistortPoints(
        pts,
        out_vec,
        cam_.K,
        cam_.D);
}

} // namespace mft