#ifndef MFT_FEATURE_TRACKER_HPP
#define MFT_FEATURE_TRACKHER_HPP

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "mft/utils.hpp"

#include "mft/common/camera.hpp"

namespace mft {

    /**
     * Note: Wanted this to be functional, ended up with an odd 
     * impure-functional style thing
     */

struct FeatureTrackerParams {
    // --- High-level parameters ---
    // How many feature are tracked at once?
    int num_features = 250; 
    // Enhance contrast before using images
    bool use_clahe = false; 
    
    // --- Detection params ---
    // Force points to be this far apart (pix)
    int minimum_distance = 15;
    // Do we perform subpixel optimization?
    bool use_subpix = false;

    // --- Tracking params ---
    // KLT window size
    int window_size = 7;
    // KLT pyramid levels
    int num_levels = 3;

    // --- Ransac params ---
    // Distance to be considered an outlier
    double ransac_pix_threshold = 1.0;
    // Confidence level of model fit
    double ransac_confidence = 0.999;
};

FeatureTrackerParams buildTrackerParams(std::string config_path);



struct Frame {
    cv::Mat img;
    std::vector<uint64_t> ids;
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> points_und;
    std::vector<uint64_t> ages;
    std::vector<double> velocities;
};

class FeatureTracker {
public:
    FeatureTracker();
    FeatureTracker(
        FeatureTrackerParams params,
        Camera cam);

    /**
     * If no previous frames to track features from, build a frame and 
     * detect features in it
     * 
     * Input: Image to build frame from, camera
     * Output: Frame with features detected in it
     */ 
    Frame
    buildFirstFrame(
        const cv::Mat& img);

    /**
     * If a previous frame exists, track features from it then detect more
     * if needed
     * 
     * Input: Image to build frame from, camera, previous frame
     * Outout: Frame with tracked and detected features
     */ 
    Frame buildNextFrame(
        const cv::Mat& img,
        const Frame& prev_frame);
    
    /**
     * Draw frame with features
     */ 
    cv::Mat
    annotateFrame(
        const Frame& f);
    

private:
    uint64_t id_counter_;
    FeatureTrackerParams params_;
    Camera cam_;

    /**
     * Input: Any frame
     * Output: Same frame with new features detected, along with updated info
     */ 
    void
    extractFeatures_(
        Frame& f);

    /**
     * Input: New frame with only img field populated
     * Output: Frame with features tracked from previous frame, no new features
     */ 
    void
    trackFeatures_(
        Frame& next_frame,
        const Frame& prev_frame);
    
    // Undistort points while retaining the existing camera matrix
    std::vector<cv::Point2f>
    undistortPoints_(
        const std::vector<cv::Point2f>& pts);
    // Undistort points and reframe with identity camera matrix
    std::vector<cv::Point2f>
    undistortAndNormalizePoints_(
        const std::vector<cv::Point2f>& pts);

    void
    detectFeaturesMask_(
        Frame& f);
    
    void
    detectFeaturesNoMask_(
        Frame& f);

    cv::Mat
    buildMask_(
        const cv::Mat& img,
        const std::vector<cv::Point2f>& pts,
        const int radius);
};

} // namespace mft

#endif