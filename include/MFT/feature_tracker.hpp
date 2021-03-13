#ifndef MFT_FEATURE_TRACKER_HPP
#define MFT_FEATURE_TRACKHER_HPP

#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "MFT/utils.hpp"

namespace mft {

    /**
     * Note: Wanted this to be functional, ended up with an odd 
     * impure-functional style thing
     */

struct FeatureTrackerParams {
    int num_features = 200;
    
    // Detection params
    int init_fast_threshold = 15;
    int min_fast_threshold = 5;
    int minimum_distance = 25;

    // Tracking params
    int window_size = 7;
    int num_levels = 3;

    // Ransac params
    double ransac_pix_threshold = 1.0;
    double ransac_confidence = 0.999;
};

struct Camera {
    cv::Mat K;
    cv::Mat D;

    double fx;
    double fy;

    double cx;
    double cy;

    int width;
    int height;
};

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
    FeatureTracker(std::string path_to_config);

    /**
     * If no previous frames to track features from, build a frame and 
     * detect features in it
     * 
     * Input: Image to build frame from, camera
     * Output: Frame with features detected in it
     */ 
    Frame
    buildFirstFrame(
        const cv::Mat& img,
        const Camera& cam);

    /**
     * If a previous frame exists, track features from it then detect more
     * if needed
     * 
     * Input: Image to build frame from, camera, previous frame
     * Outout: Frame with tracked and detected features
     */ 
    Frame buildNextFrame(
        const cv::Mat& img,
        const Camera& cam,
        const Frame& prev_frame);

    /**
     * Input: Any frame
     * Output: Same frame with new features detected, along with updated info
     */ 
    void
    extractFeatures(
        Frame& f,
        const Camera& cam);

    /**
     * Input: New frame with only img field populated
     * Output: Frame with features tracked from previous frame, no new features
     */ 
    void
    trackFeatures(
        Frame& next_frame,
        const Frame& prev_frame,
        const Camera& cam);
    
    // Undistort points while retaining the existing camera matrix
    std::vector<cv::Point2f>
    undistortPoints(
        const std::vector<cv::Point2f>& pts,
        const Camera& cam);

private:
    uint64_t id_counter_;
    FeatureTrackerParams params_;

    void
    detectFeaturesMask_(
        Frame& f,
        const Camera& cam);
    
    void
    detectFeaturesNoMask_(
        Frame& f,
        const Camera& cam);

    cv::Mat
    buildMask_(
        const cv::Mat& img,
        const std::vector<cv::Point2f>& pts,
        const int radius);
};

} // namespace mft

#endif