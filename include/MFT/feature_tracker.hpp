#ifndef MFT_FEATURE_TRACKER_HPP
#define MFT_FEATURE_TRACKHER_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace mft {

struct FeatureTrackerParams {
    int num_features = 200;
    
    // Detection params
    int init_fast_threshold = 15;
    int min_fast_threshold = 5;

    // Tracking params
    int window_size = 7;
    int num_levels = 3;
    double min_eig_threshold = 0.003;
    int max_iterations = 30;
    double precision = 0.01;

    // Ransac params
    double ransac_pix_threshold = 1.0;
    double ransac_confidence = 0.999;
};

struct Camera {
    cv::Mat K;
    cv::Mat D;
};

class FeatureTracker {
public:
    FeatureTracker();
    FeatureTracker(std::string path_to_config);

    std::pair<std::vector<uint64_t>, std::vector<cv::Point2f>>
    detectFeatures(
        const cv::Mat& img,
        const std::vector<cv::Point2f>& prev_pts = std::vector<cv::Point2f>());

    std::pair<std::vector<uint64_t>, std::vector<cv::Point2f>>
    trackFeatures(
        const std::vector<cv::Point2f>& prev_pts,
        const cv::Mat& prev_img,
        const cv::Mat& next_img);
    
    std::vector<cv::Point2f>
    undistortPoints(
        const std::vector<cv::Point2f>& pts,
        const Camera& cam);

private:

};

}

#endif