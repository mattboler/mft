#ifndef MFT_CAMERA_HPP
#define MFT_CAMERA_HPP

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

namespace mft {

struct Camera {
    cv::Mat K;
    cv::Mat D;

    cv::Size focal_length;
    double fx;
    double fy;

    cv::Size principal_point;
    double cx;
    double cy;

    cv::Size image_size;
    size_t width;
    size_t height;
};

Camera buildCamera(std::string config_path);

} // namespace mft

#endif