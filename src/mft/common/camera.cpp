#include "mft/common/camera.hpp"

namespace mft {

Camera buildCamera(std::string path) 
{
    Camera c;
    
    auto n = YAML::LoadFile(path);

    auto s = n["size"];
    c.width = s["width"].as<size_t>();
    c.height = s["height"].as<size_t>();

    c.image_size = cv::Size(c.width, c.height);

    auto i = n["intrinsics"];
    c.fx = i["fx"].as<double>();
    c.fy = i["fy"].as<double>();
    c.cx = i["cx"].as<double>();
    c.cy = i["cy"].as<double>();
    c.K = (cv::Mat1d(3, 3) <<
        c.fx, 0, c.cx,
        0, c.fy, c.cy,
        0, 0, 1);
    
    c.focal_length = cv::Size(c.fx, c.fy);
    c.principal_point = cv::Size(c.cx, c.cy);

    auto d = n["distortion"];
    c.D = (cv::Mat1d(1, 5) << 
        d["k1"].as<double>(),
        d["k2"].as<double>(),
        d["p1"].as<double>(),
        d["p2"].as<double>(),
        d["k3"].as<double>());
    
    return c;
}

} // namespace mft