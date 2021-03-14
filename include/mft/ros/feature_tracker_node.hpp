#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>

#include "mft/feature_tracker.hpp"

namespace mft {

class FeatureTrackerNode {
public:
    FeatureTrackerNode(ros::NodeHandle& nh);
    void init();
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    sensor_msgs::PointCloudPtr buildFeatureMsg(
        const Frame& f,
        const std_msgs::Header& h);
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher feature_pub_; // Point cloud
    ros::Publisher vis_pub_; // visualization

    FeatureTracker tracker_;

    Frame prev_frame_;
    bool is_first_;

};

}