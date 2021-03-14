#include "mft/ros/feature_tracker_node.hpp"

namespace mft {

FeatureTrackerNode::FeatureTrackerNode(
    ros::NodeHandle& nh) 
    : nh_(nh), it_(nh), is_first_(true) {}

void FeatureTrackerNode::init()
{
    std::string image_topic;
    this->nh_.getParam("image_topic", image_topic);
    ROS_INFO("Image topic: %s", image_topic.c_str());
    this->image_sub_ = this->it_.subscribe(image_topic, 1, &FeatureTrackerNode::imageCallback, this);

    std::string visualization_topic;
    this->nh_.getParam("visualization_topic", visualization_topic);
    ROS_INFO("Visualization topic: %s", visualization_topic.c_str());
    this->vis_pub_ = this->nh_.advertise<sensor_msgs::Image>(visualization_topic, 1);

    std::string camera_path;
    this->nh_.getParam("camera_path", camera_path);
    ROS_INFO("Camera path: %s", camera_path.c_str());
    Camera cam = buildCamera(camera_path);

    std::string tracker_path;
    this->nh_.getParam("tracker_path", tracker_path);
    ROS_INFO("Tracker path: %s", tracker_path.c_str());

    
    this->tracker_ = FeatureTracker(tracker_path, cam);
}

sensor_msgs::PointCloudPtr
FeatureTrackerNode::buildFeatureMsg(
    const Frame& f,
    const std_msgs::Header& h)
{
    sensor_msgs::PointCloudPtr features(new sensor_msgs::PointCloud);
    sensor_msgs::ChannelFloat32 id, age, u, v;

    features->header = h;
    features->header.frame_id = "world";

    for(size_t i = 0; i < f.ids.size(); ++i) {
        auto point_id = f.ids[i];
        auto point_age = f.ages[i];
        auto point_coords = f.points_und[i];

        geometry_msgs::Point32 p;
        p.x = point_coords.x;
        p.y = point_coords.y;
        p.z = 1;

        features->points.push_back(p);
        id.values.push_back(point_id);
        age.values.push_back(point_age);
        u.values.push_back(p.x);
        v.values.push_back(p.y);
    }

    features->channels.push_back(id);
    features->channels.push_back(age);
    features->channels.push_back(u);
    features->channels.push_back(v);

    return features;
}

void FeatureTrackerNode::imageCallback(
    const sensor_msgs::ImageConstPtr& msg)
{
    auto header = msg->header;
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
    cv::Mat img = cv_ptr->image;
    cv::Mat vis;

    Frame f;

    if (is_first_) {
        f = tracker_.buildFirstFrame(img);
        is_first_ = false;
    } else {
        f = tracker_.buildNextFrame(img, prev_frame_);
    }

    auto feature_msg = buildFeatureMsg(f, header);
    feature_pub_.publish(feature_msg);
}

}