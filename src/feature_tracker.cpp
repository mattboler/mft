#include "MFT/feature_tracker.hpp"

namespace mft {

std::vector<cv::Point2f>
FeatureTracker::detectFeatures(
    const cv::Mat& img,
    const std::vector<cv::Point2f>& prev_pts)
{

}

std::vector<uint64_t>
FeatureTracker::assignFeatureIds(
    const std::vector<cv::Point2f>& pts,
    const uint64_t last_id)
{
    std::vector<uint64_t> v(pts.size());
    std::iota(std::begin(v), std::end(v), last_id);
    return v;
}

} // namespace mft