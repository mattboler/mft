#ifndef MFT_UTILS_HPP
#define MFT_UTILS_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace mft {

template <typename T>
void
filterByMask(
    std::vector<T>& vec,
    const std::vector<uchar>& mask)
{
    int j = 0;

    for (size_t i = 0; i < vec.size(); ++i) {
        if (mask.at(i)) {
            vec[j++] = vec[i];
        }
    }

    vec.resize(j);
}

template <typename T>
void
filterByMaskBatch(
    std::vector<std::vector<T>>& vecs,
    const std::vector<uchar>& mask)
{
    // Check all vectors are the same length
    size_t size = mask.size();
    for (auto& v : vecs) {
        if (v.size() != size) {
            throw "Vectors are not the same length!";
        } else {
            filterByMask(v, mask);
        }
    }
}

} // namespace mft

#endif