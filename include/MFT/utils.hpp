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

} // namespace mft

#endif