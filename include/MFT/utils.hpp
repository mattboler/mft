#ifndef MFT_UTILS_HPP
#define MFT_UTILS_HPP

#include <vector>
#include <opencv2/opencv.hpp>

namespace mft {

template <typename T>
std::vector<T> 
filterByMask(
    const std::vector<T>& vec,
    const std::vector<uchar>& mask)
{
    std::vector<T> out_vec;
    out_vec.reserve(vec.size());

    for (size_t i = 0; i < vec.size(); ++i) {
        if (mask.at(i)) {
            out_vec.push_back(vec.at(i));
        }
    }

    return out_vec;
}

}

#endif