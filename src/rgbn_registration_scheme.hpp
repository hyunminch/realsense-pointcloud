#ifndef _RGBN_REGISTRATION_SCHEME_H
#define _RGBN_REGISTRATION_SCHEME_H

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>

#include "types.hpp"

class RGBNRegistrationScheme: public TwoPhaseRegistrationScheme {
public:
    rgb_point_cloud_pointer extract_features(rgb_point_cloud_pointer cloud) {
        return NULL;
    }

    rgb_point_cloud_pointer global_registration(std::vector<std::pair<rgb_point_cloud_pointer, rgb_point_cloud_pointer>>& clouds) {
        return NULL;
    }
};

#endif