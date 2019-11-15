#ifndef _FB_REGISTRATION_SCHEME_H
#define _FB_REGISTRATION_SCHEME_H

#include "types.hpp"

class FBRegistrationScheme: public TwoPhaseRegistrationScheme {
public:
    rgb_point_cloud_pointer extract_features(rgb_point_cloud_pointer cloud) {
        return NULL;
    }

    rgb_point_cloud_pointer global_registration(std::vector<std::pair<rgb_point_cloud_pointer, rgb_point_cloud_pointer>>& clouds) {
        return NULL;
    }
};

#endif