#ifndef _TYPES_H
#define _TYPES_H

#include <librealsense2/rs.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

typedef pcl::PointXYZRGB                    rgb_point;
typedef pcl::PointCloud<rgb_point>          rgb_point_cloud;
typedef rgb_point_cloud::Ptr                rgb_point_cloud_pointer;
typedef pcl::PointXYZRGBNormal              rgb_normal_point;
typedef pcl::PointCloud<rgb_normal_point>   rgb_normal_point_cloud;

class RegistrationScheme {
public:
    RegistrationScheme() {}
    ~RegistrationScheme() {}

    virtual rgb_point_cloud_pointer registration(std::vector<rgb_point_cloud_pointer>& clouds) = 0;
};

class TwoPhaseRegistrationScheme: public RegistrationScheme {
public:
    TwoPhaseRegistrationScheme(): RegistrationScheme() {}

    virtual rgb_point_cloud_pointer extract_features(rgb_point_cloud_pointer cloud) = 0;
    // Given a vector<pair<feature_cloud, original_cloud>>, compute a global point cloud
    virtual rgb_point_cloud_pointer global_registration(std::vector<std::pair<rgb_point_cloud_pointer, rgb_point_cloud_pointer>>& clouds) = 0;

    rgb_point_cloud_pointer registration(std::vector<rgb_point_cloud_pointer>& clouds) {
        std::vector<std::pair<rgb_point_cloud_pointer, rgb_point_cloud_pointer>> feature_clouds;

        // Phase 1
        for (auto& cloud: clouds)
            feature_clouds.push_back(std::make_pair(extract_features(cloud), cloud));

        // Phase 2
        return global_registration(feature_clouds); 
    }
};

#endif