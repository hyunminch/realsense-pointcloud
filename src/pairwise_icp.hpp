#ifndef _PAIRWISE_ICP_H
#define _PAIRWISE_ICP_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>

#include "types.hpp"

class PairwiseICP: public RegistrationScheme {
public:
    // TODO: implement
    rgb_point_cloud_pointer registration(std::vector<rgb_point_cloud_pointer>& clouds) {
        return nullptr;
    }
};

#endif
