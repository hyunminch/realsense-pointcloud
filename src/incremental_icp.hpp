#ifndef _INCREMENTAL_ICP_H
#define _INCREMENTAL_ICP_H

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

/**
 * Incrementally registers a set of clouds, accumulating the point cloud as we go.
 *
 * @param clouds the clouds of concern
 * @return a registered global point cloud, seen as in the perspective of the first given point cloud
 * @author Hyun Min Choi
 */
class IncrementalICP: public RegistrationScheme {
public:
    rgb_point_cloud_pointer registration(std::vector<rgb_point_cloud_pointer>& clouds) {
        pcl::ApproximateVoxelGrid<rgb_point> approx_voxel_grid;
        pcl::IterativeClosestPoint<rgb_point, rgb_point> icp;
        pcl::registration::CorrespondenceRejectorTrimmed::Ptr cor_rej_trimmed(new pcl::registration::CorrespondenceRejectorTrimmed);

//        approx_voxel_grid.setLeafSize(0.05, 0.05, 0.05);

//        icp.setMaximumIterations(30);
//        icp.setMaxCorrespondenceDistance(0.04);
//        icp.setTransformationEpsilon(1e-9);
//        icp.setEuclideanFitnessEpsilon(0.1);
//        icp.addCorrespondenceRejector(cor_rej_trimmed);

        rgb_point_cloud_pointer target_cloud = clouds[0];

        // these cloud pointers are to be used as temporary variables
        rgb_point_cloud_pointer downsized_src(new rgb_point_cloud);
        rgb_point_cloud_pointer downsized_dst(new rgb_point_cloud);

        for (int cloud_idx = 1; cloud_idx < (int)clouds.size(); cloud_idx++) {
            rgb_point_cloud_pointer aligned(new rgb_point_cloud);

            approx_voxel_grid.setInputCloud(clouds[cloud_idx]);
            approx_voxel_grid.filter(*downsized_src);

            icp.setInputSource(downsized_src);
            icp.setInputTarget(target_cloud);
            icp.align(*aligned);

            if (icp.hasConverged()) {
                rgb_point_cloud_pointer transformed(new rgb_point_cloud);
                pcl::transformPointCloud(*clouds[cloud_idx], *transformed, icp.getFinalTransformation());
                *target_cloud += *transformed;
            }
        }

        return target_cloud;
    }
};

#endif