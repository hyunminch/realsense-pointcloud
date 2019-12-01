#ifndef _EDGE_BASED_REGISTRATION_H
#define _EDGE_BASED_REGISTRATION_H

#include <pcl/features/integral_image_normal.h>
#include "types.hpp"

class EdgeBasedRegistration: public TwoPhaseRegistrationScheme {
//    pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
    pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGB, pcl::Normal, pcl::Label> oed;

    rgb_point_cloud_pointer extract_features(rgb_point_cloud_pointer cloud) {
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
        pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
        ne.setMaxDepthChangeFactor(0.02f);
        ne.setNormalSmoothingSize(10.0f);
        ne.setInputCloud(cloud);
        ne.compute(*normals);

//        // Create the normal estimation class, and pass the input dataset to it
//        ne.setInputCloud(cloud);
//
//        // Create an empty kdtree representation, and pass it to the normal estimation object.
//        // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
//        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
//        ne.setSearchMethod(tree);
//
//        // Output datasets
//        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
//
//        // Use all neighbors in a sphere of radius 3cm
//        ne.setRadiusSearch(0.03);
//
//        // Compute the features
//        ne.compute(*cloud_normals);

        oed.setInputNormals(normals);
        oed.setInputCloud(cloud);
        oed.setDepthDisconThreshold(0.2); // 2cm
        oed.setMaxSearchNeighbors(50);
        oed.setEdgeType(oed.EDGELABEL_NAN_BOUNDARY | oed.EDGELABEL_OCCLUDING | oed.EDGELABEL_OCCLUDED | oed.EDGELABEL_RGB_CANNY | oed.EDGELABEL_HIGH_CURVATURE);
        pcl::PointCloud<pcl::Label> labels;
        std::vector<pcl::PointIndices> label_indices;
        oed.compute(labels, label_indices);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr occluding_edges(new pcl::PointCloud<pcl::PointXYZRGB>),
                occluded_edges(new pcl::PointCloud<pcl::PointXYZRGB>),
                boundary_edges(new pcl::PointCloud<pcl::PointXYZRGB>),
                high_curvature_edges(new pcl::PointCloud<pcl::PointXYZRGB>),
                rgb_edges(new pcl::PointCloud<pcl::PointXYZRGB>);

        pcl::copyPointCloud(*cloud, label_indices[0].indices, *boundary_edges);
        pcl::copyPointCloud(*cloud, label_indices[1].indices, *occluding_edges);
        pcl::copyPointCloud(*cloud, label_indices[2].indices, *occluded_edges);
        pcl::copyPointCloud(*cloud, label_indices[3].indices, *high_curvature_edges);
        pcl::copyPointCloud(*cloud, label_indices[4].indices, *rgb_edges);

        return rgb_edges;
    }

    // Given a vector<pair<feature_cloud, original_cloud>>, compute a global point cloud
    rgb_point_cloud_pointer global_registration(std::vector<std::pair<rgb_point_cloud_pointer, rgb_point_cloud_pointer>>& clouds) {
        pcl::IterativeClosestPoint<rgb_point, rgb_point> icp;
        pcl::registration::CorrespondenceRejectorTrimmed::Ptr cor_rej_trimmed(new pcl::registration::CorrespondenceRejectorTrimmed);
        pcl::ApproximateVoxelGrid<rgb_point> approx_voxel_grid;

        approx_voxel_grid.setLeafSize(0.05, 0.05, 0.05);

        icp.setMaximumIterations(1000);
        icp.setMaxCorrespondenceDistance(0.01);
        icp.setTransformationEpsilon(1e-5);
        icp.setEuclideanFitnessEpsilon(0.50);
        icp.addCorrespondenceRejector(cor_rej_trimmed);

        rgb_point_cloud_pointer target_cloud = clouds[0].first;

        // these cloud pointers are to be used as temporary variables
        rgb_point_cloud_pointer downsized_src(new rgb_point_cloud);
        rgb_point_cloud_pointer downsized_dst(new rgb_point_cloud);

        for (int cloud_idx = 1; cloud_idx < (int)clouds.size(); cloud_idx++) {
            rgb_point_cloud_pointer aligned(new rgb_point_cloud);

            approx_voxel_grid.setInputCloud(clouds[cloud_idx].first);
            approx_voxel_grid.filter(*downsized_src);

            approx_voxel_grid.setInputCloud(target_cloud);
            approx_voxel_grid.filter(*downsized_dst);

            icp.setInputSource(downsized_src);
            icp.setInputTarget(downsized_dst);
            icp.align(*aligned);

            if (icp.hasConverged()) {
                rgb_point_cloud_pointer transformed(new rgb_point_cloud);
                pcl::transformPointCloud(*clouds[cloud_idx].first, *transformed, icp.getFinalTransformation());
                *target_cloud += *transformed;
            }
        }

        return target_cloud;
    }
};

#endif
