#ifndef _RGBN_REGISTRATION_SCHEME_H
#define _RGBN_REGISTRATION_SCHEME_H

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/registration/icp.h>

#include "types.hpp"

class RGBNRegistrationScheme: public TwoPhaseRegistrationScheme {
typedef pcl::Normal normal;
typedef pcl::PointCloud<normal> normal_point_cloud;
typedef normal_point_cloud::Ptr normal_point_cloud_pointer;

public:
    rgb_point_cloud_pointer extract_features(rgb_point_cloud_pointer cloud) {
        normal_point_cloud_pointer ncloud_ptr(new normal_point_cloud);
        pcl::IntegralImageNormalEstimation<rgb_point, normal> ne;

        ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
        ne.setNormalSmoothingSize(1.0f);
        ne.setBorderPolicy(ne.BORDER_POLICY_MIRROR);
        ne.setInputCloud(cloud);
        ne.compute(*ncloud_ptr);

        pcl::OrganizedEdgeFromRGBNormals<rgb_point, normal, pcl::Label> oed;
        // Parameters below need some tweaking based on experiments.
        oed.setInputNormals(ncloud_ptr);
        oed.setInputCloud(cloud);
        oed.setDepthDisconThreshold(0.01);
        oed.setMaxSearchNeighbors(50);

        oed.setEdgeType(oed.EDGELABEL_NAN_BOUNDARY | oed.EDGELABEL_OCCLUDING | oed.EDGELABEL_OCCLUDED | oed.EDGELABEL_HIGH_CURVATURE | oed.EDGELABEL_RGB_CANNY);

        pcl::PointCloud<pcl::Label> labels;
        std::vector<pcl::PointIndices> label_indices;
        oed.compute(labels, label_indices);

        rgb_point_cloud_pointer rgb_edges(new rgb_point_cloud);
        pcl::copyPointCloud(*cloud, label_indices[4].indices, *rgb_edges);

        return rgb_edges;
    }

    rgb_point_cloud_pointer global_registration(std::vector<std::pair<rgb_point_cloud_pointer, rgb_point_cloud_pointer>>& clouds) {
        pcl::IterativeClosestPoint<rgb_point, rgb_point> icp;
        // for (int i = 0; i < clouds.size(); i++) {
            
        // }
        return NULL;
    }
};

#endif