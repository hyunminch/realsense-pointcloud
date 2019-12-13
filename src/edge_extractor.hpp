#ifndef _EDGE_EXTRACTOR_H
#define _EDGE_EXTRACTOR_H

#include <pcl/features/integral_image_normal.h>
#include "types.hpp"

rgb_point_cloud_pointer extract_edge_features(rgb_point_cloud_pointer cloud) {
    pcl::OrganizedEdgeFromRGBNormals<pcl::PointXYZRGB, pcl::Normal, pcl::Label> oed;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::IntegralImageNormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
    ne.setMaxDepthChangeFactor(0.02f);
    ne.setNormalSmoothingSize(10.0f);
    ne.setInputCloud(cloud);
    ne.compute(*normals);

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

#endif