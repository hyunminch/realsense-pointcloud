#ifndef _FB_REGISTRATION_SCHEME_H
#define _FB_REGISTRATION_SCHEME_H

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>

#include "types.hpp"

class FBRegistrationScheme: public RegistrationScheme {

typedef rgb_point_cloud::ConstPtr rgb_point_cloud_const_pointer;
typedef pcl::PointXYZI i_point;
typedef pcl::PointCloud<i_point> i_point_cloud;
typedef i_point_cloud::Ptr i_point_cloud_pointer;
typedef pcl::FPFHSignature33 feature_type;

public:
    void segmentation(rgb_point_cloud_const_pointer source, rgb_point_cloud_pointer segmented) {
        std::cout << "segmentation..." << std::flush;
        // fit plane and keep points above that plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<rgb_point> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.02);

        seg.setInputCloud(source);
        seg.segment(*inliers, *coefficients);
        
        pcl::ExtractIndices<rgb_point> extract;
        extract.setInputCloud(source);
        extract.setIndices(inliers);
        extract.setNegative(true);

        extract.filter(*segmented);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*segmented, *segmented, indices);
        std::cout << "OK" << std::endl;
        
        std::cout << "clustering..." << std::flush;
        // euclidean clustering
        pcl::search::KdTree<rgb_point>::Ptr tree(new pcl::search::KdTree<rgb_point>);
        tree->setInputCloud(segmented);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<rgb_point> clustering;
        clustering.setClusterTolerance(0.02); // 2cm
        clustering.setMinClusterSize(1000);
        clustering.setMaxClusterSize(250000);
        clustering.setSearchMethod(tree);
        clustering.setInputCloud(segmented);
        clustering.extract(cluster_indices);
        
        // use largest cluster
        if (cluster_indices.size() > 0) {
            std::cout << cluster_indices.size() << " clusters found";
            if (cluster_indices.size() > 1)
                std::cout << " Using largest one...";
            std::cout << std::endl;
            pcl::IndicesPtr indices(new std::vector<int>);
            *indices = cluster_indices[0].indices;
            extract.setInputCloud(segmented);
            extract.setIndices(indices);
            extract.setNegative(false);

            extract.filter(*segmented);
        }
    }

    void detect_keypoints(rgb_point_cloud_pointer input, i_point_cloud_pointer keypoints) {
        pcl::Keypoint<rgb_point, i_point>::Ptr keypoint_detector;

        pcl::HarrisKeypoint3D<rgb_point, i_point>* harris3D = 
            new pcl::HarrisKeypoint3D<rgb_point, i_point>(
                pcl::HarrisKeypoint3D<rgb_point, i_point>::HARRIS
            );

        harris3D->setNonMaxSupression(true);
        harris3D->setRadius(0.1);
        harris3D->setRadiusSearch(0.01);

        keypoint_detector.reset(harris3D);
        harris3D->setMethod(pcl::HarrisKeypoint3D<rgb_point, i_point>::HARRIS);

        keypoint_detector->setInputCloud(input);
        keypoint_detector->setSearchSurface(input);
        keypoint_detector->compute(*keypoints);

        std::cout << "OK. keypoints found: " << keypoints->points.size() << std::endl;
    }

    void extract_descriptors(rgb_point_cloud_const_pointer input, 
                             i_point_cloud_pointer keypoints, 
                             pcl::PointCloud<pcl::FPFHSignature33>::Ptr features) {
        pcl::Feature<rgb_point, feature_type>::Ptr feature_extractor(new pcl::FPFHEstimationOMP<rgb_point, pcl::Normal, feature_type>);
        feature_extractor->setSearchMethod(pcl::search::Search<rgb_point>::Ptr(new pcl::search::KdTree<rgb_point>));
        feature_extractor->setRadiusSearch(0.05);

        rgb_point_cloud_pointer kpts(new rgb_point_cloud);
        kpts->points.resize(keypoints->points.size());

        pcl::copyPointCloud(*keypoints, *kpts);

        pcl::FeatureFromNormals<rgb_point, pcl::Normal, feature_type>::Ptr feature_from_normals = boost::dynamic_pointer_cast<pcl::FeatureFromNormals<rgb_point, pcl::Normal, feature_type>>(feature_extractor);

        feature_extractor->setSearchSurface(input);
        feature_extractor->setInputCloud(kpts);

        if (feature_from_normals) {
            std::cout << "normal estimation..." << std::flush;
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimation<rgb_point, pcl::Normal> normal_estimation;
            normal_estimation.setSearchMethod(pcl::search::Search<rgb_point>::Ptr(new pcl::search::KdTree<rgb_point>));
            normal_estimation.setRadiusSearch(0.01);
            normal_estimation.setInputCloud(input);
            normal_estimation.compute(*normals);
            feature_from_normals->setInputNormals(normals);
            std::cout << "OK" << std::endl;
        }

        std::cout << "descriptor extraction..." << std::flush;
        feature_extractor->compute(*features);
        std::cout << "OK" << std::endl;
    }

    void find_correspondences(pcl::PointCloud<feature_type>::Ptr source, 
                              pcl::PointCloud<feature_type>::Ptr target, 
                              std::vector<int>& correspondences) {
        std::cout << "correspondences assignment..." << std::flush;
        correspondences.resize(source->size());

        // Use a KdTree to search for the nearest matches in feature space
        pcl::KdTreeFLANN<feature_type> descriptor_kdtree;
        // std::cout << "1" << endl;
        descriptor_kdtree.setInputCloud(target);
        // std::cout << "2" << endl;

        // Find the index of the best match for each keypoint, and store it in "correspondences_out"
        const int k = 1;
        std::vector<int> k_indices(k);
        std::vector<float> k_squared_distances(k);
        for (std::size_t i = 0; i < source->size() - 1; i++) {
            // std::cout << "3" << endl;
            descriptor_kdtree.nearestKSearch(*source, i, k, k_indices, k_squared_distances);
            // std::cout << "4" << endl;
            correspondences[i] = k_indices[0];
        }
        std::cout << "OK" << std::endl;
    }

    void filter_correspondences(std::vector<int>& source2target, 
                               std::vector<int>& target2source, 
                               pcl::CorrespondencesPtr correspondences, 
                               i_point_cloud_pointer source_keypoints, 
                               i_point_cloud_pointer target_keypoints) {
        std::cout << "correspondence rejection..." << std::flush;
        std::vector<std::pair<unsigned, unsigned>> correspondences_aux;
        for (unsigned cIdx = 0; cIdx < source2target.size (); ++cIdx)
            if (target2source[source2target[cIdx]] == cIdx)
                correspondences_aux.push_back(std::make_pair(cIdx, source2target[cIdx]));
        
        correspondences->resize(correspondences_aux.size());
        for (unsigned cIdx = 0; cIdx < correspondences_aux.size(); ++cIdx)
        {
            (*correspondences)[cIdx].index_query = correspondences_aux[cIdx].first;
            (*correspondences)[cIdx].index_match = correspondences_aux[cIdx].second;
        }
        
        pcl::registration::CorrespondenceRejectorSampleConsensus<i_point> rejector;
        rejector.setInputSource(source_keypoints);
        rejector.setInputTarget(target_keypoints);
        rejector.setInputCorrespondences(correspondences);
        rejector.getCorrespondences(*correspondences);
        std::cout << "OK" << std::endl;
    }

    void determine_initial_transformation(i_point_cloud_pointer source_keypoints, 
                                        i_point_cloud_pointer target_keypoints, 
                                        pcl::CorrespondencesPtr correspondences, 
                                        Eigen::Matrix4f *initial_transformation_matrix, 
                                        rgb_point_cloud_pointer source_segmented, 
                                        rgb_point_cloud_pointer source_transformed) {
        std::cout << "initial alignment..." << std::flush;
        pcl::registration::TransformationEstimation<i_point, i_point>::Ptr transformation_estimation(new pcl::registration::TransformationEstimationSVD<i_point, i_point>);
        
        transformation_estimation->estimateRigidTransformation (*source_keypoints, *target_keypoints, *correspondences, *initial_transformation_matrix);
        
        pcl::transformPointCloud(*source_segmented, *source_transformed, *initial_transformation_matrix);
        std::cout << "OK" << std::endl;
    }

    void determine_final_transformation(rgb_point_cloud_pointer source_transformed, 
                                      rgb_point_cloud_pointer target_segmented, 
                                      rgb_point_cloud_pointer source_registered, 
                                      Eigen::Matrix4f *transformation_matrix) {
        std::cout << "final registration..." << std::flush;
        pcl::Registration<rgb_point, rgb_point>::Ptr registration (new pcl::IterativeClosestPoint<rgb_point, rgb_point>);
        registration->setInputCloud(source_transformed);
        //registration->setInputCloud(source_segmented_);
        registration->setInputTarget (target_segmented);
        registration->setMaxCorrespondenceDistance(0.05);
        registration->setRANSACOutlierRejectionThreshold (0.05);
        registration->setTransformationEpsilon (0.000001);
        registration->setMaximumIterations (1000);
        registration->align(*source_registered);
        *transformation_matrix = registration->getFinalTransformation();
        std::cout << "OK" << std::endl;
    }

    rgb_point_cloud_pointer register_pair(rgb_point_cloud_pointer source, rgb_point_cloud_pointer target) {
        rgb_point_cloud_pointer source_segmented(new rgb_point_cloud);
        rgb_point_cloud_pointer target_segmented(new rgb_point_cloud);
        segmentation(source, source_segmented);
        segmentation(target, target_segmented);


        i_point_cloud_pointer source_keypoints(new i_point_cloud);
        i_point_cloud_pointer target_keypoints(new i_point_cloud);
        detect_keypoints(source_segmented, source_keypoints);
        detect_keypoints(target_segmented, target_keypoints);

        pcl::PointCloud<feature_type>::Ptr source_features(new pcl::PointCloud<feature_type>);
        pcl::PointCloud<feature_type>::Ptr target_features(new pcl::PointCloud<feature_type>);
        extract_descriptors(source_segmented, source_keypoints, source_features);
        extract_descriptors(target_segmented, target_keypoints, target_features);

        std::vector<int> source2target;
        std::vector<int> target2source;
        // std::cout << "src 2 tgt begin" << endl;
        find_correspondences(source_features, target_features, source2target);
        // std::cout << "src 2 tgt end" << endl;
        // std::cout << "tgt 2 src begin" << endl;
        find_correspondences(target_features, source_features, target2source);
        // std::cout << "tgt 2 src end" << endl;

        pcl::CorrespondencesPtr correspondences(new pcl::Correspondences);
        filter_correspondences(source2target, target2source, correspondences, source_keypoints, target_keypoints);

        Eigen::Matrix4f initial_transformation_matrix;
        rgb_point_cloud_pointer source_transformed(new rgb_point_cloud);
        determine_initial_transformation(source_keypoints, target_keypoints, correspondences, &initial_transformation_matrix, source_segmented, source_transformed);

        Eigen::Matrix4f transformation_matrix;
        rgb_point_cloud_pointer source_registered(new rgb_point_cloud);
        determine_final_transformation(source_transformed, target_segmented, source_registered, &transformation_matrix);

        rgb_point_cloud_pointer source_transformed_final(new rgb_point_cloud);
        pcl::transformPointCloud(*source, *source_transformed_final, transformation_matrix);

        *target += *source_transformed_final;

        return target;
    }

    rgb_point_cloud_pointer registration(std::vector<rgb_point_cloud_pointer>& clouds) {
        rgb_point_cloud_pointer registered_cloud = clouds[0];
        for (int i = 1; i < clouds.size(); i++)
            registered_cloud = register_pair(clouds[i], registered_cloud);

        return registered_cloud;
    }

    // rgb_point_cloud_pointer extract_features(rgb_point_cloud_pointer cloud) {
    //     rgb_point_cloud_pointer segmented(new rgb_point_cloud);
    //     segmentation(cloud, segmented);

    //     i_point_cloud_pointer keypoints(new i_point_cloud_pointer);
    //     detect_keypoints(segmented, keypoints);

    //     pcl::PointCloud<feature_type>::Ptr features(new pcl::PointCloud<feature_type>);
    //     extract_descriptors(segmented, keypoints, features);

    //     return NULL;
    // }

    // rgb_point_cloud_pointer global_registration(std::vector<std::pair<rgb_point_cloud_pointer, rgb_point_cloud_pointer>>& clouds) {
    //     return NULL;
    // }
};

#endif