//
// Created by eundolee on 19. 12. 19..
//

#ifndef RS_PCL_BLUR_FILTER_HPP
#define RS_PCL_BLUR_FILTER_HPP

#include <iostream>

#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/filters/extract_indices.h>

#include "types.hpp"

class BlurFilter {
public:
    void filter(rgb_point_cloud_pointer input_cloud) {

        rgb_point_cloud_pointer cloud(new rgb_point_cloud);
        pcl::copyPointCloud(*input_cloud, *cloud);

        input_cloud->width = cloud->width * 3 / 5;
        input_cloud->height = cloud->height * 3 / 5;
        input_cloud->points.resize(input_cloud->width * input_cloud->height);

        int i = 0;
        for (int r = cloud->height / 5; r < cloud->height / 5 * 4; r++){
            for (int c = cloud->width / 5; c < cloud->width / 5 * 4; c++){
                int cloud_index = r * (cloud->width) + c;
                input_cloud->points[i] = cloud->points[cloud_index];

                i++;
            }
        }
    }
};

#endif //RS_PCL_BLUR_FILTER_HPP
