#include <librealsense2/rs.hpp>
#include <vector>

#include "types.hpp"
#include "utils.hpp"

std::tuple<int, int, int> rgb_texture(rs2::video_frame texture, rs2::texture_coordinate texture_coords) {
    // Get Width and Height coordinates of texture
    int width  = texture.get_width();  // Frame width in pixels
    int height = texture.get_height(); // Frame height in pixels

    // Normals to Texture Coordinates conversion
    int x_value = std::min(std::max(int(texture_coords.u * width + .5f), 0), width - 1);
    int y_value = std::min(std::max(int(texture_coords.v * height + .5f), 0), height - 1);

    int bytes = x_value * texture.get_bytes_per_pixel();   // Get # of bytes per pixel
    int strides = y_value * texture.get_stride_in_bytes(); // Get line width in bytes
    int text_index = (bytes + strides);

    const auto new_texture = reinterpret_cast<const uint8_t*>(texture.get_data());

    // RGB components to save in tuple
    int nt1 = new_texture[text_index];
    int nt2 = new_texture[text_index + 1];
    int nt3 = new_texture[text_index + 2];

    return { nt1, nt2, nt3 };
}

rgb_point_cloud_pointer convert_to_pcl(const rs2::points& points, const rs2::video_frame& color) {
    rgb_point_cloud_pointer cloud(new rgb_point_cloud);

    std::tuple<uint8_t, uint8_t, uint8_t> _rgb_texture;

    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width  = static_cast<uint32_t>(sp.width());
    cloud->height = static_cast<uint32_t>(sp.height());
    cloud->is_dense = false;
    cloud->points.resize((int)points.size());

    auto texture_coordinates = points.get_texture_coordinates();
    auto vertices = points.get_vertices();

    // Iterating through all points and setting XYZ coordinates
    // and RGB values
    for (int i = 0; i < (int)points.size(); i++) {
        cloud->points[i].x = vertices[i].x;
        cloud->points[i].y = vertices[i].y;
        cloud->points[i].z = vertices[i].z;

        // Obtain color texture for specific point
        _rgb_texture = rgb_texture(color, texture_coordinates[i]);

        // Mapping Color (BGR due to Camera Model)
        cloud->points[i].r = get<2>(_rgb_texture); // Reference tuple<2>
        cloud->points[i].g = get<1>(_rgb_texture); // Reference tuple<1>
        cloud->points[i].b = get<0>(_rgb_texture); // Reference tuple<0>
    }

    return cloud;
}

/*
 * Filter using two mechanisms
 */
rgb_point_cloud_pointer filter_pcl(rgb_point_cloud_pointer cloud) {
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    rgb_point_cloud_pointer cloud_pass_through(new rgb_point_cloud);
    rgb_point_cloud_pointer cloud_sor(new rgb_point_cloud);
    rgb_point_cloud_pointer cloud_voxel_grid(new rgb_point_cloud);

    // 1. Applies pass through filter
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.filter(*cloud_pass_through);
    pass.setFilterLimits(0.2, 2.5);

    // 2. Applies sor filter
    pcl::StatisticalOutlierRemoval<rgb_point> sor;
    sor.setInputCloud(cloud_pass_through);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.5);
    sor.filter(*cloud_sor);

    pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> voxel_grid;
    voxel_grid.setLeafSize(0.01, 0.01, 0.01);
    voxel_grid.setInputCloud(cloud_sor);
    voxel_grid.filter(*cloud_voxel_grid);

    return cloud_voxel_grid;
}

rgb_point_cloud_pointer get_cloud(rs2::pipeline pipe) {
    rs2::pointcloud pc;
    rs2::points points;
    rs2::spatial_filter spatial_filter;

    spatial_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.25f);

    for (int i = 0; i < 30; i++)
        auto frames = pipe.wait_for_frames();

    auto frames = pipe.wait_for_frames();
    auto color = frames.get_color_frame();

    // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
    if (!color)
        color = frames.get_infrared_frame();

    // Tell pointcloud object to map to this color frame
    pc.map_to(color);

    auto depth = frames.get_depth_frame();
    auto filtered_depth = spatial_filter.process(depth);

    // Generate the pointcloud and texture mappings
    points = pc.calculate(filtered_depth);

    auto pcl = convert_to_pcl(points, color);
//    auto filtered = filter_pcl(pcl);

    return pcl;
}

std::vector<rgb_point_cloud_pointer> get_clouds(rs2::pipeline pipe, int nr_frames) {
    std::vector<rgb_point_cloud_pointer> clouds;

    rs2::pointcloud pc;
    rs2::points points;

    for (int frame = 0; frame < nr_frames; frame++) {
        std::cout << "[RS] Capturing frame [" << frame << "]" << std::endl;

        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        auto color = frames.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frames.get_infrared_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color);

        auto depth = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);

        auto pcl = convert_to_pcl(points, color);
//        auto filtered = filter_pcl(pcl);

        std::cout << "  " << "[RS] Successfully filtered" << std::endl;

        clouds.push_back(pcl);

        std::cout << "[RS] Captured frame [" << frame << "]" << std::endl;

        sleep(2);
    }

    return clouds;
}