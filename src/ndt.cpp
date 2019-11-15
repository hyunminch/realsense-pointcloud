#include <librealsense2/rs.hpp>
#include "utils.hpp"
#include <iostream>
#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

struct state {
    state() : yaw(0.0), pitch(0.0), last_x(0.0), last_y(0.0),
              ml(false), offset_x(0.0f), offset_y(0.0f) {}
    double yaw, pitch, last_x, last_y; bool ml; float offset_x, offset_y;
};

typedef pcl::PointXYZRGB rgb_cloud;
typedef pcl::PointCloud<rgb_cloud> point_cloud;
typedef point_cloud::Ptr cloud_pointer;
typedef pcl::PointXYZRGBNormal rgb_normal_cloud;
typedef pcl::PointCloud<rgb_normal_cloud> point_cloud_with_normals;

using pcl_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

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


cloud_pointer convert_to_pcl(const rs2::points& points, const rs2::video_frame& color) {
    cloud_pointer cloud(new point_cloud);

    std::tuple<uint8_t, uint8_t, uint8_t> _rgb_texture;

    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width  = static_cast<uint32_t>(sp.width());
    cloud->height = static_cast<uint32_t>(sp.height());
    cloud->is_dense = false;
    cloud->points.resize(points.size());

    auto texture_coordinates = points.get_texture_coordinates();
    auto vertices = points.get_vertices();

    // Iterating through all points and setting XYZ coordinates
    // and RGB values
    for (int i = 0; i < (int)points.size(); i++)
    {
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

    return cloud; // PCL RGB Point Cloud generated
}

std::vector<cloud_pointer> get_clouds(rs2::pipeline pipe, int nr_frames) {
    int _nr_frames = nr_frames;

    std::vector<cloud_pointer> clouds;

    rs2::pointcloud pc;
    rs2::points points;

    while (_nr_frames--) {
        std::cout << "Capturing..." << std::endl;

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
        clouds.push_back(pcl);

        std::cout << "Capture end" << std::endl;

        sleep(2);
    }

    return clouds;
}

void register_glfw_callbacks(window& app, state& app_state) {
    app.on_left_mouse = [&](bool pressed) {
        app_state.ml = pressed;
    };

    app.on_mouse_scroll = [&](double xoffset, double yoffset) {
        app_state.offset_x += static_cast<float>(xoffset);
        app_state.offset_y += static_cast<float>(yoffset);
    };

    app.on_mouse_move = [&](double x, double y) {
        if (app_state.ml) {
            app_state.yaw -= (x - app_state.last_x);
            app_state.yaw = std::max(app_state.yaw, -120.0);
            app_state.yaw = std::min(app_state.yaw, +120.0);
            app_state.pitch += (y - app_state.last_y);
            app_state.pitch = std::max(app_state.pitch, -80.0);
            app_state.pitch = std::min(app_state.pitch, +80.0);
        }
        app_state.last_x = x;
        app_state.last_y = y;
    };

    app.on_key_release = [&](int key) {
        // Escape
        if (key == 32) {
            app_state.yaw = app_state.pitch = 0; app_state.offset_x = app_state.offset_y = 0.0;
        }
    };
}



void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points) {
    // OpenGL commands that prep screen for the pointcloud
    glPopMatrix();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    float width = app.width(), height = app.height();

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(0, 0, +0.5f + app_state.offset_y*0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    glPointSize(width / 640);
    glEnable(GL_TEXTURE_2D);

    int color = 0;

    for (auto&& pc : points) {
        glBegin(GL_POINTS);

        /* this segment actually prints the pointcloud */
        for (auto & p : pc->points) {
            if (p.z) {
                // upload the point and texture coordinates only for points we have depth data for
                glColor3f(p.r / 255.0, p.g / 255.0, p.b / 255.0);
                glVertex3f(p.x, p.y, p.z);
            }
 				}

        glEnd();
    }

    // OpenGL cleanup
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
    glPushMatrix();
}



void register_glft_callbacks(window& app, state& app_state);

void ndtAlign(
        const cloud_pointer cloud_src,
        const cloud_pointer cloud_tgt,
        cloud_pointer output,
        Eigen::Matrix4f &final_transform,
        bool downsample=false) {
    cloud_pointer src(new point_cloud);
    cloud_pointer tgt(new point_cloud);
    pcl::VoxelGrid<rgb_cloud> grid;
    if (downsample) {
        grid.setLeafSize(0.2, 0.2, 0.2);

        grid.setInputCloud(cloud_src);
        grid.filter(*src);

        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
    } else {
        src = cloud_src;
        tgt = cloud_tgt;
    }

		 pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	  pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
  	approximate_voxel_filter.setLeafSize (0.2, 0.2, 0.2);
  	approximate_voxel_filter.setInputCloud (src);
  	approximate_voxel_filter.filter (*filtered_cloud);

	
		pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;
		ndt.setTransformationEpsilon(0.01);
		ndt.setStepSize(0.1);
		ndt.setResolution(1.0);

		ndt.setMaximumIterations(35);
		ndt.setInputSource(src);
		
		ndt.setInputTarget(tgt);
		Eigen::AngleAxisf init_rotation (0, Eigen::Vector3f::UnitZ());
		Eigen::Translation3f init_translation(0,0,0);
		Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();
		ndt.align(*output, init_guess);


    //
    // Transform target back in source frame
    pcl::transformPointCloud (*cloud_tgt, *output, ndt.getFinalTransformation());

    // add the source to the transformed target
    *output += *src;

}


int
main (int argc, char** argv)
{
	int nr_frames = 20;
	rs2::pipeline pipe;
	pipe.start();

	auto clouds = get_clouds(pipe, nr_frames);

	cloud_pointer ndt_result(new point_cloud);
	cloud_pointer sum(new point_cloud);
	Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;
    for (int i = 1; i < nr_frames; i++) {
        cloud_pointer temp(new point_cloud);
        ndtAlign(clouds[i - 1], clouds[i], temp, pairTransform, true);

        //transform current pair into the global transform
        pcl::transformPointCloud (*temp, *ndt_result, GlobalTransform);

        //update the global transform
        GlobalTransform *= pairTransform;

        *sum += *ndt_result;
    }


	window app(1280, 720, "NDT example");
	state app_state;
	register_glfw_callbacks(app, app_state);
	while(app){
		draw_pointcloud(app, app_state, {sum});
	}

  return (0);
}