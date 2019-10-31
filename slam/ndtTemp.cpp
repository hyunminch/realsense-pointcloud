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


void draw_pointcloud(window& app, state& app_state, pcl::PointCloud<pcl::PointXYZRGB>::Ptr ptr) {
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
		
		glBegin(GL_POINTS);
		for (auto p : ptr->points){
			if (p.z){
				glColor3f(p.r / 255.0, p.g / 255.0, p.b / 255.0);
				glVertex3f(p.x, p.y, p.z);
			}
		}
		glEnd();
    // OpenGL cleanup
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
    glPushMatrix();
}

void register_glft_callbacks(window& app, state& app_state);

int
main (int argc, char** argv)
{
  // Loading first scan of room.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[1], *target_cloud) == -1)
  {
    return (-1);
  }

  // Loading second scan of room from new perspective.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (argv[2], *input_cloud) == -1)
  {
    return (-1);
  }

  // Filtering input scan to roughly 10% of original size to increase speed of registration.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize (0.2, 0.2, 0.2);
  approximate_voxel_filter.setInputCloud (input_cloud);
  approximate_voxel_filter.filter (*filtered_cloud);

  // Initializing Normal Distributions Transform (NDT).
  pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  ndt.setTransformationEpsilon (0.01);
  // Setting maximum step size for More-Thuente line search.
  ndt.setStepSize (0.1);
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
  ndt.setResolution (1.0);

  // Setting max number of registration iterations.
  ndt.setMaximumIterations (35);

  // Setting point cloud to be aligned.
  ndt.setInputSource (filtered_cloud);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget (target_cloud);

  // Set initial alignment estimate found using robot odometry.
  Eigen::AngleAxisf init_rotation (0, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (0, 0, 0);
  Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

  // Calculating required rigid transform to align the input cloud to the target cloud.
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  ndt.align (*output_cloud, init_guess);


  // Transforming unfiltered, input cloud using found transform.
  pcl::transformPointCloud (*input_cloud, *output_cloud, ndt.getFinalTransformation ());

  // Saving transformed input cloud.
  pcl::io::savePCDFileASCII ("../pcdSample/out.pcd", *output_cloud);

	window app(1280, 720, "NDT example");
	state app_state;
	register_glfw_callbacks(app, app_state);
	while(app){
		draw_pointcloud(app, app_state, output_cloud);
	}

  return (0);
}
