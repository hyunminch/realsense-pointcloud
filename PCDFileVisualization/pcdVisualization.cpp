#include <iostream>
#include <thread>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>


using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace std;

pcl::visualization::PCLVisualizer::Ptr visualXYZ (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "show cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "show cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


visualization::PCLVisualizer::Ptr visualXYZRGB(PointCloud<PointXYZRGB>::ConstPtr cloud){
	visualization::PCLVisualizer::Ptr viewer(new visualization::PCLVisualizer("3D viewer"));
	viewer->setBackgroundColor(0,0,0);
	visualization::PointCloudColorHandlerRGBField<PointXYZRGB> rgb(cloud);
	viewer->addPointCloud<PointXYZRGB>(cloud, rgb, "show cloud");
	viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 3, "show cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return(viewer);
}



int main(int argc, char **argv){

	PointCloud<PointXYZRGB>::Ptr sourceCloud(new PointCloud<PointXYZRGB>);
	loadPCDFile<PointXYZRGB>(argv[1], *sourceCloud);
	
	PointCloud<PointXYZRGB>::Ptr pointCloud(new PointCloud<PointXYZRGB>);
	
	sourceCloud->width = (int) sourceCloud->points.size();
	sourceCloud->height = 1;
	NormalEstimation<PointXYZRGB, Normal> ne;
	ne.setInputCloud(sourceCloud);
	cout << "set input cloud"<<endl;
	search::KdTree<PointXYZRGB>::Ptr tree(new search::KdTree<PointXYZRGB>());
	ne.setSearchMethod(tree);
	cout << "set search method"<<endl;
	PointCloud<pcl::Normal>::Ptr cloudNormals1(new PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.05);
	ne.compute(*cloudNormals1);
	cout << "compute"<< endl;
	PointCloud<Normal>::Ptr cloudNormals2(new PointCloud<Normal>);
	ne.setRadiusSearch(0.1);
	ne.compute(*cloudNormals2);

	cout << "compute 2"<<endl;
	visualization::PCLVisualizer::Ptr viewer;
	viewer = visualXYZRGB(sourceCloud);
	cout << "Viewer loading"<<endl;
	while(!viewer->wasStopped()){
		viewer->spinOnce(100);
		this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}
