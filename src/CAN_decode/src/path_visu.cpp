#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include "can_decode/rtk.h"

using namespace std;

bool init = 0;
Eigen::Matrix3d R_ENU2IMU;
Eigen::Matrix3d R_ENU;
Eigen::Matrix3d R_IMU;
Eigen::Vector3d t_ECEF2ENU(0, 0, 0);
Eigen::Vector3d t_ECEF(0, 0, 0);
Eigen::Vector3d t_ENU(0, 0, 0);
Eigen::Vector3d t(0, 0, 0);
Eigen::Quaternion<double> q(1, 0, 0, 0);
const double e = 0.0818191908425;
const double R = 6378137;
const double torad = M_PI/180;
double Re = 0;

nav_msgs::Path pathGT;

ros::Publisher rtk_pub;
ros::Publisher pubOdomGT;
ros::Publisher pubPathGT;

void rtk_callback(const boost::shared_ptr<const can_decode::rtk> &input_msg)
{
    if (!init)
    {
        R_ENU2IMU = Eigen::AngleAxisd(input_msg->heading * torad, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(input_msg->roll * torad, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(input_msg->pitch * torad, Eigen::Vector3d::UnitX());
        Re = R / sqrt(1 - e * e * sin(input_msg->latitude * torad) * cos(input_msg->longitude * torad));
        t_ECEF2ENU[0] = (Re + input_msg->height) * cos(input_msg->latitude * torad) * cos(input_msg->longitude * torad);
        t_ECEF2ENU[1] = (Re + input_msg->height) * cos(input_msg->latitude * torad) * sin(input_msg->longitude * torad);
        t_ECEF2ENU[2] = (Re * (1 - e * e) + input_msg->height) * sin(input_msg->latitude * torad);
        init = !init;
    }
    R_ENU = Eigen::AngleAxisd(input_msg->heading * torad, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(input_msg->roll * torad, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(input_msg->pitch * torad, Eigen::Vector3d::UnitX());
    R_IMU = R_ENU * R_ENU2IMU.inverse();
    q = R_IMU.inverse();

    t_ECEF[0] = (Re + input_msg->height) * cos(input_msg->latitude * torad) * cos(input_msg->longitude * torad) - t_ECEF2ENU[0];
    t_ECEF[1] = (Re + input_msg->height) * cos(input_msg->latitude * torad) * sin(input_msg->longitude * torad) - t_ECEF2ENU[1];
    t_ECEF[2] = (Re * (1 - e * e) + input_msg->height) * sin(input_msg->latitude * torad) - t_ECEF2ENU[2];

    t_ENU[0] = - sin(input_msg->longitude * torad) * t_ECEF[0] + cos(input_msg->longitude * torad) * t_ECEF[1];
    t_ENU[1] = - sin(input_msg->latitude * torad) * cos(input_msg->longitude * torad) * t_ECEF[0] 
               - sin(input_msg->latitude * torad) * sin(input_msg->longitude * torad) * t_ECEF[1]
               + cos(input_msg->latitude * torad) * t_ECEF[2];
    t_ENU[2] =   cos(input_msg->latitude * torad) * cos(input_msg->longitude * torad) * t_ECEF[0] 
               + cos(input_msg->latitude * torad) * sin(input_msg->longitude * torad) * t_ECEF[1]
               + sin(input_msg->latitude * torad) * t_ECEF[2];

    t = R_ENU2IMU * t_ENU;
    nav_msgs::Odometry odomGT;
    odomGT.header.frame_id = "/imu_init";
    odomGT.child_frame_id = "/rtk_truth";

    odomGT.header.stamp = ros::Time::now();
    odomGT.pose.pose.orientation.x = q.x();
    odomGT.pose.pose.orientation.y = q.y();
    odomGT.pose.pose.orientation.z = q.z();
    odomGT.pose.pose.orientation.w = q.w();
    odomGT.pose.pose.position.x = t(0);
    odomGT.pose.pose.position.y = t(1);
    odomGT.pose.pose.position.z = t(2);
    pubOdomGT.publish(odomGT);

    geometry_msgs::PoseStamped poseGT;
    poseGT.header = odomGT.header;
    poseGT.pose = odomGT.pose.pose;
    pathGT.header.frame_id = "/imu_init";
    pathGT.header.stamp = odomGT.header.stamp;
    pathGT.poses.push_back(poseGT);
    pubPathGT.publish(pathGT);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "path_visu");
    ros::NodeHandle nh;
    pubOdomGT = nh.advertise<nav_msgs::Odometry>("/odometry_gt", 5);
    pubPathGT = nh.advertise<nav_msgs::Path>("/path_gt", 5);

    ros::Subscriber subrtkdata = nh.subscribe<can_decode::rtk>("/rtk_data", 100, rtk_callback);
    ros::spin();
    return 0;
}
