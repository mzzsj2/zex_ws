#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include "ICANCmd.h"
#include "location_msgs/RTK.h"

using namespace std;
#define NEWFLAG -100000
void rtkdata_reset(location_msgs::RTK &rtkdata)
{
    rtkdata.gpstime = NEWFLAG;
    rtkdata.longitude = NEWFLAG;
    rtkdata.latitude = NEWFLAG;
    rtkdata.height = NEWFLAG;
    rtkdata.heading = NEWFLAG;
    rtkdata.roll = NEWFLAG;
    rtkdata.pitch = NEWFLAG;
}

bool rtkdata_isNew(const location_msgs::RTK &rtkdata)
{
    return (rtkdata.gpstime != NEWFLAG &&
            rtkdata.latitude != NEWFLAG &&
            rtkdata.longitude != NEWFLAG &&
            rtkdata.height != NEWFLAG &&
            rtkdata.heading != NEWFLAG &&
            rtkdata.pitch != NEWFLAG &&
            rtkdata.roll != NEWFLAG);
}

// 0 - 原码, 1 - 反码
double CAN_decode(CAN_DataFrame raw_data, int lsb, int length, double ratio, double bias, int mode = 0)
{
    int lsb_byte = lsb / 8;
    int lsb_bit = lsb % 8;
    int msb_bit = (lsb_bit + length - 1) % 8;

    int num_byte;
    if ((lsb_bit + length - 1) / 8 == 0)
        num_byte = 1;
    else
        num_byte = 1 + 1 + (length - (8 - lsb_bit) - (msb_bit + 1)) / 8;
    int msb_byte = lsb_byte - num_byte + 1;
    // printf("num_byte : %d ,lsb_byte : %d, lsb_bit : %d, msb_bit : %d \n",num_byte, lsb_byte, lsb_bit, msb_bit);
    int deviation = 0;
    int data = 0;

    if (num_byte == 1)
    {
        int tmp = 0;
        for (int i = lsb_bit; i <= msb_bit; i++)
            tmp += (int)pow(2, i);
        data += (int)(raw_data.arryData[lsb_byte] & tmp);
        data = ratio * data + bias;
        return data;
    }

    for (int byte = lsb_byte; byte > lsb_byte - num_byte; byte--)
    {
        // printf("calculate byte %d : %lf \n", byte, data);
        if (byte == lsb_byte)
        {
            int tmp = 0;
            for (int i = lsb_bit; i <= 7; i++)
                tmp += round(pow(2, i));
            data += (int)((raw_data.arryData[byte] & tmp) >> lsb_bit);
            deviation += (8 - lsb_bit);
        }
        else if (byte == lsb_byte - num_byte + 1)
        {
            int tmp = 0;
            for (int i = 0; i <= msb_bit; i++)
                tmp += round(pow(2, i));
            data += (int)((raw_data.arryData[byte] & tmp) << deviation);
        }
        else
        {
            data += (int)((raw_data.arryData[byte]) << deviation);
            deviation += 8;
        }
        // printf("calculate byte %d : %lf \n", byte, data);
    }

    if (mode == 1)
    {
        int tmp = round(pow(2, msb_bit));
        if ((raw_data.arryData[msb_byte] & tmp) == tmp)
        {
            data = (-1) * (pow(2, length) - data - 1);
        }
    }

    double output = ratio * data + bias;
    return output;
}

DWORD dwDeviceHandle;
CAN_InitConfig config;
int main(int argc, char **argv)
{
    ros::init(argc, argv, "rtk");
    ros::NodeHandle nh;

    if ((dwDeviceHandle = CAN_DeviceOpen(ACUSB_132B, 1, 0)) == 1)
    {
        ROS_INFO_STREAM(" >>open device success!");
    }
    else
    {
        ROS_ERROR_STREAM(" >>open device error!");
        return 0;
        exit(1);
    }

    CAN_InitConfig config;
    config.dwAccCode = 0;
    config.dwAccMask = 0xffffffff;
    config.nFilter = 0;     // 滤波方式(0表示未设置滤波功能,1表示双滤波,2表示单滤波)
    config.bMode = 0;       // 工作模式(0表示正常模式,1表示只听模式)
    config.nBtrType = 1;    // 位定时参数模式(1表示SJA1000,0表示LPC21XX)
    config.dwBtr[0] = 0x00; // BTR0   0014 -1M 0016-800K 001C-500K 011C-250K 031C-12K 041C-100K 091C-50K 181C-20K 311C-10K BFFF-5K
    config.dwBtr[1] = 0x1c; // BTR1
    config.dwBtr[2] = 0;
    config.dwBtr[3] = 0;

    if (CAN_ChannelStart(dwDeviceHandle, 0, &config) != CAN_RESULT_OK)
    {
        ROS_ERROR_STREAM(" >>Init CAN0 error!");

        return 0;
    }

    if (CAN_ChannelStart(dwDeviceHandle, 1, &config) != CAN_RESULT_OK)
    {
        ROS_ERROR_STREAM(" >>Init CAN1 error!");

        return 0;
    }

    int reclen = 0;
    CAN_DataFrame rec[3000]; //buffer
    int CANInd = 0;          //CAN1=0, CAN2=1

    ros::Publisher rtk_pub = nh.advertise<location_msgs::RTK>("rtk_data", 1000);
    ros::Publisher pubOdomGT = nh.advertise<nav_msgs::Odometry>("/odometry_gt", 5);
    ros::Publisher pubPathGT = nh.advertise<nav_msgs::Path>("/path_gt", 5);
    ros::Publisher pubPose = nh.advertise<geometry_msgs::TransformStamped>("/calibration/transform", 5);
    ros::Publisher pubImu = nh.advertise<sensor_msgs::Imu>("/imu_raw", 5);
    nav_msgs::Path pathGT;
    pathGT.header.frame_id = "/imu_init";

    location_msgs::RTK rtkdata;
    rtkdata_reset(rtkdata);

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
    const double torad = M_PI / 180;
    const double g = 9.7964;
    double Re = 0;

    while (ros::ok())
    {   
        ROS_INFO("begin collect can data");
        if ((reclen = CAN_ChannelReceive(dwDeviceHandle, 0, rec, 3000, 10)) > 0) //调用接收函数,得到数据
        {   
            ROS_INFO("begin decode");
            for (int i = 0; i < reclen; i++)
            {
                if (rec[i].uID == 0x320)
                {   
                    rtkdata.gpstime = CAN_decode(rec[i], 40, 32, 1e-3, 0, 0);
                }

                if (rec[i].uID == 0x324)
                {
                    rtkdata.latitude = CAN_decode(rec[i], 24, 32, 1e-7, 0, 0);
                    rtkdata.longitude = CAN_decode(rec[i], 56, 32, 1e-7, 0, 0);
                    //printf("longitude : %f , latitude : %f \n", rtkdata.longitude, rtkdata.latitude);
                }

                if (rec[i].uID == 0x321)
                {
                    rtkdata.AngRateRawX = CAN_decode(rec[i], 20, 20, 1e-2, 0, 1) * torad;
                    rtkdata.AngRateRawY = CAN_decode(rec[i], 32, 20, 1e-2, 0, 1) * torad;
                    rtkdata.AngRateRawZ = CAN_decode(rec[i], 60, 20, 1e-2, 0, 1) * torad;
                }

                if (rec[i].uID == 0x322)
                {
                    rtkdata.AccelRawX = CAN_decode(rec[i], 20, 20, 1e-4, 0, 1) * g;
                    rtkdata.AccelRawY = CAN_decode(rec[i], 32, 20, 1e-4, 0, 1) * g;
                    rtkdata.AccelRawZ = CAN_decode(rec[i], 60, 20, 1e-4, 0, 1) * g;
                }

                if (rec[i].uID == 0x325)
                {
                    rtkdata.height = CAN_decode(rec[i], 24, 32, 1e-3, 0, 0);
                    //printf("height : %f \n", rtkdata.height);
                }

                if (rec[i].uID == 0x32A)
                {
                    rtkdata.heading = CAN_decode(rec[i], 8, 16, 1e-2, 0, 0);
                    rtkdata.pitch = CAN_decode(rec[i], 24, 16, 1e-2, 0, 1);
                    rtkdata.roll = CAN_decode(rec[i], 40, 16, 1e-2, 0, 1);
                    //printf("heading : %f , pitch : %f , roll : %f \n", rtkdata.heading, rtkdata.pitch, rtkdata.roll);
                }

                if (rec[i].uID == 0x323)
                {
                    rtkdata.status = CAN_decode(rec[i], 16, 8, 1, 0, 0);
                }
            }
            ROS_INFO("finished decode");
        }

        if (rtkdata_isNew(rtkdata))
        {   
            ROS_INFO("begin publish");
            rtkdata.stamp = ros::Time::now();
            if (!init)
            {
                R_ENU2IMU = Eigen::AngleAxisd(rtkdata.heading * torad, Eigen::Vector3d::UnitZ()) *
                            Eigen::AngleAxisd(rtkdata.roll * torad, Eigen::Vector3d::UnitY()) *
                            Eigen::AngleAxisd(rtkdata.pitch * torad, Eigen::Vector3d::UnitX());
                Re = R / sqrt(1 - e * e * sin(rtkdata.latitude * torad) * cos(rtkdata.longitude * torad));
                t_ECEF2ENU[0] = (Re + rtkdata.height) * cos(rtkdata.latitude * torad) * cos(rtkdata.longitude * torad);
                t_ECEF2ENU[1] = (Re + rtkdata.height) * cos(rtkdata.latitude * torad) * sin(rtkdata.longitude * torad);
                t_ECEF2ENU[2] = (Re * (1 - e * e) + rtkdata.height) * sin(rtkdata.latitude * torad);
                init = !init;
            }
            R_ENU = Eigen::AngleAxisd(rtkdata.heading * torad, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(rtkdata.roll * torad, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(rtkdata.pitch * torad, Eigen::Vector3d::UnitX());
            R_IMU = R_ENU * R_ENU2IMU.inverse();
            q = R_IMU.inverse();

            t_ECEF[0] = (Re + rtkdata.height) * cos(rtkdata.latitude * torad) * cos(rtkdata.longitude * torad) - t_ECEF2ENU[0];
            t_ECEF[1] = (Re + rtkdata.height) * cos(rtkdata.latitude * torad) * sin(rtkdata.longitude * torad) - t_ECEF2ENU[1];
            t_ECEF[2] = (Re * (1 - e * e) + rtkdata.height) * sin(rtkdata.latitude * torad) - t_ECEF2ENU[2];

            t_ENU[0] = -sin(rtkdata.longitude * torad) * t_ECEF[0] + cos(rtkdata.longitude * torad) * t_ECEF[1];
            t_ENU[1] = -sin(rtkdata.latitude * torad) * cos(rtkdata.longitude * torad) * t_ECEF[0] - sin(rtkdata.latitude * torad) * sin(rtkdata.longitude * torad) * t_ECEF[1] + cos(rtkdata.latitude * torad) * t_ECEF[2];
            t_ENU[2] = cos(rtkdata.latitude * torad) * cos(rtkdata.longitude * torad) * t_ECEF[0] + cos(rtkdata.latitude * torad) * sin(rtkdata.longitude * torad) * t_ECEF[1] + sin(rtkdata.latitude * torad) * t_ECEF[2];

            t = R_ENU2IMU * t_ENU;
            nav_msgs::Odometry odomGT;
            odomGT.header.frame_id = "/imu_init";
            odomGT.child_frame_id = "/rtk_truth";

            odomGT.header.stamp = rtkdata.stamp;
            odomGT.pose.pose.orientation.x = q.x();
            odomGT.pose.pose.orientation.y = q.y();
            odomGT.pose.pose.orientation.z = q.z();
            odomGT.pose.pose.orientation.w = q.w();
            odomGT.pose.pose.position.x = t(0);
            odomGT.pose.pose.position.y = t(1);
            odomGT.pose.pose.position.z = t(2);

            geometry_msgs::TransformStamped trans;
            trans.header = odomGT.header;
            trans.transform.translation.x = t(0);
            trans.transform.translation.y = t(1);
            trans.transform.translation.z = t(2);
            trans.transform.rotation.x = q.x();
            trans.transform.rotation.y = q.y();
            trans.transform.rotation.z = q.z();
            trans.transform.rotation.w = q.w();
            pubPose.publish(trans);

            geometry_msgs::PoseStamped poseGT;
            poseGT.header = odomGT.header;
            poseGT.pose = odomGT.pose.pose;
            pathGT.header.stamp = odomGT.header.stamp;
            pathGT.poses.push_back(poseGT);
            pubPathGT.publish(pathGT);

            odomGT.pose.pose.position.x = 0;
            odomGT.pose.pose.position.y = 0;
            odomGT.pose.pose.position.z = 0;
            pubOdomGT.publish(odomGT);

            rtk_pub.publish(rtkdata);
            rtkdata_reset(rtkdata);
            ROS_INFO("finished publish");
        }

        ROS_INFO("enter next loop");
    }
    CAN_DeviceClose(dwDeviceHandle);
    return 0;
}
