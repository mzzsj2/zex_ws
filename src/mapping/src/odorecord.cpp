#include <location_msgs/RTK.h>
#include <fstream>
#include <string>
#include <ros/ros.h>
#include <eigen3/Eigen/Dense>

#include <unistd.h>
using namespace std;

class rtk_pose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    rtk_pose()
    {
        e_ = 0.0818191908425;
        R_ = 6378137;
        torad_ = M_PI / 180;
        Re_ = 0;

        init_ = false;
    }

    void rtk_data_in(const location_msgs::RTK rtk)
    {
        longitude_.push_back(rtk.longitude);
        latitude_.push_back(rtk.latitude);

        Eigen::Vector3d t(0, 0, 0);
        Eigen::Quaternion<double> q(1, 0, 0, 0);

        if (!init_)
        {
            R_ENU2IMU_ = Eigen::AngleAxisd(rtk.heading * torad_, Eigen::Vector3d::UnitZ()) *
                         Eigen::AngleAxisd(rtk.roll * torad_, Eigen::Vector3d::UnitY()) *
                         Eigen::AngleAxisd(rtk.pitch * torad_, Eigen::Vector3d::UnitX());
            Re_ = R_ / sqrt(1 - e_ * e_ * sin(rtk.latitude * torad_) * cos(rtk.longitude * torad_));
            t_ECEF2ENU_[0] = (Re_ + rtk.height) * cos(rtk.latitude * torad_) * cos(rtk.longitude * torad_);
            t_ECEF2ENU_[1] = (Re_ + rtk.height) * cos(rtk.latitude * torad_) * sin(rtk.longitude * torad_);
            t_ECEF2ENU_[2] = (Re_ * (1 - e_ * e_) + rtk.height) * sin(rtk.latitude * torad_);

            map_q_.push_back(q);
            map_t_.push_back(t);
            map_time_.push_back(rtk.stamp.toSec());

            init_ = true;
        }
        else
        {
            Eigen::Matrix3d R_ENU;
            Eigen::Matrix3d R_IMU;

            R_ENU = Eigen::AngleAxisd(rtk.heading * torad_, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(rtk.roll * torad_, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(rtk.pitch * torad_, Eigen::Vector3d::UnitX());
            R_IMU = R_ENU * R_ENU2IMU_.inverse();

            q = R_IMU.inverse();

            Eigen::Vector3d t_ECEF(0, 0, 0);
            Eigen::Vector3d t_ENU(0, 0, 0);

            t_ECEF[0] = (Re_ + rtk.height) * cos(rtk.latitude * torad_) * cos(rtk.longitude * torad_) - t_ECEF2ENU_[0];
            t_ECEF[1] = (Re_ + rtk.height) * cos(rtk.latitude * torad_) * sin(rtk.longitude * torad_) - t_ECEF2ENU_[1];
            t_ECEF[2] = (Re_ * (1 - e_ * e_) + rtk.height) * sin(rtk.latitude * torad_) - t_ECEF2ENU_[2];

            t_ENU[0] = -sin(rtk.longitude * torad_) * t_ECEF[0] + cos(rtk.longitude * torad_) * t_ECEF[1];
            t_ENU[1] = -sin(rtk.latitude * torad_) * cos(rtk.longitude * torad_) * t_ECEF[0] - sin(rtk.latitude * torad_) * sin(rtk.longitude * torad_) * t_ECEF[1] + cos(rtk.latitude * torad_) * t_ECEF[2];
            t_ENU[2] = cos(rtk.latitude * torad_) * cos(rtk.longitude * torad_) * t_ECEF[0] + cos(rtk.latitude * torad_) * sin(rtk.longitude * torad_) * t_ECEF[1] + sin(rtk.latitude * torad_) * t_ECEF[2];

            t = R_ENU2IMU_ * t_ENU;

            map_q_.push_back(q);
            map_t_.push_back(t);
            map_time_.push_back(rtk.stamp.toSec());
        }
    }

    void saveTUMformat()
    {
        char end1 = 0x0d; // "/n"
        char end2 = 0x0a;

        // rtk odometry
        FILE *fp = NULL;

        string rtk_tum_file = "/home/zsj/dataset/0917/rtk.txt";
        fp = fopen(rtk_tum_file.c_str(), "w+");

        if (fp == NULL)
        {
            printf("fail to open file (rtk odometry file) ! \n");
            exit(1);
        }
        else
            printf("TUM : write rtk data to %s \n", rtk_tum_file.c_str());

        for (int i = 0; i < map_q_.size(); i++)
        {
            Eigen::Quaterniond q = map_q_[i];
            Eigen::Vector3d t = map_t_[i];
            double time = map_time_[i];
            fprintf(fp, "%.3lf %.3lf %.3lf %.3lf %.5lf %.5lf %.5lf %.5lf%c",
                    time, t(0), t(1), t(2),
                    q.x(), q.y(), q.z(), q.w(), end2);
        }

        fclose(fp);
    }

    // bool get_pose(double time, Eigen::Quaternion<double> &q, Eigen::Vector3d &t);

    vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> map_q_;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> map_t_;
    vector<double> map_time_;

    vector<double> longitude_;
    vector<double> latitude_;

private:
    bool init_;
    Eigen::Matrix3d R_ENU2IMU_;
    Eigen::Vector3d t_ECEF2ENU_;

    double e_;
    double R_;
    double torad_;
    double Re_;
};

rtk_pose rtk;

void OdmrecordHandle(const location_msgs::RTK::ConstPtr &msg)
{
    rtk.rtk_data_in(*msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "recordresult");
    ros::NodeHandle nh;

    ros::Subscriber Odmrecord = nh.subscribe<location_msgs::RTK>("/rtk_data", 100, OdmrecordHandle);

    ros::spin();
    rtk.saveTUMformat();
    return 0;
}