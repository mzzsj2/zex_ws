#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <string>
#include <mutex>
#include <thread>
#include <chrono>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <eigen3/Eigen/Dense>

#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pclomp/ndt_omp.h>

#include <location_msgs/RTK.h>

const double torad = M_PI / 180;

using std::string;
using std::vector;

class location
{
public:
    location(ros::NodeHandle nh) : nh_(nh)
    {
        nh_.param<string>("map_path", map_path_, "");

        nh_.param<string>("odo_topic_name", odo_topic_name_, "");
        nh_.param<string>("lidar_topic_name", lidar_topic_name_, "");
        nh_.param<string>("rtk_topic_name", rtk_topic_name_, "");

        nh_.param<bool>("init_by_rviz", init_by_rviz_, true);

        nh_.param<double>("world/longitude", longitude_world_, 0);
        nh_.param<double>("world/latitude", latitude_world_, 0);
        nh_.param<double>("world/height", height_world_, 0);

        nh_.param<double>("map/longitude", longitude_map_, 0);
        nh_.param<double>("map/latitude", latitude_map_, 0);
        nh_.param<double>("map/height", height_map_, 0);
        nh_.param<double>("map/heading", heading_map_, 0);
        nh_.param<double>("map/pitch", pitch_map_, 0);
        nh_.param<double>("map/roll", roll_map_, 0);

        nh_.param<vector<double>>("extrinsicRot", rot_lidar2imu_, vector<double>());
        nh_.param<vector<double>>("extrinsicTrans", trans_lidar2imu_, vector<double>());

        //lidar to imu
        q_lidar2imu_ = Eigen::AngleAxisd(rot_lidar2imu_[0] * torad, Eigen::Vector3d::UnitZ()) *
                       Eigen::AngleAxisd(rot_lidar2imu_[1] * torad, Eigen::Vector3d::UnitY()) *
                       Eigen::AngleAxisd(rot_lidar2imu_[2] * torad, Eigen::Vector3d::UnitX());

        t_lidar2imu_ = Eigen::Vector3d(trans_lidar2imu_[0], trans_lidar2imu_[1], trans_lidar2imu_[2]);

        //init
        init_pose_ = false;
        location_mode_ = 0;

        pcloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        psourcecloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        ptargetcloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        paligncloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        new_pointcloud_flag_ = false;

        odo_q_ = Eigen::Quaterniond(1, 0, 0, 0);
        odo_t_ = Eigen::Vector3d(0, 0, 0);

        odo_q_last_ = Eigen::Quaterniond(1, 0, 0, 0);
        odo_t_last_ = Eigen::Vector3d(0, 0, 0);

        first_lidar_odo_ = true;

        anh_ndt_.setResolution(1.0);
        anh_ndt_.setMaximumIterations(50);
        anh_ndt_.setStepSize(0.1);
        anh_ndt_.setTransformationEpsilon(0.005);
        anh_ndt_.setNumThreads(4);
        anh_ndt_.setNeighborhoodSearchMethod(pclomp::DIRECT7);

        downsize_filter_25_.setLeafSize(0.25, 0.25, 0.25);

        //gps to local
        Re = R / sqrt(1 - e * e * sin(latitude_world_ * torad) * cos(longitude_world_ * torad));
        t_ecef2enu_[0] = (Re + height_world_) * cos(latitude_world_ * torad) * cos(longitude_world_ * torad);
        t_ecef2enu_[1] = (Re + height_world_) * cos(latitude_world_ * torad) * sin(longitude_world_ * torad);
        t_ecef2enu_[2] = (Re * (1 - e * e) + height_world_) * sin(latitude_world_ * torad);

        //get map entrance
        t_ecef_ = Eigen::Vector3d(0, 0, 0);
        t_ecef_[0] = (Re + height_map_) * cos(latitude_map_ * torad) * cos(longitude_map_ * torad) - t_ecef2enu_[0];
        t_ecef_[1] = (Re + height_map_) * cos(latitude_map_ * torad) * sin(longitude_map_ * torad) - t_ecef2enu_[1];
        t_ecef_[2] = (Re * (1 - e * e) + height_map_) * sin(latitude_map_ * torad) - t_ecef2enu_[2];

        t_map2enu_[0] = -sin(longitude_map_ * torad) * t_ecef_[0] + cos(longitude_map_ * torad) * t_ecef_[1];
        t_map2enu_[1] = -sin(latitude_map_ * torad) * cos(longitude_map_ * torad) * t_ecef_[0] - sin(latitude_map_ * torad) * sin(longitude_map_ * torad) * t_ecef_[1] + cos(latitude_map_ * torad) * t_ecef_[2];
        t_map2enu_[2] = cos(latitude_map_ * torad) * cos(longitude_map_ * torad) * t_ecef_[0] + cos(latitude_map_ * torad) * sin(longitude_map_ * torad) * t_ecef_[1] + sin(latitude_map_ * torad) * t_ecef_[2];

        q_map2enu_ = Eigen::AngleAxisd(-heading_map_ * torad, Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(roll_map_ * torad, Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(pitch_map_ * torad, Eigen::Vector3d::UnitX());

        q_map2enu_ = q_map2enu_ * q_lidar2imu_;

        q_enu2map_ = q_map2enu_.conjugate();
        t_enu2map_ = -1 * (q_enu2map_ * t_map2enu_);

        tf_map_transform_.setOrigin(tf::Vector3(t_map2enu_(0), t_map2enu_(1), t_map2enu_(2)));
        tf_map_q_.setW(q_map2enu_.w());
        tf_map_q_.setX(q_map2enu_.x());
        tf_map_q_.setY(q_map2enu_.y());
        tf_map_q_.setZ(q_map2enu_.z());
        tf_map_transform_.setRotation(tf_map_q_);

        LoadMap();
        anh_ndt_.setInputTarget(ptargetcloud_);

        // pub & sub
        pub_odom_ = nh_.advertise<geometry_msgs::PoseStamped>("/odometry", 5);
        pub_path_ = nh_.advertise<nav_msgs::Path>("/path", 1);
        pub_map_ = nh_.advertise<sensor_msgs::PointCloud2>("/map", 1);

        sub_rtk_ = nh_.subscribe<location_msgs::RTK>(rtk_topic_name_, 1, &location::RtkHandler, this);

        if (init_by_rviz_)
            sub_init_pose_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1, &location::RvizInitPoseHandler, this);

        // sub_pointcloud_ = nh_.subscribe<sensor_msgs::PointCloud2>(lidar_topic_name_, 1, &location::PointCloudHandler, this);
    }

    void RtkHandler(const location_msgs::RTKConstPtr &msg)
    {
        // printf("longitude : %.8lf, latitude : %.8lf \n", msg->longitude, msg->latitude);
        if (location_mode_ == 0)
        {
            odo_q_ = Eigen::AngleAxisd(-msg->heading * torad, Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(msg->roll * torad, Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(msg->pitch * torad, Eigen::Vector3d::UnitX());

            t_ecef_[0] = (Re + msg->height) * cos(msg->latitude * torad) * cos(msg->longitude * torad) - t_ecef2enu_[0];
            t_ecef_[1] = (Re + msg->height) * cos(msg->latitude * torad) * sin(msg->longitude * torad) - t_ecef2enu_[1];
            t_ecef_[2] = (Re * (1 - e * e) + msg->height) * sin(msg->latitude * torad) - t_ecef2enu_[2];

            odo_t_[0] = -sin(msg->longitude * torad) * t_ecef_[0] + cos(msg->longitude * torad) * t_ecef_[1];
            odo_t_[1] = -sin(msg->latitude * torad) * cos(msg->longitude * torad) * t_ecef_[0] - sin(msg->latitude * torad) * sin(msg->longitude * torad) * t_ecef_[1] + cos(msg->latitude * torad) * t_ecef_[2];
            odo_t_[2] = cos(msg->latitude * torad) * cos(msg->longitude * torad) * t_ecef_[0] + cos(msg->latitude * torad) * sin(msg->longitude * torad) * t_ecef_[1] + sin(msg->latitude * torad) * t_ecef_[2];

            msg_odo_time_ = msg->stamp;
            odo_time_ = msg_odo_time_.toSec();
            ROS_INFO("time %.3lf : %.2lf, %.2lf, %.2lf", odo_time_, odo_t_(0), odo_t_(1), odo_t_(2));
            PoseVisulization();
            ChangeMode();
        }
    }

    void ChangeMode()
    {
        if (location_mode_ == 0)
        {
            double dis = sqrt((odo_t_[0] - t_map2enu_[0]) * (odo_t_[0] - t_map2enu_[0]) +
                              (odo_t_[1] - t_map2enu_[1]) * (odo_t_[1] - t_map2enu_[1]));

            if (dis <= 4)
            {
                sub_rtk_.shutdown();
                sub_pointcloud_ = nh_.subscribe<sensor_msgs::PointCloud2>(lidar_topic_name_, 1, &location::PointCloudHandler, this);
                // LoadMap();
                // anh_ndt_.setInputTarget(ptargetcloud_);

                odo_q_ = odo_q_ * q_lidar2imu_;
                odo_q_ = q_enu2map_ * odo_q_;
                odo_t_ = q_enu2map_ * odo_t_ + t_enu2map_;

                // odo_q_ = Eigen::Quaterniond(1, 0, 0, 0);
                // odo_t_ = Eigen::Vector3d(0, 0, 0);

                location_mode_ = 1;
                init_pose_ = true;

                ROS_INFO("lidar location ~");
            }
        }
    }

    void PointCloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        mtx_pointcloud_.lock();
        pcl::fromROSMsg(*msg, *pcloud_);
        new_pointcloud_flag_ = true;
        mtx_pointcloud_.unlock();
    }

    void OdometryHandler()
    {
        ros::Rate rate(100);
        while (ros::ok())
        {
            if (new_pointcloud_flag_)
                OdometryEstimate();
            rate.sleep();
        }
    }

    void OdometryEstimate()
    {
        odo_time_ = pcloud_->header.stamp * 0.000001;
        msg_odo_time_ = msg_odo_time_.fromSec(odo_time_);

        if (!init_pose_)
            return;

        mtx_pointcloud_.lock();
        PointCloudPreprocess(pcloud_, psourcecloud_);
        new_pointcloud_flag_ = false;
        mtx_pointcloud_.unlock();

        pcl::PointCloud<pcl::PointXYZI>::Ptr psourcecloud_ds(new pcl::PointCloud<pcl::PointXYZI>());
        downsize_filter_25_.setInputCloud(psourcecloud_);
        downsize_filter_25_.filter(*psourcecloud_ds);
        anh_ndt_.setInputSource(psourcecloud_ds);

        Eigen::Matrix4f trans(Eigen::Matrix4f::Identity());

        //guess
        Eigen::Vector3d t_guess;
        Eigen::Quaterniond q_guess;
        if (!first_lidar_odo_)
        {
            t_guess = odo_t_ + (odo_t_ - odo_t_last_);
            q_guess = odo_q_ * odo_q_last_.conjugate() * odo_q_;
        }
        else
        {
            t_guess = odo_t_;
            q_guess = odo_q_;
            first_lidar_odo_ = false;
        }

        Eigen::Translation3d init_translation(t_guess(0), t_guess(1), t_guess(2));
        Eigen::AngleAxisd init_rotation(q_guess);
        Eigen::Matrix4d init_guess = (init_translation * init_rotation) * Eigen::Matrix4d::Identity();

        anh_ndt_.align(*paligncloud_, init_guess.cast<float>());

        //result
        trans = anh_ndt_.getFinalTransformation();
        Eigen::Matrix3d R;
        R << trans(0, 0), trans(0, 1), trans(0, 2), trans(1, 0), trans(1, 1), trans(1, 2), trans(2, 0), trans(2, 1), trans(2, 2);
        odo_t_ = Eigen::Vector3d(trans(0, 3), trans(1, 3), trans(2, 3));
        odo_q_ = Eigen::Quaterniond(R);

        odo_t_last_ = odo_t_;
        odo_q_last_ = odo_q_;

        ROS_INFO("time %.3lf : %.2lf, %.2lf, %.2lf", odo_time_, odo_t_(0), odo_t_(1), odo_t_(2));

        PoseVisulization();
    }

    void PointCloudPreprocess(const pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_in,
                              pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_out)
    {
        double x, y, z, s;
        x = y = z = s = 0;

        pcloud_out->clear();
        for (uint32_t i = 0; i < pcloud_in->points.size(); i++)
        {

            x = pcloud_in->points[i].x;
            y = pcloud_in->points[i].y;
            z = pcloud_in->points[i].z;

            s = sqrt(x * x + y * y + z * z);
            if (s <= 80)
                if (!(x > -3 && x < 1 && y > -1 && y < 1))
                    pcloud_out->points.push_back(pcloud_in->points[i]);
        }

        pcloud_out->height = 1;
        pcloud_out->width = pcloud_out->points.size();
    }

    void RvizInitPoseHandler(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
    {
        if (init_pose_)
            return;
        odo_t_(0) = msg->pose.pose.position.x;
        odo_t_(1) = msg->pose.pose.position.y;
        odo_t_(2) = msg->pose.pose.position.z;
        odo_q_.x() = msg->pose.pose.orientation.x;
        odo_q_.y() = msg->pose.pose.orientation.y;
        odo_q_.z() = msg->pose.pose.orientation.z;
        odo_q_.w() = msg->pose.pose.orientation.w;

        odo_t_last_ = odo_t_;
        odo_q_last_ = odo_q_;

        ROS_INFO("receive init pose from rviz ~");

        geometry_msgs::PoseStamped odom;
        odom.pose.orientation.x = odo_q_.x();
        odom.pose.orientation.y = odo_q_.y();
        odom.pose.orientation.z = odo_q_.z();
        odom.pose.orientation.w = odo_q_.w();
        odom.pose.position.x = odo_t_(0);
        odom.pose.position.y = odo_t_(1);
        odom.pose.position.z = odo_t_(2);
        odom.header.frame_id = "/map";
        odom.header.stamp = ros::Time::now();
        pub_odom_.publish(odom);

        init_pose_ = true;
    }

    void LoadMap()
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_path_, *ptargetcloud_) == -1)
        {
            ROS_WARN("can't read open pcd");
            exit(-1);
        }
        else
        {
            ROS_INFO("open pcd file : %s", map_path_.c_str());
        }
    }

    void PoseVisulization()
    {
        if (location_mode_ == 0)
        {
            geometry_msgs::PoseStamped odom;
            odom.pose.orientation.x = odo_q_.x();
            odom.pose.orientation.y = odo_q_.y();
            odom.pose.orientation.z = odo_q_.z();
            odom.pose.orientation.w = odo_q_.w();
            odom.pose.position.x = odo_t_(0);
            odom.pose.position.y = odo_t_(1);
            odom.pose.position.z = odo_t_(2);
            odom.header.frame_id = "/map";
            odom.header.stamp = msg_odo_time_;
            pub_odom_.publish(odom);

            path_.poses.push_back(odom);
            path_.header.frame_id = "/map";
            path_.header.stamp = msg_odo_time_;
            pub_path_.publish(path_);
        }
        else if (location_mode_ == 1)
        {
            geometry_msgs::PoseStamped odom;
            odom.pose.orientation.x = odo_q_.x();
            odom.pose.orientation.y = odo_q_.y();
            odom.pose.orientation.z = odo_q_.z();
            odom.pose.orientation.w = odo_q_.w();
            odom.pose.position.x = odo_t_(0);
            odom.pose.position.y = odo_t_(1);
            odom.pose.position.z = odo_t_(2);
            odom.header.frame_id = "/lidar_map";
            odom.header.stamp = msg_odo_time_;
            pub_odom_.publish(odom);

            Eigen::Vector3d t = odo_t_;
            Eigen::Quaterniond q = odo_q_;
            tf_odo_transform_.setOrigin(tf::Vector3(t(0), t(1), t(2)));
            tf_odo_q_.setW(q.w());
            tf_odo_q_.setX(q.x());
            tf_odo_q_.setY(q.y());
            tf_odo_q_.setZ(q.z());
            tf_odo_transform_.setRotation(tf_odo_q_);
            tf_br_.sendTransform(tf::StampedTransform(tf_odo_transform_, msg_odo_time_, "/lidar_map", "/rslidar"));
            tf_br_.sendTransform(tf::StampedTransform(tf_map_transform_, msg_odo_time_, "/map", "/lidar_map"));
        }
    }

    void MapVisulization()
    {
        ros::Rate rate(1);
        while (ros::ok())
        {
            if (location_mode_ == 1)
            {
                sensor_msgs::PointCloud2 map2;
                pcl::toROSMsg(*ptargetcloud_, map2);
                map2.header.stamp = msg_odo_time_;
                map2.header.frame_id = "/lidar_map";
                pub_map_.publish(map2);
            }
            rate.sleep();
        }
    }

private:
    //const member
    const double e = 0.0818191908425;
    const double R = 6378137;
    const double g = 9.7964;
    double Re;

    ros::NodeHandle nh_;

    string map_path_;
    string odo_topic_name_;
    string lidar_topic_name_;
    string rtk_topic_name_;

    bool init_pose_;
    bool init_by_rviz_;
    int location_mode_; // 0 - rtk, 1 - lidar
    //pointcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr psourcecloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ptargetcloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr paligncloud_;
    bool new_pointcloud_flag_;

    //pose
    Eigen::Quaterniond odo_q_;
    Eigen::Vector3d odo_t_;

    Eigen::Quaterniond odo_q_last_;
    Eigen::Vector3d odo_t_last_;

    bool first_lidar_odo_;

    double odo_time_;

    //ndt
    pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> anh_ndt_;

    //Voxel
    pcl::VoxelGrid<pcl::PointXYZI> downsize_filter_25_;

    //ros sub and pub
    ros::Subscriber sub_pointcloud_;
    ros::Subscriber sub_init_pose_;
    ros::Subscriber sub_rtk_;

    ros::Publisher pub_odom_;
    ros::Publisher pub_path_;
    ros::Publisher pub_map_;

    ros::Time msg_odo_time_;

    nav_msgs::Path path_;

    tf::TransformBroadcaster tf_br_;
    tf::Transform tf_odo_transform_;
    tf::Quaternion tf_odo_q_;

    tf::Transform tf_map_transform_;
    tf::Quaternion tf_map_q_;

    //thread
    std::mutex mtx_pointcloud_;

    //world llh
    double longitude_world_;
    double latitude_world_;
    double height_world_;

    //map llh
    double longitude_map_;
    double latitude_map_;
    double height_map_;
    double heading_map_;
    double pitch_map_;
    double roll_map_;

    //gps to local
    Eigen::Vector3d t_ecef2enu_;
    Eigen::Vector3d t_ecef_;
    Eigen::Vector3d t_map2enu_;
    Eigen::Quaterniond q_map2enu_;

    Eigen::Quaterniond q_enu2map_;
    Eigen::Vector3d t_enu2map_;

    //lidar to imu
    Eigen::Quaterniond q_lidar2imu_;
    Eigen::Vector3d t_lidar2imu_;
    vector<double> rot_lidar2imu_;
    vector<double> trans_lidar2imu_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "location");
    ros::NodeHandle nh;

    location LO(nh);
    std::thread odometry_thread(&location::OdometryHandler, &LO);
    std::thread visulization_thread(&location::MapVisulization, &LO);

    ros::Rate rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }

    odometry_thread.join();
    visulization_thread.join();
}