#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <string>
#include <vector>
#include <deque>

#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <eigen3/Eigen/Dense>

#include <ndt_cpu/NormalDistributionsTransform.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

using std::deque;
using std::string;
using std::vector;
using namespace gtsam;
#define foreach BOOST_FOREACH

class mapping
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    mapping(ros::NodeHandle nh) : nh_(nh)
    {
        //load params from launch
        nh_.getParam("bag_path", bag_path_);
        nh_.getParam("map_path", map_path_);
        nh_.getParam("map_dist", map_dist_);
        nh_.getParam("save_map", save_map_);

        dist_ = map_dist_ + 0.01;
        keyframe_flag_ = false;
        lateset_frame_id_ = -1;

        //manul params
        surrounding_keyposes_search_radius_ = 10;
        recent_keyframes_num_ = 10;

        //params init
        pcloud_raw_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pcloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        psourcecloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        ptargetcloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        ptargetcloud_ds_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pkeyposes_3d_.reset(new pcl::PointCloud<pcl::PointXYZI>());

        kdtree_surrounding_keyposes_.reset(new pcl::KdTreeFLANN<pcl::PointXYZI>());

        odo_q_ = Eigen::Quaterniond(1, 0, 0, 0);
        odo_t_ = Eigen::Vector3d(0, 0, 0);

        odo_q_last_ = Eigen::Quaterniond(1, 0, 0, 0);
        odo_t_last_ = Eigen::Vector3d(0, 0, 0);

        //ndt
        anh_ndt_.setResolution(1.0);
        anh_ndt_.setMaximumIterations(50);
        anh_ndt_.setStepSize(0.1);
        anh_ndt_.setTransformationEpsilon(0.005);

        //voxel grid
        downsize_filter_25_.setLeafSize(0.25, 0.25, 0.25);
        downsize_filter_50_.setLeafSize(0.5, 0.5, 0.5);

        //gtsam
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam_ = new ISAM2(parameters);

        gtsam::Vector v1(6);
        v1 << 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3; // rotation xyz
        odometry_noise_ = noiseModel::Diagonal::Variances(v1);

        gtsam::Vector v2(6);
        v2 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
        prior_noise_ = noiseModel::Diagonal::Variances(v2);

        pub_pointcloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/map", 2);
        pub_targetcloud_ = nh.advertise<sensor_msgs::PointCloud2>("/targetcloud", 2);
        pub_odom_ = nh.advertise<geometry_msgs::PoseStamped>("/odometry", 5);
        pub_path_ = nh.advertise<nav_msgs::Path>("/path", 1);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr TransformPointCloud(const Eigen::Quaterniond &q, const Eigen::Vector3d &t, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>());
        cloud_out->resize(cloud_in->points.size());

        for (uint32_t i = 0; i < cloud_in->points.size(); i++)
        {
            Eigen::Vector3d point(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z);
            Eigen::Vector3d point2 = q * point + t;

            pcl::PointXYZI p;
            p.x = point2[0];
            p.y = point2[1];
            p.z = point2[2];
            p.intensity = cloud_in->points[i].intensity;

            cloud_out->points[i] = p;
        }

        cloud_out->height = 1;
        cloud_out->width = cloud_in->points.size();

        return cloud_out;
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

    void OdoEstimate()
    {
        if (!pkeyposes_3d_->points.empty())
        {
            Eigen::Matrix4f trans(Eigen::Matrix4f::Identity());

            //guess
            Eigen::Vector3d t_guess = odo_t_ + (odo_t_ - odo_t_last_);
            Eigen::Quaterniond q_guess = odo_q_ * odo_q_last_.conjugate() * odo_q_;
            Eigen::Translation3d init_translation(t_guess(0), t_guess(1), t_guess(2));
            Eigen::AngleAxisd init_rotation(q_guess);
            Eigen::Matrix4d init_guess = (init_translation * init_rotation) * Eigen::Matrix4d::Identity();

            //pointcloud downsample
            downsize_filter_25_.setInputCloud(pcloud_);
            downsize_filter_25_.filter(*psourcecloud_);

            anh_ndt_.setInputSource(psourcecloud_);
            anh_ndt_.align(init_guess.cast<float>());

            //result
            trans = anh_ndt_.getFinalTransformation();
            Eigen::Matrix3d R;
            R << trans(0, 0), trans(0, 1), trans(0, 2), trans(1, 0), trans(1, 1), trans(1, 2), trans(2, 0), trans(2, 1), trans(2, 2);
            odo_t_ = Eigen::Vector3d(trans(0, 3), trans(1, 3), trans(2, 3));
            odo_q_ = Eigen::Quaterniond(R);

            dist_ += sqrt((odo_t_(0) - odo_t_last_(0)) * (odo_t_(0) - odo_t_last_(0)) + (odo_t_(1) - odo_t_last_(1)) * (odo_t_(1) - odo_t_last_(1)));

            odo_t_last_ = odo_t_;
            odo_q_last_ = odo_q_;
        }
    }

    void SaveKeyFrame()
    {
        if (dist_ < map_dist_)
        {
            keyframe_flag_ = false;
            return;
        }

        dist_ -= map_dist_;
        if (pkeyposes_3d_->points.empty())
        {
            gtsam_graph_.add(PriorFactor<Pose3>(0, Pose3(Rot3(odo_q_), Point3(odo_t_)), prior_noise_));
            initial_estimate_.insert(0, Pose3(Rot3(odo_q_), Point3(odo_t_)));
        }
        else
        {
            Eigen::Quaterniond q_last = keyposes_q_[keyposes_q_.size() - 1];
            Eigen::Vector3d t_last = keyposes_t_[keyposes_t_.size() - 1];

            gtsam::Pose3 pose_from = Pose3(Rot3(q_last), Point3(t_last));
            gtsam::Pose3 pose_to = Pose3(Rot3(odo_q_), Point3(odo_t_));

            gtsam_graph_.add(BetweenFactor<Pose3>(pkeyposes_3d_->points.size() - 1, pkeyposes_3d_->points.size(), pose_from.between(pose_to), odometry_noise_));
            initial_estimate_.insert(pkeyposes_3d_->points.size(), pose_to);
        }

        //update isam
        isam_->update(gtsam_graph_, initial_estimate_);
        isam_->update();

        gtsam_graph_.resize(0);
        initial_estimate_.clear();

        //result
        Pose3 lastest_estimate;
        isam_current_estimate_ = isam_->calculateEstimate();
        lastest_estimate = isam_current_estimate_.at<Pose3>(isam_current_estimate_.size() - 1);
        odo_q_ = lastest_estimate.rotation().matrix();
        odo_t_ = Eigen::Vector3d(lastest_estimate.translation().x(), lastest_estimate.translation().y(), lastest_estimate.translation().z());

        keyposes_q_.push_back(odo_q_);
        keyposes_t_.push_back(odo_t_);
        keyposes_time_.push_back(odo_time_);
        current_pose_.x = odo_t_(0);
        current_pose_.y = odo_t_(1);
        current_pose_.z = odo_t_(2);
        current_pose_.intensity = keyposes_q_.size() - 1;
        pkeyposes_3d_->points.push_back(current_pose_);

        pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_keyframe(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::copyPointCloud(*pcloud_, *pcloud_keyframe);
        pcloud_keyframes_.push_back(pcloud_keyframe);
        keyframe_flag_ = true;

        ROS_INFO("dist : %.3lf", dist_);
        ROS_INFO("odo_t: %.2lf, %.2lf, %.2lf", odo_t_(0), odo_t_(1), odo_t_(2));
        ROS_INFO("save keyframe %ld !", keyposes_q_.size() - 1);
    }

    void DetectLoopClosure()
    {
        if (!keyframe_flag_)
            return;

        vector<int> point_search_id;
        vector<float> point_search_dis;

        kdtree_surrounding_keyposes_->setInputCloud(pkeyposes_3d_);
        pcl::PointXYZI search_pose;
        search_pose.x = odo_t_(0);
        search_pose.y = odo_t_(1);
        search_pose.z = odo_t_(2);
        kdtree_surrounding_keyposes_->radiusSearch(search_pose, surrounding_keyposes_search_radius_, point_search_id, point_search_dis, 0);

        int current_frame_id = keyposes_time_.size() - 1;
        int ls_frame_id = current_frame_id;
        for (int i = 0; i < point_search_id.size(); i++)
        {
            int id = point_search_id[i];
            if (abs(keyposes_time_[id] - odo_time_) <= 60)
                continue;
            ls_frame_id = id < ls_frame_id ? id : ls_frame_id;
        }
        if (ls_frame_id == current_frame_id)
            return;

        ROS_INFO("Find Loopclousre between id  %d and %d", current_frame_id, ls_frame_id);
        pcl::PointCloud<pcl::PointXYZI>::Ptr ls_targetcloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr ls_targetcloud_ds(new pcl::PointCloud<pcl::PointXYZI>());
        for (int i = -recent_keyframes_num_ / 2; i <= recent_keyframes_num_ / 2; i++)
        {
            int id = ls_frame_id + i;
            if (id < 0 || id > current_frame_id)
                continue;
            *ls_targetcloud += *TransformPointCloud(keyposes_q_[id], keyposes_t_[id], pcloud_keyframes_[id]);
        }

        downsize_filter_25_.setInputCloud(ls_targetcloud);
        downsize_filter_25_.filter(*ls_targetcloud_ds);
        pcl::PointCloud<pcl::PointXYZI>::Ptr ls_sourcecloud(new pcl::PointCloud<pcl::PointXYZI>());
        ls_sourcecloud = TransformPointCloud(keyposes_q_[current_frame_id], keyposes_t_[current_frame_id], pcloud_keyframes_[current_frame_id]);

        cpu::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
        ndt.setResolution(1);
        ndt.setMaximumIterations(500);
        ndt.setStepSize(0.1);
        ndt.setTransformationEpsilon(0.005);
        ndt.setInputSource(ls_sourcecloud);
        ndt.setInputTarget(ls_targetcloud);

        Eigen::Matrix4f trans(Eigen::Matrix4f::Identity());
        Eigen::Matrix4d init_guess(Eigen::Matrix4d::Identity());
        ndt.align(init_guess.cast<float>());

        //ndt result
        double fitness_score = ndt.getFitnessScore();
        ROS_INFO("loop closure ndt result :");
        ROS_INFO("hasConverged : %d", ndt.hasConverged());
        ROS_INFO("iteration : %d", ndt.getFinalNumIteration());
        ROS_INFO("fitness_score : %f", fitness_score);
        ROS_INFO("trans_probability :%f", ndt.getTransformationProbability());

        if (fitness_score > 0.8)
        {
            ROS_INFO("loop closure bad ndt match !!!");
            return;
        }

        trans = ndt.getFinalTransformation();
        Eigen::Matrix3d R;
        R << trans(0, 0), trans(0, 1), trans(0, 2), trans(1, 0), trans(1, 1), trans(1, 2), trans(2, 0), trans(2, 1), trans(2, 2);
        Eigen::Quaterniond q_lc = Eigen::Quaterniond(R);
        Eigen::Vector3d t_lc = Eigen::Vector3d(trans(0, 3), trans(1, 3), trans(2, 3));

        float noise_score = 0.5;
        gtsam::Vector v(6);
        v << noise_score, noise_score, noise_score, noise_score, noise_score, noise_score;
        constraint_noise_ = noiseModel::Diagonal::Variances(v);
        robust_noise_model_ = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(v));

        Eigen::Quaterniond q_to = keyposes_q_[ls_frame_id];
        Eigen::Vector3d t_to = keyposes_t_[ls_frame_id];

        Eigen::Quaterniond q_from = q_lc * keyposes_q_[current_frame_id];
        Eigen::Vector3d t_from = q_lc * keyposes_t_[current_frame_id] + t_lc;

        gtsam::Pose3 pose_from = Pose3(Rot3(q_from), Point3(t_from));
        gtsam::Pose3 pose_to = Pose3(Rot3(q_to), Point3(t_to));

        gtsam_graph_.add(BetweenFactor<Pose3>(current_frame_id, ls_frame_id, pose_from.between(pose_to), robust_noise_model_));

        isam_->update(gtsam_graph_);
        isam_->update();
        isam_->update();
        isam_->update();

        isam_current_estimate_ = isam_->calculateEstimate();
        gtsam_graph_.resize(0);
        initial_estimate_.clear();

        //update keyframes
        int num_poses = isam_current_estimate_.size();
        for (int i = 0; i < num_poses; i++)
        {
            double x = isam_current_estimate_.at<Pose3>(i).translation().x();
            double y = isam_current_estimate_.at<Pose3>(i).translation().y();
            double z = isam_current_estimate_.at<Pose3>(i).translation().z();

            keyposes_q_[i] = isam_current_estimate_.at<Pose3>(i).rotation().matrix();
            keyposes_t_[i](0) = x;
            keyposes_t_[i](1) = y;
            keyposes_t_[i](2) = z;

            pkeyposes_3d_->points[i].x = x;
            pkeyposes_3d_->points[i].y = y;
            pkeyposes_3d_->points[i].z = z;

            if (i < num_poses - 1)
            {
                path_.poses[i].pose.orientation.x = keyposes_q_[i].x();
                path_.poses[i].pose.orientation.y = keyposes_q_[i].y();
                path_.poses[i].pose.orientation.z = keyposes_q_[i].z();
                path_.poses[i].pose.orientation.w = keyposes_q_[i].w();
                path_.poses[i].pose.position.x = x;
                path_.poses[i].pose.position.y = y;
                path_.poses[i].pose.position.z = z;
            }
        }

        current_pose_.x = isam_current_estimate_.at<Pose3>(num_poses - 1).translation().x();
        current_pose_.y = isam_current_estimate_.at<Pose3>(num_poses - 1).translation().y();
        current_pose_.z = isam_current_estimate_.at<Pose3>(num_poses - 1).translation().z();
        odo_q_ = keyposes_q_[current_frame_id];
        odo_t_ = keyposes_t_[current_frame_id];

        ROS_INFO("target : %.3lf, %.3lf, %.3lf", t_from(0), t_from(1), t_from(2));
        ROS_INFO("res    : %.3lf, %.3lf, %.3lf", current_pose_.x, current_pose_.y, current_pose_.z);

        // match map
        pcloud_recent_keyframes_.clear();
        for (int i = 0; i < recent_keyframes_num_; i++)
        {
            int id = num_poses - 1 - i;
            if (id < 0)
                break;

            pcloud_recent_keyframes_.push_front(TransformPointCloud(keyposes_q_[id], keyposes_t_[id], pcloud_keyframes_[id]));
        }
        lateset_frame_id_ = pkeyposes_3d_->points.size() - 1;

        ptargetcloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        for (int i = 0; i < pcloud_recent_keyframes_.size(); i++)
            *ptargetcloud_ += *pcloud_recent_keyframes_[i];

        downsize_filter_25_.setInputCloud(ptargetcloud_);
        downsize_filter_25_.filter(*ptargetcloud_ds_);
        anh_ndt_.setInputTarget(ptargetcloud_ds_);
    }

    void ExtractSurroundingKeyFrames()
    {
        bool update_targetcloud_flag = false;

        if (pkeyposes_3d_->points.empty())
            return;

        if (pcloud_recent_keyframes_.size() < recent_keyframes_num_)
        {
            if (lateset_frame_id_ != pkeyposes_3d_->points.size() - 1)
            {
                lateset_frame_id_ = pkeyposes_3d_->points.size() - 1;
                pcloud_recent_keyframes_.push_back(TransformPointCloud(keyposes_q_[lateset_frame_id_], keyposes_t_[lateset_frame_id_], pcloud_keyframes_[lateset_frame_id_]));
                update_targetcloud_flag = true;
            }
        }
        else
        {
            if (lateset_frame_id_ != pkeyposes_3d_->points.size() - 1)
            {
                pcloud_recent_keyframes_.pop_front();
                lateset_frame_id_ = pkeyposes_3d_->points.size() - 1;
                pcloud_recent_keyframes_.push_back(TransformPointCloud(keyposes_q_[lateset_frame_id_], keyposes_t_[lateset_frame_id_], pcloud_keyframes_[lateset_frame_id_]));
                update_targetcloud_flag = true;
            }
        }

        if (!update_targetcloud_flag)
            return;

        ptargetcloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        for (int i = 0; i < pcloud_recent_keyframes_.size(); i++)
            *ptargetcloud_ += *pcloud_recent_keyframes_[i];

        downsize_filter_25_.setInputCloud(ptargetcloud_);
        downsize_filter_25_.filter(*ptargetcloud_ds_);
        anh_ndt_.setInputTarget(ptargetcloud_ds_);
    }

    void Visulization()
    {
        if (!keyframe_flag_)
            return;

        geometry_msgs::PoseStamped odom;
        odom.pose.orientation.x = odo_q_.x();
        odom.pose.orientation.y = odo_q_.y();
        odom.pose.orientation.z = odo_q_.z();
        odom.pose.orientation.w = odo_q_.w();
        odom.pose.position.x = odo_t_(0);
        odom.pose.position.y = odo_t_(1);
        odom.pose.position.z = odo_t_(2);
        odom.header.frame_id = "/map";
        odom.header.stamp = odom.header.stamp.fromSec(odo_time_);
        pub_odom_.publish(odom);

        path_.poses.push_back(odom);
        path_.header.frame_id = "/map";
        path_.header.stamp = odom.header.stamp;
        pub_path_.publish(path_);

        sensor_msgs::PointCloud2 targetcloud2;
        pcl::toROSMsg(*ptargetcloud_ds_, targetcloud2);
        targetcloud2.header.stamp = odom.header.stamp;
        targetcloud2.header.frame_id = "/map";
        pub_targetcloud_.publish(targetcloud2);
    }

    void MapFromBagFile()
    {
        rosbag::Bag bag;
        bag.open(bag_path_, rosbag::bagmode::Read);

        std::vector<std::string> topics;
        topics.push_back(std::string("/rslidar_points"));
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        foreach (rosbag::MessageInstance const m, view)
        {
            std::string topic = m.getTopic();

            sensor_msgs::PointCloud2::ConstPtr msg = m.instantiate<sensor_msgs::PointCloud2>();
            if (msg != NULL)
            {
                odo_time_ = msg->header.stamp.toSec();
                pcl::fromROSMsg(*msg, *pcloud_raw_);

                PointCloudPreprocess(pcloud_raw_, pcloud_);
                OdoEstimate();
                SaveKeyFrame();
                DetectLoopClosure();
                ExtractSurroundingKeyFrames();
                Visulization();

                keyframe_flag_ = false;
                poses_q_.push_back(odo_q_);
                poses_t_.push_back(odo_t_);
                poses_time_.push_back(odo_time_);
                ROS_INFO("time %.3lf : %.2lf, %.2lf, %.2lf", odo_time_, odo_t_(0), odo_t_(1), odo_t_(2));

                if(!ros::ok())
                    break;
            }
        }
        bag.close();
        ROS_INFO("mapping finished !");
    }

    void SaveMap()
    {
        ROS_INFO("wait a second ...");
        ROS_INFO("final map visulization ...");

        pcl::PointCloud<pcl::PointXYZI>::Ptr pmap(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr pmap_ds(new pcl::PointCloud<pcl::PointXYZI>());

        for(int i = 0; i < pkeyposes_3d_->points.size(); i++)
            *pmap += *TransformPointCloud(keyposes_q_[i], keyposes_t_[i], pcloud_keyframes_[i]);

        downsize_filter_25_.setInputCloud(pmap);
        downsize_filter_25_.filter(*pmap_ds);

        pmap_ds->height = 1;
        pmap_ds->width = pmap_ds->points.size();

        sensor_msgs::PointCloud2 map2;
        pcl::toROSMsg(*pmap_ds, map2);
        map2.header.stamp = ros::Time().now();
        map2.header.frame_id = "/map";
        pub_pointcloud_.publish(map2);

        if(!save_map_)
            return;
        
        ROS_INFO("saving map ... \n");
        string pcd_path = map_path_ + "map.pcd";
        pcl::io::savePCDFileASCII(pcd_path, *pmap_ds);
        printf("save map pcd at %s \n", pcd_path.c_str());
    }

    void SaveTumFormat()
    {   
        FILE *fp = NULL;
        char end1 = 0x0d; // "/n"
        char end2 = 0x0a;

        // lidar odometry
        string lidar_tum_file = map_path_ + "lidar.txt";
        fp = fopen(lidar_tum_file.c_str(), "w+");

        if (fp == NULL)
        {
            printf("fail to open file %s ! \n", lidar_tum_file.c_str());
            exit(1);
        }
        else
            printf("TUM : write lidar data to %s \n", lidar_tum_file.c_str());

        for (int i = 0; i < poses_q_.size(); i++)
        {
            Eigen::Quaterniond q = poses_q_[i];
            Eigen::Vector3d t = poses_t_[i];
            double time = poses_time_[i];
            fprintf(fp, "%.3lf %.3lf %.3lf %.3lf %.5lf %.5lf %.5lf %.5lf%c",
                    time, t(0), t(1), t(2),
                    q.x(), q.y(), q.z(), q.w(), end2);
        }
        fclose(fp);
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_pointcloud_;
    ros::Publisher pub_odom_;
    ros::Publisher pub_path_;
    ros::Publisher pub_targetcloud_;

    nav_msgs::Path path_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_raw_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ptargetcloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ptargetcloud_ds_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr psourcecloud_;

    //params
    string bag_path_;
    string map_path_;
    double dist_;
    double map_dist_;
    bool save_map_;
    bool keyframe_flag_;

    //pose
    Eigen::Quaterniond odo_q_;
    Eigen::Vector3d odo_t_;
    Eigen::Quaterniond odo_q_last_;
    Eigen::Vector3d odo_t_last_;
    double odo_time_;

    pcl::PointXYZI current_pose_;

    //keyframes
    pcl::PointCloud<pcl::PointXYZI>::Ptr pkeyposes_3d_;
    vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> keyposes_q_;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> keyposes_t_;
    vector<double> keyposes_time_;

    vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> poses_q_;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> poses_t_;
    vector<double> poses_time_;

    vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcloud_keyframes_;

    //kdtree
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_surrounding_keyposes_;
    double surrounding_keyposes_search_radius_;

    //surrounding pointcloud
    deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> pcloud_recent_keyframes_;
    int recent_keyframes_num_;
    int lateset_frame_id_;

    //VoxelGrid
    pcl::VoxelGrid<pcl::PointXYZI> downsize_filter_25_;
    pcl::VoxelGrid<pcl::PointXYZI> downsize_filter_50_;

    //ndt
    cpu::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> anh_ndt_;

    //gtsam
    NonlinearFactorGraph gtsam_graph_;
    Values initial_estimate_;
    Values optimized_estimate_;
    ISAM2 *isam_;
    Values isam_current_estimate_;

    noiseModel::Diagonal::shared_ptr prior_noise_;
    noiseModel::Diagonal::shared_ptr odometry_noise_;
    noiseModel::Diagonal::shared_ptr constraint_noise_;
    noiseModel::Base::shared_ptr robust_noise_model_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapping");
    ros::NodeHandle nh;

    mapping map(nh);
    map.MapFromBagFile();
    map.SaveMap();
    map.SaveTumFormat();
    return 0;
}