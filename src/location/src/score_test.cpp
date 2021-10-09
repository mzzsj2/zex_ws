#include <unistd.h>

#include <ros/ros.h>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/impl/io.hpp>

#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <mutex>

#include <ndt_cpu/NormalDistributionsTransform.h>
#include <ndt_cpu/VoxelGrid.h>

#include <eigen3/Eigen/Dense>

using namespace std::chrono_literals;
using std::string;
class score_test
{
public:
    score_test(string target_path, string source_path) : target_path_(target_path), source_path_(source_path)
    {
        pSourceCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pTargetCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());
        pScoreCloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

        if (pcl::io::loadPCDFile<pcl::PointXYZI>(source_path_, *pSourceCloud_) == -1)
        {
            printf("can't read open pcd \n");
            exit(-1);
        }
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(target_path_, *pTargetCloud_) == -1)
        {
            printf("can't read open pcd \n");
            exit(-1);
        }

        anh_ndt_.setResolution(1.0);
        anh_ndt_.setMaximumIterations(500);
        anh_ndt_.setStepSize(0.1);
        anh_ndt_.setTransformationEpsilon(0.01);

        anh_ndt_.setInputTarget(pTargetCloud_);
        anh_ndt_.setInputSource(pSourceCloud_);

        score_flag_ = false;
        q_score_ = Eigen::Quaterniond(1, 0, 0, 0);
        t_score_ = Eigen::Vector3d(0, 0, 0);
    }

    void main_loop()
    {
        ros::Rate rate(10);

        int count = 0;

        while (ros::ok())
        {
            ROS_INFO("start ndt");
            Eigen::Matrix4f trans(Eigen::Matrix4f::Identity());

            Eigen::Translation3d init_translation(-2, -2, -2);
            Eigen::AngleAxisd init_rotation(Eigen::Quaterniond(1, 0, 0, 0));
            Eigen::Matrix4d init_guess = (init_translation * init_rotation) * Eigen::Matrix4d::Identity();
            anh_ndt_.align(init_guess.cast<float>());

            trans = anh_ndt_.getFinalTransformation();
            int iteration = anh_ndt_.getFinalNumIteration();
            double trans_probability = anh_ndt_.getTransformationProbability();

            ROS_INFO("hasConverged : %d", anh_ndt_.hasConverged());
            ROS_INFO("score : %f", anh_ndt_.getFitnessScore());
            count++;

            if (count == 10)
            {
                mtx_.lock();
                ROS_INFO("start copy");
                pcl::copyPointCloud(*pSourceCloud_, *pScoreCloud_);
                map_for_score_ = anh_ndt_.getVoxelGrid();
                Eigen::Matrix3d dR;
                dR << trans(0, 0), trans(0, 1), trans(0, 2), trans(1, 0), trans(1, 1), trans(1, 2), trans(2, 0), trans(2, 1), trans(2, 2);
                q_score_ = Eigen::Quaterniond(dR);
                t_score_ = Eigen::Vector3d(trans(0, 3), trans(1, 3), trans(2, 3));
                score_flag_ = true;
                count = 0;
                ROS_INFO("finish copy");
                mtx_.unlock();
            }
            rate.sleep();
        }
    }

    void score_loop()
    {
        ros::Rate rate(1);

        while (ros::ok())
        {
            if (score_flag_)
            {
                ROS_INFO("start score");
                mtx_.lock();
                double fitness_score = cal_score();
                ROS_INFO("my fitness_score : %f", fitness_score);
                score_flag_ = false;
                mtx_.unlock();
            }

            rate.sleep();
        }
    }

    double cal_score()
    {
        double fitness_score = 0;
        double distance = 0;
        int nr = 0;

        Eigen::Vector3d p_xyz(0, 0, 0);
        pcl::PointXYZI p_xyzi;

        for (int i = 0; i < pScoreCloud_->points.size(); i++)
        {
            Eigen::Vector3d p(pScoreCloud_->points[i].x, pScoreCloud_->points[i].y, pScoreCloud_->points[i].z);
            p_xyz = q_score_ * p + t_score_;
            p_xyzi.x = p_xyz(0);
            p_xyzi.y = p_xyz(1);
            p_xyzi.z = p_xyz(2);

            distance = map_for_score_.nearestNeighborDistance(p_xyzi, 100000);

            if (distance < 100000)
            {
                fitness_score += distance;
                nr++;
            }
        }

        if (nr > 0)
            return (fitness_score / nr);

        return 100000;
    }

private:
    string target_path_;
    string source_path_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr pSourceCloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pTargetCloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pScoreCloud_;

    cpu::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> anh_ndt_;

    cpu::VoxelGrid<pcl::PointXYZI> map_for_score_;

    bool score_flag_;

    Eigen::Quaterniond q_score_;
    Eigen::Vector3d t_score_;

    std::mutex mtx_;
};

int main(int argc, char **argv)
{

    ros::init(argc, argv, "test");
    ros::NodeHandle nh;

    score_test t(argv[1], argv[2]);

    std::thread score_thread(&score_test::score_loop, &t);

    t.main_loop();

    score_thread.join();
    return 0;
}