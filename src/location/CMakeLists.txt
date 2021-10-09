cmake_minimum_required(VERSION 3.0.2)
project(location)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  sensor_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  ndt_cpu
  ndt_omp
  location_msgs
  tf
)

find_package(PCL 1.8 REQUIRED)
find_package(Eigen3)

catkin_package(
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)


add_executable(location src/location.cpp)
target_link_libraries(location ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(location ${catkin_EXPORTED_TARGETS})

add_executable(score_test src/score_test.cpp)
target_link_libraries(score_test ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(score_test ${catkin_EXPORTED_TARGETS})