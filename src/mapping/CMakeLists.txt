cmake_minimum_required(VERSION 3.0.2)
project(mapping)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  sensor_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  ndt_cpu
  location_msgs
)

find_package(PCL 1.8 REQUIRED)
find_package(Eigen3)
find_package(GTSAM REQUIRED QUIET)


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
  ${GTSAM_LIBRARY_DIRS}
)

set(LIB_TBB /usr/lib/x86_64-linux-gnu)
add_library(libtbb SHARED IMPORTED)
set_target_properties(libtbb PROPERTIES IMPORTED_LOCATION ${LIB_TBB}/libtbb.so.2)

add_executable(map src/map.cpp)
target_link_libraries(map ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam libtbb)
add_dependencies(map ${catkin_EXPORTED_TARGETS})

add_executable(odorecord src/odorecord.cpp)
target_link_libraries(odorecord ${catkin_LIBRARIES})
add_dependencies(odorecord ${catkin_EXPORTED_TARGETS})