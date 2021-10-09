cmake_minimum_required(VERSION 2.8.3)
project(can_decode)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g ")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  location_msgs
  message_generation
)

find_package(Eigen3 REQUIRED)

add_message_files(
  FILES
  rtk.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
  message_runtime 
  message_generation
  geometry_msgs 
  sensor_msgs 
  std_msgs
  nav_msgs
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

link_directories(${catkin_LIB_DIRS} lib)

add_executable(can_decode
    src/can_decode.cpp)

target_link_libraries(can_decode
  ${catkin_LIBRARIES}
  libCanCmd.so
)

add_dependencies(can_decode ${catkin_EXPORTED_TARGETS})

# add_executable(path_visu
#     src/path_visu.cpp)

# target_link_libraries(path_visu
#   ${catkin_LIBRARIES}
# )

