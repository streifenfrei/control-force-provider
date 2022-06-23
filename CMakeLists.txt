cmake_minimum_required(VERSION 3.17)
project(control_force_provider)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
        catkin REQUIRED COMPONENTS
        tf
        tf_conversions
        message_generation
        roscpp)

find_package(Eigen3 REQUIRED)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
        CATKIN_DEPENDS
        eigen_conversions
        geometry_msgs
        tf
        tf_conversions
        message_runtime
        roscpp
        INCLUDE_DIRS include)

add_library(
        ${PROJECT_NAME}
        src/control_force_provider.cpp
)

target_link_libraries(
        ${PROJECT_NAME} PUBLIC
        ${Franka_LIBRARIES}
        ${catkin_LIBRARIES})

target_include_directories(
        ${PROJECT_NAME} PUBLIC
        include
        ${catkin_INCLUDE_DIRS})
