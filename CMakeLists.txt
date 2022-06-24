cmake_minimum_required(VERSION 3.17)
project(control_force_provider)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
        catkin REQUIRED COMPONENTS
        rviz_visual_tools
        roscpp)

find_package(Eigen3 REQUIRED)
find_package(ryml REQUIRED)

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
        CATKIN_DEPENDS
        rviz_visual_tools
        roscpp
        INCLUDE_DIRS include)

add_library(
        ${PROJECT_NAME}
        src/control_force_provider.cpp
        src/control_force_calculator.cpp
        src/obstacle.cpp
        src/visualizer.cpp)

target_link_libraries(
        ${PROJECT_NAME} PUBLIC
        ${Franka_LIBRARIES}
        Eigen3::Eigen
        ryml::ryml
        ${catkin_LIBRARIES})

target_include_directories(
        ${PROJECT_NAME} PUBLIC
        include
        ${catkin_INCLUDE_DIRS})
