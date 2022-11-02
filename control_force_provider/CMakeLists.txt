cmake_minimum_required(VERSION 3.16)
project(control_force_provider)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(RT "use realtime threads" OFF)

find_package(
        catkin REQUIRED COMPONENTS
        geometry_msgs
        eigen_conversions
        rviz_visual_tools
        control_force_provider_msgs
        roscpp)

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})

catkin_python_setup()

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES ${PROJECT_NAME}
        CATKIN_DEPENDS
        message_runtime
        geometry_msgs
        eigen_conversions
        rviz_visual_tools
        control_force_provider_msgs
        roscpp)

add_library(
        ${PROJECT_NAME}
        src/control_force_provider.cpp
        src/control_force_calculator.cpp
        src/obstacle.cpp
        src/visualizer.cpp
        src/utils.cpp)

target_link_libraries(
        ${PROJECT_NAME} PUBLIC
        Eigen3::Eigen
        yaml-cpp
        ${TORCH_LIBRARIES}
        ${catkin_LIBRARIES})

target_include_directories(
        ${PROJECT_NAME} PUBLIC
        include
        ${catkin_INCLUDE_DIRS})
if (RT)
    target_compile_definitions(${PROJECT_NAME} PUBLIC REALTIME)
endif()

add_executable(service src/service.cpp)
if (RT)
    target_compile_definitions(${PROJECT_NAME} PUBLIC REALTIME)
endif()

target_link_libraries(service ${PROJECT_NAME})

catkin_install_python(PROGRAMS scripts/rl_training
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
