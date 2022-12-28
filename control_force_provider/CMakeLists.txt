cmake_minimum_required(VERSION 3.18)
project(control_force_provider)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(RT "use realtime threads" OFF)

# SETUP
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
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib)
find_package(pybind11 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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

# MAIN LIBRARY
add_library(
        ${PROJECT_NAME}
        src/control_force_provider.cpp
        src/control_force_calculator.cpp
        src/obstacle.cpp
        src/time.cpp
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

# PYTHON CORE BINDING
pybind11_add_module(native src/rl_environment.cpp)
target_link_libraries(native PUBLIC ${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
set_target_properties(native PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION})

# EXECUTABLES
add_executable(uds_service src/uds_service.cpp)
if (RT)
    target_compile_definitions(uds_service PUBLIC REALTIME)
endif ()
target_link_libraries(uds_service ${PROJECT_NAME})
catkin_install_python(PROGRAMS scripts/rl_training DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
