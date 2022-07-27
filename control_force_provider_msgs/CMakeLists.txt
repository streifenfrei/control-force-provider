cmake_minimum_required(VERSION 3.17)
project(control_force_provider_msgs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(
        catkin REQUIRED COMPONENTS
        std_msgs
        message_generation)

add_service_files(
        FILES
        UpdateNetwork.srv
)

generate_messages(
        DEPENDENCIES
        std_msgs
)

catkin_package(
        CATKIN_DEPENDS
        std_msgs
        message_runtime
)

