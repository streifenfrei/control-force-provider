#include <pthread.h>
#include <ros/ros.h>

#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <csignal>

#include "control_force_provider/control_force_provider.h"

using namespace std::string_literals;
using namespace boost;

const char* socket_file = "/tmp/cfp_uds";
const int vector_size = 4 * sizeof(double);
bool stop = false;

void sigint_handler(int signum) { stop = true; }

#ifdef REALTIME
bool setRealtimePriority() {
  const int thread_priority = sched_get_priority_max(SCHED_FIFO);
  if (thread_priority == -1) {
    ROS_WARN_STREAM_NAMED("control_force_provider/service", "Unable to get maximum possible thread priority: " << std::strerror(errno));
    return false;
  }
  sched_param thread_param{};
  thread_param.sched_priority = thread_priority;
  if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &thread_param) != 0) {
    ROS_WARN_STREAM_NAMED("control_force_provider/service", "Unable to set realtime scheduling: " << std::strerror(errno));
    return false;
  }
  return true;
}
#endif

int main(int argc, char* argv[]) {
  bool detached = argc >= 2 && std::string(argv[1]) == "-d";
  control_force_provider::ControlForceProvider cfp;
  asio::io_context io_context_{};
  remove(socket_file);
  asio::local::stream_protocol::endpoint uds_ep_{socket_file};
  asio::local::stream_protocol::acceptor acceptor(io_context_, uds_ep_);
  asio::local::stream_protocol::socket uds_socket_{io_context_};
  acceptor.accept(uds_socket_);
  void* data = malloc(vector_size);
  asio::mutable_buffer input_buffer(data, vector_size);
#ifdef REALTIME
  if (setRealtimePriority()) ROS_INFO_STREAM_NAMED("control_force_provider/service", "Successfully set realtime priority for current thread.");
#endif
  signal(SIGINT, sigint_handler);
  boost::shared_ptr<control_force_provider::SimulatedRobot> robot;
  if (detached) robot = boost::make_shared<control_force_provider::SimulatedRobot>(cfp.getRCM(), Eigen::Vector4d(0.3, 0.0, 0.3, 0.0), cfp);
  while (!stop) {
    if (uds_socket_.available() >= vector_size) {
      Eigen::Vector4d force = Eigen::Vector4d::Zero();
      read(uds_socket_, input_buffer);
      if (detached) {
        robot->update();
      } else {
        Eigen::Vector4d ee_position((double*)input_buffer.data());
        cfp.getForce(force, ee_position);
      }
      write(uds_socket_, asio::const_buffer((const void*)force.data(), vector_size));
    }
  }
  remove(socket_file);
  free(data);
}
