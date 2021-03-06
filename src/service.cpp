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
void* data;

void cleanup(int signum) {
  free(data);
  remove(socket_file);
}

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

int main(int argc, char* argv[]) {
  control_force_provider::ControlForceProvider cfp;
  asio::io_context io_context_{};
  remove(socket_file);
  asio::local::stream_protocol::endpoint uds_ep_{socket_file};
  asio::local::stream_protocol::acceptor acceptor(io_context_, uds_ep_);
  asio::local::stream_protocol::socket uds_socket_{io_context_};
  acceptor.accept(uds_socket_);
  data = malloc(vector_size);
  signal(SIGTERM, cleanup);
  asio::mutable_buffer input_buffer(data, vector_size);
  if (setRealtimePriority()) ROS_INFO_STREAM_NAMED("control_force_provider/service", "Successfully set realtime priority for current thread.");
  while (true) {  // TODO make loop interruptable. Does not work currently as read() blocks and async_read() would spawn a non RT thread;
    read(uds_socket_, input_buffer);
    Eigen::Vector4d ee_position((double*)input_buffer.data());
    Eigen::Vector4d force;
    cfp.getForce(force, ee_position);
    write(uds_socket_, asio::const_buffer((const void*)force.data(), vector_size));
  }
}
