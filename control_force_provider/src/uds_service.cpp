#ifdef REALTIME
#include <pthread.h>
#endif

#include <boost/asio.hpp>
#include <boost/optional.hpp>
#include <csignal>

#include "control_force_provider/control_force_provider.h"

using namespace std::string_literals;
using namespace boost;
using namespace control_force_provider;

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
  ControlForceProvider cfp;
  signal(SIGINT, sigint_handler);
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
  while (!stop) {
    if (uds_socket_.available() >= vector_size) {
      read(uds_socket_, input_buffer);
      torch::Tensor ee_position = torch::from_blob(input_buffer.data(), {1, 3}, utils::getTensorOptions());
      torch::Tensor force = torch::empty({1, 3}, utils::getTensorOptions());
      cfp.getForce(force, ee_position);
      // TODO: make this nicer
      Eigen::Vector3d force_eigen = utils::tensorToVector(force);
      Eigen::Vector4d force_eigen2(force_eigen[0], force_eigen[1], force_eigen[2], 0);
      write(uds_socket_, asio::const_buffer((const void*)force_eigen2.data(), vector_size));
    }
  }
  remove(socket_file);
  free(data);
}
