#include "control_force_provider/utils.h"

#include <ros/ros.h>

#include <boost/regex.hpp>
#include <iostream>

namespace control_force_provider {
namespace exceptions {
Error::Error(const std::string &message, const std::string &error_type) : runtime_error(message) {
  ROS_ERROR_STREAM_NAMED("control_force_provider", error_type << ": " << message);
}
ConfigError::ConfigError(const std::string &message) : Error(message, "ConfigError") {}
PythonError::PythonError(const std::string &message) : Error(message, "PythonError") {}
}  // namespace exceptions

namespace utils {
using namespace exceptions;
using namespace Eigen;

std::vector<std::string> regexFindAll(const std::string &regex, const std::string &str) {
  boost::sregex_token_iterator iter(str.begin(), str.end(), boost::regex(regex), 0);
  boost::sregex_token_iterator end;
  std::vector<std::string> result;
  for (; iter != end; iter++) result.emplace_back(*iter);
  return result;
}

Vector3d vectorFromList(const std::vector<double> &list, unsigned int start_index) {
  return Vector3d(list[start_index], list[start_index + 1], list[start_index + 2]);
}
}  // namespace utils
}  // namespace control_force_provider