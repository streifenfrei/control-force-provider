#include "control_force_provider/utils.h"

#include <ros/ros.h>

#include <boost/algorithm/string.hpp>
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
template <typename T>
std::vector<T> getConfigValue(const ryml::NodeRef &config, const std::string &key);
template <>
std::vector<ryml::NodeRef> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  ryml::csubstr key_r = to_csubstr(key);
  if (!config.has_child(key_r)) {
    throw ConfigError("Missing key '" + key + "'");
  }
  ryml::NodeRef node = config[key_r];
  std::vector<ryml::NodeRef> value;
  if (node.is_seq()) {
    for (ryml::NodeRef const &child : node.children()) {
      value.push_back(child);
    }
  } else {
    value.emplace_back(node);
  }
  return value;
}
template <>
std::vector<std::string> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<ryml::NodeRef> nodes = getConfigValue<ryml::NodeRef>(config, key);
  std::vector<std::string> strings;
  for (auto &node : nodes) {
    strings.emplace_back(until_newline(node.val().data()));
  }
  return strings;
}
template <>
std::vector<int> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<std::string> strings = getConfigValue<std::string>(config, key);
  std::vector<int> values;
  for (auto &string : strings) {
    values.push_back(std::stoi(string));
  }
  return values;
}
template <>
std::vector<double> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<std::string> strings = getConfigValue<std::string>(config, key);
  std::vector<double> values;
  for (auto &string : strings) {
    values.push_back(std::stod(string));
  }
  return values;
}
template <>
std::vector<bool> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<std::string> strings = getConfigValue<std::string>(config, key);
  std::vector<bool> values;
  for (auto &string : strings) {
    values.push_back(boost::algorithm::to_lower_copy(string) == "true");
  }
  return values;
}

std::vector<std::string> regex_findall(std::string regex, std::string str) {
  boost::sregex_token_iterator iter(str.begin(), str.end(), boost::regex(regex), 0);
  boost::sregex_token_iterator end;
  std::vector<std::string> result;
  for (; iter != end; iter++) result.emplace_back(*iter);
  return result;
}

}  // namespace utils
}  // namespace control_force_provider