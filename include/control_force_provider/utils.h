#pragma once

#include <boost/algorithm/string.hpp>
#include <iostream>
#include <ryml.hpp>

namespace control_force_provider {
namespace exceptions {
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string &message, const std::string &error_type) : runtime_error(message) {
    ROS_ERROR_STREAM_NAMED("control_force_provider", error_type << ": " << message);
  }
};
class ConfigError : Error {
 public:
  explicit ConfigError(const std::string &message) : Error(message, "ConfigError") {}
};
class PythonError : Error {
 public:
  explicit PythonError(const std::string &message) : Error(message, "PythonError") {}
};
}  // namespace exceptions

namespace utils {
using namespace exceptions;

// == ryml::to_csubstr but static (hacky workaround for gzserver error)
static ryml::csubstr to_csubstr(std::string const &s) {
  const char *data = !s.empty() ? &s[0] : nullptr;
  return ryml::csubstr(data, s.size());
}

static std::string until_newline(std::string const &s) {
  std::string::size_type pos = s.find('\n');
  if (pos != std::string::npos)
    return s.substr(0, pos);
  else
    return s;
}

template <typename T>
std::vector<T> getConfigValue(const ryml::NodeRef &config, const std::string &key);
template <>
inline std::vector<ryml::NodeRef> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
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
inline std::vector<std::string> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<ryml::NodeRef> nodes = getConfigValue<ryml::NodeRef>(config, key);
  std::vector<std::string> strings;
  for (auto &node : nodes) {
    strings.emplace_back(until_newline(node.val().data()));
  }
  return strings;
}
template <>
inline std::vector<int> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<std::string> strings = getConfigValue<std::string>(config, key);
  std::vector<int> values;
  for (auto &string : strings) {
    values.push_back(std::stoi(string));
  }
  return values;
}
template <>
inline std::vector<double> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<std::string> strings = getConfigValue<std::string>(config, key);
  std::vector<double> values;
  for (auto &string : strings) {
    values.push_back(std::stod(string));
  }
  return values;
}
template <>
inline std::vector<bool> getConfigValue(const ryml::NodeRef &config, const std::string &key) {
  std::vector<std::string> strings = getConfigValue<std::string>(config, key);
  std::vector<bool> values;
  for (auto &string : strings) {
    values.push_back(boost::algorithm::to_lower_copy(string) == "true");
  }
  return values;
}
}  // namespace utils
}  // namespace control_force_provider