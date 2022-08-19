#pragma once

#include <ryml.hpp>

namespace control_force_provider {
namespace exceptions {
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string &message, const std::string &error_type);
};
class ConfigError : Error {
 public:
  explicit ConfigError(const std::string &message);
};
class PythonError : Error {
 public:
  explicit PythonError(const std::string &message);
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
std::vector<ryml::NodeRef> getConfigValue(const ryml::NodeRef &config, const std::string &key);
template <>
std::vector<std::string> getConfigValue(const ryml::NodeRef &config, const std::string &key);
template <>
std::vector<int> getConfigValue(const ryml::NodeRef &config, const std::string &key);
template <>
std::vector<double> getConfigValue(const ryml::NodeRef &config, const std::string &key);
template <>
std::vector<bool> getConfigValue(const ryml::NodeRef &config, const std::string &key);

std::vector<std::string> regex_findall(std::string regex, std::string str);

}  // namespace utils
}  // namespace control_force_provider