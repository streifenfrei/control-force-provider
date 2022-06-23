#pragma once

#include <yaml-cpp/yaml.h>

namespace control_force_provider::utils {
template <typename T>
std::vector<T> getConfigValue(const YAML::Node &config, const std::string &key) {
  YAML::Node node = config[key];
  if (!node.IsDefined()) {
    throw ConfigError("Missing key: " + key);
  }
  try {
    std::vector<T> value;
    if (node.IsSequence()) {
      for (auto &&i : node) {
        value.push_back(i.as<T>());
      }
    } else {
      value.push_back(node.as<T>());
    }
    return value;
  } catch (const YAML::BadConversion &exception) {
    throw ConfigError("Bad data type for key '" + key + "'. Expected: " + typeid(T).name());
  }
}
}  // namespace control_force_provider::utils