#pragma once

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <type_traits>

namespace control_force_provider {

class Time {
 private:
  static boost::shared_ptr<Time> instance;
  virtual double now_() = 0;

 protected:
  Time() = default;

 public:
  static boost::shared_ptr<Time> getInstance();
  static double now();
  template <typename T>
  static void setType() {
    static_assert(std::is_base_of<Time, T>());
    instance = boost::make_shared<T>();
  }
};

class ROSTime : public Time {
 public:
  ROSTime() = default;
  double now_() override;
};

class ManualTime : public Time {
 private:
  double now = 0;

 public:
  ManualTime() = default;
  double now_() override;
  void operator+=(double t);
};

}  // namespace control_force_provider
