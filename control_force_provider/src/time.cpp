#include "control_force_provider/time.h"

#include <ros/time.h>

namespace control_force_provider {

boost::shared_ptr<Time> Time::instance = nullptr;

boost::shared_ptr<Time> Time::getInstance() {
  if (!instance) setType<ROSTime>();
  return instance;
}

double Time::now() { return getInstance()->now_(); }

double ROSTime::now_() { return ros::Time::now().toSec(); }

double ManualTime::now_() { return now; }

void ManualTime::operator+=(double t) { this->now += t; }
}  // namespace control_force_provider