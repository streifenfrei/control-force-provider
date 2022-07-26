#pragma once

#include <Python.h>

#include "utils.h"

// https://www.codeproject.com/Articles/820116/Embedding-Python-program-in-a-C-Cplusplus-code
using namespace control_force_provider::exceptions;
namespace control_force_provider::backend {
class PythonObject {
 private:
  PyObject* p;

 public:
  PythonObject() : p(nullptr) {}

  PythonObject(PyObject* _p) : p(_p) {}

  ~PythonObject() {
    if (p) Py_DECREF(p);
  }

  PyObject* getObject() { return p; }

  PyObject* addRef() {
    if (p) {
      Py_INCREF(p);
    }
    return p;
  }

  PyObject* operator->() { return p; }

  operator PyObject*() { return p; }

  PyObject* operator=(PyObject* pp) {
    p = pp;
    return p;
  }

  explicit operator bool() { return p != nullptr; }

  PyObject* callFunction(const std::string& function_name, PyObject* args = nullptr, PyObject* kwargs = nullptr) {
    if (!p) throw PythonError("Failed to call python function '" + function_name + "': Parent object is NULL.");
    PyObject* function = PyObject_GetAttrString(p, function_name.c_str());
    if (function) {
      if (!args) args = PyTuple_New(0);
      if (PyMethod_Check(function)) {
        function = PyMethod_Function(function);
        unsigned int num_args = PyTuple_Size(args) + 1;
        PyObject* method_args = PyTuple_New(num_args);
        addRef();
        PyTuple_SetItem(method_args, 0, p);
        for (size_t i = 1; i < num_args; i++) {
          PyTuple_SetItem(method_args, i, PyTuple_GetItem(args, i - 1));
        }
        args = method_args;
      }
      if (PyCallable_Check(function)) {
        PyObject* result = PyObject_Call(function, args, kwargs);
        if (PyErr_Occurred()) {
          PyErr_Print();
          throw PythonError("Failed to call python function '" + function_name + "': Python exception occurred (see message above).");
        }
        if (!result) throw PythonError("Failed to call python function '" + function_name + "': PyObject_Call returned NULL without exception.");
        return result;
      }
    }
    throw PythonError("Failed to call python function '" + function_name + "': Could not find such an attribute.");
  }
};

class PythonEnvironment {
 private:
  inline static unsigned int use_counter = 0;

 public:
  PythonEnvironment() {
    if (use_counter == 0) Py_Initialize();
    use_counter++;
  }
  ~PythonEnvironment() {
    use_counter--;
    if (use_counter == 0) Py_Finalize();
  }
  static PythonObject loadPythonModule(const std::string& module_name) {
    PythonObject py_module = PyImport_ImportModule(module_name.c_str());
    if (PyErr_Occurred()) PyErr_Print();
    if (!py_module) throw PythonError("Failed to import the python module '" + module_name + "'");
    return py_module;
  }
};
}  // namespace control_force_provider::backend