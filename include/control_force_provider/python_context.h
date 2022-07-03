#pragma once

#include <Python.h>

#include "utils.h"

// https://www.codeproject.com/Articles/820116/Embedding-Python-program-in-a-C-Cplusplus-code
using namespace control_force_provider::exceptions;
namespace control_force_provider::backend {
class PythonEnvironment {
 protected:
  class PythonObject {
   private:
    PyObject* p;

   public:
    PythonObject() : p(nullptr) {}

    PythonObject(PyObject* _p) : p(_p) {}

    ~PythonObject() { Release(); }

    PyObject* getObject() { return p; }

    PyObject* setObject(PyObject* _p) { return (p = _p); }

    PyObject* AddRef() {
      if (p) {
        Py_INCREF(p);
      }
      return p;
    }

    void Release() {
      if (p) {
        Py_DECREF(p);
      }

      p = nullptr;
    }

    PyObject* operator->() { return p; }

    bool is() { return p != nullptr; }

    operator PyObject*() { return p; }

    PyObject* operator=(PyObject* pp) {
      p = pp;
      return p;
    }

    explicit operator bool() { return p != nullptr; }

    PythonObject callFunction(const std::string& function_name, PythonObject args = nullptr) {
      PythonObject function = PyObject_GetAttrString(p, function_name.c_str());
      if (function) {
        if (PyMethod_Check(function)) function = PyMethod_Function(function);
        if (PyCallable_Check(function)) return PyObject_CallObject(function, args);
      }
      throw PythonError("Failed to call python function '" + function_name + "'");
    }
  };
  PythonEnvironment() { Py_Initialize(); }
  ~PythonEnvironment() { Py_Finalize(); }
  static PythonObject loadPythonModule(const std::string& module_name) {
    PythonObject py_module = PyImport_ImportModule(module_name.c_str());
    if (!py_module) throw PythonError("Failed to import the python module '" + module_name + "'");
    return py_module;
  }
};

}  // namespace control_force_provider::backend