#pragma once

#include <Python.h>

// https://www.codeproject.com/Articles/820116/Embedding-Python-program-in-a-C-Cplusplus-code
namespace control_force_provider::backend {
class PythonEnvironment {
 public:
  PythonEnvironment() { Py_Initialize(); }
  ~PythonEnvironment() { Py_Finalize(); }
};

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

  explicit operator PyObject*() { return p; }

  PyObject* operator=(PyObject* pp) {
    p = pp;
    return p;
  }

  explicit operator bool() { return p != nullptr; }
};
}  // namespace control_force_provider::backend