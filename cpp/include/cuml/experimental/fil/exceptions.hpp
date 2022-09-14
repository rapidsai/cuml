#pragma once
#include <exception>
#include <string>

namespace ML {
namespace experimental {
namespace fil {

struct unusable_model_exception : std::exception {
  unusable_model_exception () : msg_{"Model is not compatible with Herring"}
  {
  }
  unusable_model_exception (std::string msg) : msg_{msg}
  {
  }
  unusable_model_exception (char const* msg) : msg_{msg}
  {
  }
  virtual char const* what() const noexcept { return msg_.c_str(); }
 private:
  std::string msg_;
};

struct model_import_error : std::exception {
  model_import_error() : model_import_error("Error while importing model") {}
  model_import_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

struct type_error : std::exception {
  type_error() : type_error("Model cannot be used with given data type") {}
  type_error(char const* msg) : msg_{msg} {}
  virtual char const* what() const noexcept { return msg_; }

 private:
  char const* msg_;
};

}
}
}
