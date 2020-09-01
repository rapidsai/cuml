#pragma once

#include <iostream>
#include <mutex>

#define SPDLOG_HEADER_ONLY
#include <spdlog/sinks/base_sink.h>
#include <spdlog/common.h>
#include <spdlog/details/log_msg.h>

namespace spdlog {
namespace sinks {

typedef void(*LogCallback)(int lvl, const char* msg);

template<class Mutex>
class CallbackSink : public base_sink<Mutex> {
 public:
  explicit CallbackSink(std::string tag = "spdlog",
                        LogCallback callback = nullptr,
                        void(*flush)() = nullptr) : _callback{callback}, _flush{flush}
  {
  };

  void set_callback(LogCallback callback) {
    _callback = callback;
  }
  void set_flush(void(*flush)()) {
    _flush = flush;
  }

 protected:
  void sink_it_(const details::log_msg& msg) override {
    spdlog::memory_buf_t formatted;
    base_sink<Mutex>::formatter_->format(msg, formatted);
    std::string msg_string = fmt::to_string(formatted);

    if (_callback) {
      _callback(static_cast<int>(msg.level), msg_string.c_str());
    } else {
      std::cout << msg_string;
    }
  }

  void flush_() override {
    if (_flush) {
      _flush();
    } else {
      std::cout << std::flush;
    }
  }

  LogCallback _callback;
  void (*_flush)();
};

using callback_sink_mt = CallbackSink<std::mutex>;
using callback_sink_st = CallbackSink<details::null_mutex>;

}  // end namespace sinks
}  // end namespace spdlog
