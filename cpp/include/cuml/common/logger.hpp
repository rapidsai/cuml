/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <stdarg.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include <cuml/common/log_levels.hpp>

namespace spdlog {
class logger;
namespace sinks {
template <class Mutex>
class CallbackSink;
using callback_sink_mt = CallbackSink<std::mutex>;
};  // namespace sinks
};  // namespace spdlog

namespace ML {

/**
 * @defgroup CStringFormat Expand a C-style format string
 *
 * @brief Expands C-style formatted string into std::string
 *
 * @param[in] fmt format string
 * @param[in] vl  respective values for each of format modifiers in the string
 *
 * @return the expanded `std::string`
 *
 * @{
 */
std::string format(const char* fmt, va_list& vl);
std::string format(const char* fmt, ...);
/** @} */

/**
 * @brief The main Logging class for cuML library.
 *
 * This class acts as a thin wrapper over the underlying `spdlog` interface. The
 * design is done in this way in order to avoid us having to also ship `spdlog`
 * header files in our installation.
 *
 * @todo This currently only supports logging to stdout. Need to add support in
 *       future to add custom loggers as well [Issue #2046]
 */
class Logger {
 public:
  /**
   * @brief Singleton method to get the underlying logger object
   *
   * @return the singleton logger object
   */
  static Logger& get();

  /**
   * @brief Set the logging level.
   *
   * Only messages with level equal or above this will be printed
   *
   * @param[in] level logging level
   *
   * @note The log level will actually be set only if the input is within the
   *       range [CUML_LEVEL_TRACE, CUML_LEVEL_OFF]. If it is not, then it'll
   *       be ignored. See documentation of decisiontree for how this gets used
   */
  void setLevel(int level);

  /**
   * @brief Set the logging pattern
   *
   * @param[in] pattern the pattern to be set. Refer this link
   *                    https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
   *                    to know the right syntax of this pattern
   */
  void setPattern(const std::string& pattern);

  /**
   * @brief Register a callback function to be run in place of usual log call
   *
   * @param[in] callback the function to be run on all logged messages
   */
  void setCallback(void (*callback)(int lvl, const char* msg));

  /**
   * @brief Register a flush function compatible with the registered callback
   *
   * @param[in] flush the function to use when flushing logs
   */
  void setFlush(void (*flush)());

  /**
   * @brief Tells whether messages will be logged for the given log level
   *
   * @param[in] level log level to be checked for
   * @return true if messages will be logged for this level, else false
   */
  bool shouldLogFor(int level) const;

  /**
   * @brief Query for the current log level
   *
   * @return the current log level
   */
  int getLevel() const;

  /**
   * @brief Get the current logging pattern
   * @return the pattern
   */
  std::string getPattern() const { return currPattern; }

  /**
   * @brief Main logging method
   *
   * @param[in] level logging level of this message
   * @param[in] fmt   C-like format string, followed by respective params
   */
  void log(int level, const char* fmt, ...);

  /**
   * @brief Flush logs by calling flush on underlying logger
   */
  void flush();

 private:
  Logger();
  ~Logger() {}

  std::shared_ptr<spdlog::sinks::callback_sink_mt> sink;
  std::shared_ptr<spdlog::logger> logger;
  std::string currPattern;
  static const std::string DefaultPattern;
};  // class Logger

/**
 * @brief RAII based pattern setter for Logger class
 *
 * @code{.cpp}
 * {
 *   PatternSetter _("%l -- %v");
 *   CUML_LOG_INFO("Test message\n");
 * }
 * @endcode
 */
class PatternSetter {
 public:
  /**
   * @brief Set the pattern for the rest of the log messages
   * @param[in] pattern pattern to be set
   */
  PatternSetter(const std::string& pattern = "%v");

  /**
   * @brief This will restore the previous pattern that was active during the
   *        moment this object was created
   */
  ~PatternSetter();

 private:
  std::string prevPattern;
};  // class PatternSetter

/**
 * @defgroup LoggerMacros Helper macros for dealing with logging
 * @{
 */
#if (CUML_ACTIVE_LEVEL >= CUML_LEVEL_TRACE)
#define CUML_LOG_TRACE(fmt, ...)                               \
  do {                                                         \
    std::stringstream ss;                                      \
    ss << ML::format("%s:%d ", __FILE__, __LINE__);            \
    ss << ML::format(fmt, ##__VA_ARGS__);                      \
    ML::Logger::get().log(CUML_LEVEL_TRACE, ss.str().c_str()); \
  } while (0)
#else
#define CUML_LOG_TRACE(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL >= CUML_LEVEL_DEBUG)
#define CUML_LOG_DEBUG(fmt, ...)                               \
  do {                                                         \
    std::stringstream ss;                                      \
    ss << ML::format("%s:%d ", __FILE__, __LINE__);            \
    ss << ML::format(fmt, ##__VA_ARGS__);                      \
    ML::Logger::get().log(CUML_LEVEL_DEBUG, ss.str().c_str()); \
  } while (0)
#else
#define CUML_LOG_DEBUG(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL >= CUML_LEVEL_INFO)
#define CUML_LOG_INFO(fmt, ...) ML::Logger::get().log(CUML_LEVEL_INFO, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_INFO(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL >= CUML_LEVEL_WARN)
#define CUML_LOG_WARN(fmt, ...) ML::Logger::get().log(CUML_LEVEL_WARN, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_WARN(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL >= CUML_LEVEL_ERROR)
#define CUML_LOG_ERROR(fmt, ...) ML::Logger::get().log(CUML_LEVEL_ERROR, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_ERROR(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL >= CUML_LEVEL_CRITICAL)
#define CUML_LOG_CRITICAL(fmt, ...) ML::Logger::get().log(CUML_LEVEL_CRITICAL, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_CRITICAL(fmt, ...) void(0)
#endif
/** @} */

};  // namespace ML
