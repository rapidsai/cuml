/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <string>

namespace spdlog {
class logger;
};

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
 * @defgroup CumlLogLevels Logging levels used in cuML
 *
 * @note exactly match the corresponding ones in spdlog for wrapping purposes
 *
 * @{
 */
#define CUML_LEVEL_TRACE 0
#define CUML_LEVEL_DEBUG 1
#define CUML_LEVEL_INFO 2
#define CUML_LEVEL_WARN 3
#define CUML_LEVEL_ERROR 4
#define CUML_LEVEL_CRITICAL 5
#define CUML_LEVEL_OFF 6
/** @} */

#if !defined(CUML_ACTIVE_LEVEL)
#define CUML_ACTIVE_LEVEL CUML_LEVEL_INFO
#endif

/**
 * @brief The main Logging class for cuML library.
 *
 * This class acts as a thin wrapper over the underlying `spdlog` interface. The
 * design is done in this way in order to avoid us having to also ship `spdlog`
 * header files in our installation.
 *
 * @todo This currently only supports logging to stdout. Need to add support in
 *       future to add custom loggers as well
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
   * @brief Main logging method
   *
   * @param[in] level logging level of this message
   * @param[in] fmt   C-like format string, followed by respective params
   */
  void log(int level, const char* fmt, ...);

 private:
  Logger();
  ~Logger() {}

  std::shared_ptr<spdlog::logger> logger;
  static const std::string DefaultPattern;
};  // class Logger

/**
 * @defgroup LoggerMacros Helper macros for dealing with logging
 * @{
 */
#if (CUML_ACTIVE_LEVEL <= CUML_LEVEL_TRACE)
#define CUML_LOG_TRACE(fmt, ...)                                        \
  do {                                                                  \
    auto msg = format("%s:%d " fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    ML::Logger::get().log(CUML_LEVEL_TRACE, msg);                       \
  } while (0)
#else
#define CUML_LOG_TRACE(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL <= CUML_LEVEL_DEBUG)
#define CUML_LOG_DEBUG(fmt, ...)                                        \
  do {                                                                  \
    auto msg = format("%s:%d " fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    ML::Logger::get().log(CUML_LEVEL_DEBUG, msg);                       \
  } while (0)
#else
#define CUML_LOG_DEBUG(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL <= CUML_LEVEL_INFO)
#define CUML_LOG_INFO(fmt, ...) \
  ML::Logger::get().log(CUML_LEVEL_INFO, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_INFO(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL <= CUML_LEVEL_WARN)
#define CUML_LOG_WARN(fmt, ...) \
  ML::Logger::get().log(CUML_LEVEL_WARN, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_WARN(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL <= CUML_LEVEL_ERROR)
#define CUML_LOG_ERROR(fmt, ...) \
  ML::Logger::get().log(CUML_LEVEL_ERROR, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_ERROR(fmt, ...) void(0)
#endif

#if (CUML_ACTIVE_LEVEL <= CUML_LEVEL_CRITICAL)
#define CUML_LOG_CRITICAL(fmt, ...) \
  ML::Logger::get().log(CUML_LEVEL_CRITICAL, fmt, ##__VA_ARGS__)
#else
#define CUML_LOG_CRITICAL(fmt, ...) void(0)
#endif
/** @} */

};  // namespace ML
