/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#define SPDLOG_HEADER_ONLY
#include <spdlog/sinks/stdout_color_sinks.h>  // NOLINT
#include <spdlog/spdlog.h>                    // NOLINT

#include <algorithm>
#include <cuml/common/callbackSink.hpp>
#include <cuml/common/logger.hpp>
#include <memory>

namespace ML {

std::string format(const char* fmt, va_list& vl)
{
  char buf[4096];
  vsnprintf(buf, sizeof(buf), fmt, vl);
  return std::string(buf);
}

std::string format(const char* fmt, ...)
{
  va_list vl;
  va_start(vl, fmt);
  std::string str = format(fmt, vl);
  va_end(vl);
  return str;
}

int convert_level_to_spdlog(int level)
{
  level = std::max(CUML_LEVEL_OFF, std::min(CUML_LEVEL_TRACE, level));
  return CUML_LEVEL_TRACE - level;
}

const std::string Logger::DefaultPattern("[%L] [%H:%M:%S.%f] %v");

Logger& Logger::get()
{
  static Logger logger;
  return logger;
}

Logger::Logger()
  : sink{std::make_shared<spdlog::sinks::callback_sink_mt>()},
    logger{std::make_shared<spdlog::logger>("cuml", sink)},
    currPattern()
{
  setPattern(DefaultPattern);
  setLevel(CUML_LEVEL_INFO);
}

void Logger::setLevel(int level)
{
  level = convert_level_to_spdlog(level);
  logger->set_level(static_cast<spdlog::level::level_enum>(level));
}

void Logger::setPattern(const std::string& pattern)
{
  currPattern = pattern;
  logger->set_pattern(pattern);
}

void Logger::setCallback(spdlog::sinks::LogCallback callback) { sink->set_callback(callback); }

void Logger::setFlush(void (*flush)()) { sink->set_flush(flush); }

bool Logger::shouldLogFor(int level) const
{
  level        = convert_level_to_spdlog(level);
  auto level_e = static_cast<spdlog::level::level_enum>(level);
  return logger->should_log(level_e);
}

int Logger::getLevel() const
{
  auto level_e = logger->level();
  return CUML_LEVEL_TRACE - static_cast<int>(level_e);
}

void Logger::log(int level, const char* fmt, ...)
{
  level        = convert_level_to_spdlog(level);
  auto level_e = static_cast<spdlog::level::level_enum>(level);
  // explicit check to make sure that we only expand messages when required
  if (logger->should_log(level_e)) {
    va_list vl;
    va_start(vl, fmt);
    auto msg = format(fmt, vl);
    va_end(vl);
    logger->log(level_e, msg);
  }
}

void Logger::flush() { logger->flush(); }

PatternSetter::PatternSetter(const std::string& pattern) : prevPattern()
{
  prevPattern = Logger::get().getPattern();
  Logger::get().setPattern(pattern);
}

PatternSetter::~PatternSetter() { Logger::get().setPattern(prevPattern); }

}  // namespace ML
