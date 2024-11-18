/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>

#include <gtest/gtest.h>

#include <string>

namespace ML {

TEST(Logger, Test)
{
  CUML_LOG_CRITICAL("This is a critical message");
  CUML_LOG_ERROR("This is an error message");
  CUML_LOG_WARN("This is a warning message");
  CUML_LOG_INFO("This is an info message");

  default_logger().set_level(ML::level_enum::warn);
  ASSERT_EQ(ML::level_enum::warn, default_logger().level());
  default_logger().set_level(ML::level_enum::info);
  ASSERT_EQ(ML::level_enum::info, default_logger().level());

  ASSERT_FALSE(default_logger().should_log(ML::level_enum::trace));
  ASSERT_FALSE(default_logger().should_log(ML::level_enum::debug));
  ASSERT_TRUE(default_logger().should_log(ML::level_enum::info));
  ASSERT_TRUE(default_logger().should_log(ML::level_enum::warn));
}

std::string logged = "";
void exampleCallback(int lvl, const char* msg) { logged = std::string(msg); }

int flushCount = 0;
void exampleFlush() { ++flushCount; }

class LoggerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    flushCount = 0;
    logged     = "";
    default_logger().set_level(ML::level_enum::trace);
  }

  void TearDown() override
  {
    default_logger().setCallback(nullptr);
    default_logger().setFlush(nullptr);
    default_logger().set_level(ML::level_enum::info);
  }
};

TEST_F(LoggerTest, callback)
{
  std::string testMsg;
  default_logger().setCallback(exampleCallback);

  testMsg = "This is a critical message";
  CUML_LOG_CRITICAL(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is an error message";
  CUML_LOG_ERROR(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is a warning message";
  CUML_LOG_WARN(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is an info message";
  CUML_LOG_INFO(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is a debug message";
  CUML_LOG_DEBUG(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);
}

TEST_F(LoggerTest, flush)
{
  default_logger().setFlush(exampleFlush);
  default_logger().flush();
  ASSERT_EQ(1, flushCount);
}

}  // namespace ML
