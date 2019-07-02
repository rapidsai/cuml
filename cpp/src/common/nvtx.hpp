/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <stdint.h>
#include <string>

namespace ML {

/**
 * @brief Helper method to generate 'visually distinct' colors.
 * Inspired from https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
 * However, if an associated tag is passed, it will look up in its history for
 * any generated color against this tag and if found, just returns it, else
 * generates a new color, assigns a tag to it and stores it for future usage.
 * Such a thing is very useful for nvtx markers where the ranges associated
 * with a specific tag should ideally get the same color for the purpose of
 * visualizing it on nsight-systems timeline.
 * @param tag look for any previously generated colors with this tag or
 * associate the currently generated color with it
 * @return returns 32b RGB integer with alpha channel set of 0xff
 */
uint32_t generateNextColor(const std::string &tag = "");

#ifdef NVTX_ENABLED

#include <nvToolsExt.h>

#define PUSH_RANGE(name)                               \
  do {                                                 \
    nvtxEventAttributes_t eventAttrib = {0};           \
    eventAttrib.version = NVTX_VERSION;                \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
    eventAttrib.colorType = NVTX_COLOR_ARGB;           \
    eventAttrib.color = generateNextColor(name);       \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name;                  \
    nvtxRangePushEx(&eventAttrib);                     \
  } while (0)

#define POP_RANGE() nvtxRangePop()

#else  // NVTX_ENABLED

#define PUSH_RANGE(name)
#define POP_RANGE()

#endif  // NVTX_ENABLED

}  // end namespace ML
