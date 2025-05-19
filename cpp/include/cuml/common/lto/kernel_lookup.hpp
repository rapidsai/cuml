/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "load_fatbins.hpp"

#include <cuda.h>

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

template <typename... Ts>
std::string make_db_key(Ts&&... args)
{
  // lets find the mangled name for `Ts`
  std::stringstream combined;
  (combined << ... << typeid(args).name());
  return combined.str();
}

template <typename... Ts>
std::string make_db_key()
{
  std::stringstream combined;
  (combined << ... << typeid(Ts).name());
  return combined.str();
}

struct KernelEntry {
  // Find the entry point in `lib` that patches this entry
  CUkernel get_kernel(CUlibrary lib) const;

  bool operator==(const KernelEntry& rhs) const { return launch_key == rhs.launch_key; }

  std::string launch_key{};
  std::string file{};
};

template <>
struct std::hash<KernelEntry> {
  std::size_t operator()(KernelEntry const& ke) const
  {
    return std::hash<std::string>{}(ke.launch_key);
  }
};

using KernelDatabase = std::unordered_set<KernelEntry>;

KernelEntry find_entry(KernelDatabase const& db, std::string const& kernel_name);

CUkernel get_kernel(KernelDatabase const& db, std::string const& kernel_name);
