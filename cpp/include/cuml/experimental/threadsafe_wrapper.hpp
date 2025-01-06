/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cuml/experimental/ordered_mutex.hpp>

#include <algorithm>
#include <atomic>
#include <memory>

namespace ML {
namespace experimental {

/* A proxy to an underlying object that holds a lock for its lifetime. This
 * ensures that the underlying object cannot be accessed unless the lock has
 * been acquired.
 */
template <typename T, typename L>
struct threadsafe_proxy {
  // Acquire ownership of the lock on construction
  threadsafe_proxy(T* wrapped, L&& lock) : wrapped_{wrapped}, lock_{std::move(lock)} {}
  // Do not allow copy or move. Proxy object is intended to be used
  // immediately.
  threadsafe_proxy(threadsafe_proxy const&) = delete;
  threadsafe_proxy(threadsafe_proxy&&)      = delete;
  // Access the wrapped object via -> operator
  auto* operator->() { return wrapped_; }

 private:
  T* wrapped_;
  L lock_;
};

/* This struct wraps an object which may be modified from some host threads
 * but accessed without modification from others. Because multiple users can safely
 * access the object simultaneously so long as it is not being modified, any
 * const access to a threadsafe_wrapper<T> will acquire a lock solely to
 * increment an atomic counter indicating that it is currently accessing the
 * underlying object. It will then decrement that counter once the const call
 * to the underlying object has been completed. Non-const access will
 * acquire a lock on the same underlying mutex but not proceed with the
 * non-const call until the counter reaches 0.
 *
 * A special lock (ordered_lock) ensures that the mutex is acquired in the
 * order that threads attempt to acquire it. This ensures that
 * modifying threads are not indefinitely delayed.
 *
 * Example usage:
 *
 * struct foo() {
 *   foo(int data) : data_{data} {}
 *   auto get_data() const { return data_; }
 *   void set_data(int new_data) { data_ = new_data; }
 *  private:
 *   int data_;
 * };
 *
 * auto f = threadsafe_wrapper<foo>{5};
 * f->set_data(6);
 * f->get_data();  // Safe but inefficient. Returns 6.
 * std::as_const(f)->get_data();  // Safe and efficient. Returns 6.
 * std::as_const(f)->set_data(7);  // Fails to compile.
 */
template <typename T>
struct threadsafe_wrapper {
  template <typename... Args>
  threadsafe_wrapper(Args&&... args) : wrapped{std::make_unique<T>(std::forward<Args>(args)...)}
  {
  }
  auto operator->()
  {
    return threadsafe_proxy<T*, modifier_lock>{wrapped.get(), modifier_lock{mtx_}};
  }
  auto operator->() const
  {
    return threadsafe_proxy<T const*, accessor_lock>{wrapped.get(), accessor_lock{mtx_}};
  }

 private:
  // A class for coordinating access to a resource that may be modified by some
  // threads and accessed without modification by others.
  class modification_mutex {
    // Wait until all ongoing const access has completed and do not allow
    // additional const or non-const access to begin until the modifying lock on this mutex has been
    // released.
    void acquire_for_modifier()
    {
      // Prevent any new users from incrementing work counter
      lock_ = std::make_unique<ordered_lock>(mtx_);
      // Wait until all work in progress is done
      while (currently_using_.load() != 0)
        ;
      std::atomic_thread_fence(std::memory_order_acquire);
    }
    // Allow other threads to initiate const or non-const access again
    void release_from_modifier() { lock_.reset(); }
    // Wait until ongoing non-const access has completed, then increment a
    // counter indicating the number of threads performing const access
    void acquire_for_access() const
    {
      auto tmp_lock = ordered_lock{mtx_};
      ++currently_using_;
    }
    // Decrement counter of the number of threads performing const access
    void release_from_accessor() const
    {
      std::atomic_thread_fence(std::memory_order_release);
      --currently_using_;
    }
    mutable ordered_mutex mtx_{};
    mutable std::atomic<int> currently_using_{};
    mutable std::unique_ptr<ordered_lock> lock_{nullptr};
    friend struct modifier_lock;
    friend struct accessor_lock;
  };

  // A lock acquired to modify the wrapped object. While this lock is acquired,
  // no other thread can perform const or non-const access to the underlying
  // object.
  struct modifier_lock {
    modifier_lock(modification_mutex& mtx)
      : mtx_{[&mtx]() {
          mtx.acquire_for_modifier();
          return &mtx;
        }()}
    {
    }
    ~modifier_lock() { mtx_->release_from_modifier(); }

   private:
    modification_mutex* mtx_;
  };

  // A lock acquired to access but not modify the wrapped object. We ensure that
  // only const methods can be accessed while protected by this lock. While
  // this lock is acquired, no other thread can perform non-const access, but
  // other threads may perform const access.
  struct accessor_lock {
    accessor_lock(modification_mutex const& mtx)
      : mtx_{[&mtx]() {
          mtx.acquire_for_access();
          return &mtx;
        }()}
    {
    }
    ~accessor_lock() { mtx_->release_from_accessor(); }

   private:
    modification_mutex const* mtx_;
  };
  modification_mutex mtx_;
  std::unique_ptr<T> wrapped;
};

}  // namespace experimental
}  // namespace ML
