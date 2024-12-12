/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <condition_variable>
#include <mutex>
#include <queue>

namespace ML {
namespace experimental {

/* A mutex which yields to threads in the order in which they attempt to
 * acquire a lock.
 *
 * Note that this order is somewhat ambiguously defined. If one thread has a lock on this mutex and
 * several other threads simultaneously attempt to acquire it, they will do so in the order in which
 * they are able to acquire the lock on the underlying raw mutex. What
 * ordered_mutex ensures is that if it is locked and several threads attempt to
 * acquire it in unambiguously serial fashion (i.e. one does not make the
 * attempt until a previous one has released the underlying raw mutex), those
 * threads will acquire the lock in the same order.
 *
 * In particular, this mutex is useful to ensure that a thread's acquisition of
 * a lock is not indefinitely deferred by other threads' acquisitions. If N
 * threads attempt to simultaneously lock the ordered_mutex, and then N-1
 * threads successfully acquire it, the remaining thread is guaranteed to get
 * the lock next before any of the other N-1 threads get the lock again.
 */
struct ordered_mutex {
  void lock()
  {
    auto scoped_lock = std::unique_lock<std::mutex>{raw_mtx_};
    if (locked_) {
      // Another thread is using this mutex, so get in line and wait for
      // another thread to notify this one to continue.
      auto thread_condition = std::condition_variable{};
      control_queue_.push(&thread_condition);
      thread_condition.wait(scoped_lock);
    } else {
      // No other threads have acquired the ordered_mutex, so we will not wait
      // for another thread to notify this one that it is its turn
      locked_ = true;
    }
  }

  void unlock()
  {
    auto scoped_lock = std::unique_lock<std::mutex>{raw_mtx_};
    if (control_queue_.empty()) {
      // No waiting threads, so the next thread that attempts to acquire may
      // simply proceed.
      locked_ = false;
    } else {
      // We must notify under the scoped_lock to avoid having a new thread
      // acquire the raw mutex before a waiting thread gets notified.
      control_queue_.front()->notify_one();
      control_queue_.pop();
    }
  }

 private:
  // Use a pointer here rather than storing the object in the queue to ensure
  // that the variable is not deallocated while it is being used.
  std::queue<std::condition_variable*> control_queue_{};
  std::mutex raw_mtx_{};
  bool locked_ = false;
};

/* A scoped lock based on ordered_mutex, which will be acquired in the order in which
 * threads attempt to acquire the underlying mutex */
struct ordered_lock {
  explicit ordered_lock(ordered_mutex& mtx)
    : mtx_{[&mtx]() {
        mtx.lock();
        return &mtx;
      }()}
  {
  }

  ~ordered_lock() { mtx_->unlock(); }

 private:
  ordered_mutex* mtx_;
};

}  // namespace experimental
}  // namespace ML
