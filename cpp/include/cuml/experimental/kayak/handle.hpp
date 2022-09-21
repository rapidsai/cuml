#pragma once
#include <algorithm>
#include <cstddef>
#include <cuml/experimental/kayak/cuda_stream.hpp>
#ifdef ENABLE_GPU
#include <raft/core/handle.hpp>
#endif

namespace kayak {
#ifdef ENABLE_GPU
struct handle_t {
  handle_t(raft::handle_t const* handle_ptr=nullptr) : raft_handle_{handle_ptr} {}
  handle_t(raft::handle_t const& raft_handle) : raft_handle_{&raft_handle} {}
  auto get_next_usable_stream() const {
    return kayak::cuda_stream{raft_handle_->get_next_usable_stream().value()};
  }
  auto get_stream_pool_size() const {
    return raft_handle_->get_stream_pool_size();
  }
  auto get_usable_stream_count() const {
    return std::min(get_stream_pool_size(), std::size_t{1});
  }
  void synchronize() const {
    raft_handle_->sync_stream_pool();
    raft_handle_->sync_stream();
  }
 private:
  // Have to store a pointer because handle is not movable
  raft::handle_t const* raft_handle_;
};
#else
struct handle_t {
  auto get_next_usable_stream() const {
    return kayak::cuda_stream{};
  }
  auto get_stream_pool_size() const {
    return std::size_t{};
  }
  auto get_usable_stream_count() const {
    return std::min(get_stream_pool_size(), std::size_t{1});
  }
  void synchronize() const {
  }
};
#endif
}

