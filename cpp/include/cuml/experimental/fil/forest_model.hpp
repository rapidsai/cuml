#pragma once
#include <cstddef>
#include <type_traits>
#include <variant>
#include <cuml/experimental/fil/decision_forest.hpp>
#include <cuml/experimental/fil/detail/index_type.hpp>
#include <cuml/experimental/kayak/buffer.hpp>
#include <cuml/experimental/kayak/gpu_support.hpp>
#include <cuml/experimental/kayak/handle.hpp>

namespace ML {
namespace experimental {
namespace fil {

struct forest_model {
  forest_model(
    decision_forest_variant&& forest = decision_forest_variant{}
  ) : decision_forest_{forest} {}

  auto num_feature() {
    return std::visit([](auto&& concrete_forest) {
      return concrete_forest.num_feature();
    }, decision_forest_);
  }

  auto num_outputs() {
    return std::size_t{std::visit([](auto&& concrete_forest) {
      return concrete_forest.num_outputs();
    }, decision_forest_)};
  }

  auto memory_type() {
    return std::visit([](auto&& concrete_forest) {
      return concrete_forest.memory_type();
    }, decision_forest_);
  }

  auto device_index() {
    return std::visit([](auto&& concrete_forest) {
      return concrete_forest.device_index();
    }, decision_forest_);
  }

  auto is_double_precision() {
    return std::visit([](auto&& concrete_forest) {
      return std::is_same_v<
        typename std::remove_reference_t<decltype(concrete_forest)>::io_type, double
      >;
    }, decision_forest_);
  }

  template <typename io_t>
  void predict(
    kayak::buffer<io_t>& output,
    kayak::buffer<io_t> const& input,
    kayak::cuda_stream stream = kayak::cuda_stream{},
    std::optional<index_type> specified_chunk_size=std::nullopt
  ) {
    std::visit([this, &output, &input, &stream, &specified_chunk_size](auto&& concrete_forest) {
      if constexpr(std::is_same_v<typename std::remove_reference_t<decltype(concrete_forest)>::io_type, io_t>) {
        concrete_forest.predict(output, input, stream, specified_chunk_size);
      } else {
        throw type_error("Input type does not match model_type");
      }
    }, decision_forest_);
  }

  template <typename io_t>
  void predict(
    kayak::handle_t const& handle,
    kayak::buffer<io_t>& output,
    kayak::buffer<io_t> const& input,
    std::optional<index_type> specified_chunk_size=std::nullopt
  ) {
    std::visit([this, &handle, &output, &input, &specified_chunk_size](auto&& concrete_forest) {
      using model_io_t = typename std::remove_reference_t<decltype(concrete_forest)>::io_type;
      if constexpr(std::is_same_v<model_io_t, io_t>) {
        if (output.memory_type() == memory_type() && input.memory_type() == memory_type()) {
          concrete_forest.predict(
            output,
            input,
            handle.get_next_usable_stream(),
            specified_chunk_size
          );
        } else {
          auto constexpr static const MIN_CHUNKS_PER_PARTITION = std::size_t{64};
          auto constexpr static const MAX_CHUNK_SIZE = std::size_t{64};

          auto row_count = input.size() / num_feature();
          auto partition_size = std::max(
            kayak::ceildiv(row_count, handle.get_usable_stream_count()),
            specified_chunk_size.value_or(MAX_CHUNK_SIZE) * MIN_CHUNKS_PER_PARTITION
          );
          auto partition_count = kayak::ceildiv(row_count, partition_size);
          for (auto i = std::size_t{}; i < partition_count; ++i) {
            auto stream = handle.get_next_usable_stream();
            auto rows_in_this_partition = std::min(partition_size, row_count - i * partition_size);
            auto partition_in = kayak::buffer<io_t>{};
            if (input.memory_type() != memory_type()) {
              partition_in = kayak::buffer<io_t>{
                rows_in_this_partition * num_feature(),
                memory_type()
              };
              kayak::copy<kayak::DEBUG_ENABLED>(
                partition_in,
                input,
                0,
                i * partition_size * num_feature(),
                partition_in.size(),
                stream
              );
            } else {
              partition_in = kayak::buffer<io_t>{
                input.data() + i * partition_size * num_feature(),
                rows_in_this_partition * num_feature(),
                memory_type()
              };
            }
            auto partition_out = kayak::buffer<io_t>{};
            if (output.memory_type() != memory_type()) {
              partition_out = kayak::buffer<io_t>{
                rows_in_this_partition * num_outputs(),
                memory_type()
              };
            } else {
              partition_out = kayak::buffer<io_t>{
                output.data() + i * partition_size * num_outputs(),
                rows_in_this_partition * num_outputs(),
                memory_type()
              };
            }
            concrete_forest.predict(
              partition_out,
              partition_in,
              stream,
              specified_chunk_size
            );
            if (output.memory_type() != memory_type()) {
              kayak::copy<kayak::DEBUG_ENABLED>(
                output,
                partition_out,
                i * partition_size * num_outputs(),
                0,
                partition_out.size(),
                stream
              );
            }
          }
        }
      } else {
        throw type_error("Input type does not match model_type");
      }
    }, decision_forest_);
  }

  template <typename io_t>
  void predict(
    kayak::handle_t const& handle,
    io_t* output,
    io_t* input,
    std::size_t num_rows,
    kayak::device_type out_mem_type,
    kayak::device_type in_mem_type,
    std::optional<index_type> specified_chunk_size=std::nullopt
  ) {
    // TODO(wphicks): Make sure buffer lands on same device as model
    auto out_buffer = kayak::buffer{
      output,
      num_rows * num_outputs(),
      out_mem_type
    };
    auto in_buffer = kayak::buffer{
      input,
      num_rows * num_feature(),
      in_mem_type
    };
    predict(handle, out_buffer, in_buffer, specified_chunk_size);
  }

 private:
  decision_forest_variant decision_forest_;
};

}
}
}
