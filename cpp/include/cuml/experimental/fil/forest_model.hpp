#pragma once
#include <type_traits>
#include <variant>
#include <herring3/decision_forest.hpp>
#include <herring3/detail/index_type.hpp>

namespace herring {

struct forest_model {
  forest_model(
    decision_forest_variant&& forest = decision_forest_variant{}
  ) : decision_forest_{forest} {}

  auto num_feature() {
    return std::visit([this](auto&& concrete_forest) {
      return concrete_forest.num_feature();
    }, decision_forest_);
  }

  auto num_outputs() {
    return std::visit([this](auto&& concrete_forest) {
      return concrete_forest.num_outputs();
    }, decision_forest_);
  }

  auto memory_type() {
    return std::visit([this](auto&& concrete_forest) {
      return concrete_forest.memory_type();
    }, decision_forest_);
  }

  auto device_index() {
    return std::visit([this](auto&& concrete_forest) {
      return concrete_forest.device_index();
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
        throw type_error(
          "Input type does not match model_type"
        );
      }
    }, decision_forest_);
  }
 private:
  decision_forest_variant decision_forest_;
};

}
