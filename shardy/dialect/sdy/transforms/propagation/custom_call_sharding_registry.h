/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_CUSTOM_CALL_SHARDING_REGISTRY_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_CUSTOM_CALL_SHARDING_REGISTRY_H_

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Mutex.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
namespace mlir {
namespace sdy {

// A registry for custom-call sharding information.
//
// This registry records the name of a custom-call and its corresponding
// callback function that either inspects the custom-call op to produce an
// `OpShardingRuleAttr` or update the `TensorShardingAttr` for the operands
// and results of the op.
//
// A process can have at most one globally shared instance of this object. This
// instance is created lazily the first time the registry is accessed.
//
// This class is thread-safe.
class CustomCallShardingRegistry {
 public:
  // A ShardingRuleCallBack is a function that inspects a custom-call op and
  // returns an `OpShardingRuleAttr` that specifies the sharding rule for the
  // op.
  using ShardingRuleCallBack =
      std::function<OpShardingRuleAttr(mlir::Operation*)>;
  // A ShardingPropagationCallBack is a function that takes a custom-call op
  // and updates the `TensorShardingAttr` for the operands and results of the
  // op.
  using ShardingPropagationCallBack =
      std::function<void(mlir::Operation*, PropagationDirection direction)>;
  // A ShardingRegistry is either a ShardingRuleCallBack or a
  // ShardingPropagationCallBack.
  using ShardingRegistry =
      std::variant<ShardingRuleCallBack, ShardingPropagationCallBack>;

  static LogicalResult RegisterShardingRuleCallBack(
      const std::string& op_name, ShardingRuleCallBack call_back_func);
  static std::optional<ShardingRuleCallBack> GetShardingRuleCallBack(
      const std::string& op_name);

  static LogicalResult RegisterShardingPropagationCallBack(
      const std::string& op_name, ShardingPropagationCallBack call_back_func);
  static std::optional<ShardingPropagationCallBack>
  GetShardingPropagationCallBack(const std::string& op_name);

 private:
  static CustomCallShardingRegistry& GetRegistry();

  template <typename T>
  static std::optional<T> GetCallBack(const std::string& op_name) {
    CustomCallShardingRegistry& registry = GetRegistry();
    llvm::sys::ScopedLock scopedLock(registry.mutex_);
    if (auto iter = registry.name_to_call_back_.find(op_name);
        iter != registry.name_to_call_back_.end()) {
      if (std::holds_alternative<T>(iter->second)) {
        return std::get<T>(iter->second);
      }
    }
    return std::nullopt;
  }

  template <typename T>
  static LogicalResult RegisterCallBack(const std::string& op_name,
                                        T call_back_func) {
    CustomCallShardingRegistry& registry = GetRegistry();
    llvm::sys::ScopedLock scopedLock(registry.mutex_);
    auto [it, emplaced] = registry.name_to_call_back_.try_emplace(
        op_name, std::move(call_back_func));
    if (!emplaced) {
      return failure();
    }
    return success();
  }

 private:
  llvm::sys::Mutex mutex_;
  std::unordered_map<std::string, ShardingRegistry> name_to_call_back_;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_CUSTOM_CALL_SHARDING_REGISTRY_H_
