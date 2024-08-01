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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_CUSTOM_CALL_SHARDING_RULE_REGISTRY_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_CUSTOM_CALL_SHARDING_RULE_REGISTRY_H_

#include <functional>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Mutex.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
namespace mlir {
namespace sdy {

// A registry for custom-call sharding rule information.
//
// This registry records the name of a custom-call and its corresponding
// callback function that inspects the custom-call op to produce an
// `OpShardingRuleAttr`.
//
// A process can have at most one globally shared instance of this object. This
// instance is created lazily the first time the registry is accessed.
//
// This class is thread-safe.
class CustomCallShardingRuleRegistry {
 public:
  using ShardingRuleCallBack =
      std::function<OpShardingRuleAttr(mlir::Operation*)>;

  static LogicalResult RegisterShardingRuleCallBack(
      const std::string& op_name, ShardingRuleCallBack call_back_func);

  static std::optional<ShardingRuleCallBack> GetShardingRuleCallBack(
      const std::string& op_name);

 private:
  llvm::sys::Mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<ShardingRuleCallBack>>
      name_to_call_back_;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_CUSTOM_CALL_SHARDING_RULE_REGISTRY_H_
