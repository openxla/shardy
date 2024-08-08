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

#include "shardy/dialect/sdy/transforms/propagation/custom_call_sharding_rule_registry.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sdy {

namespace {
static llvm::ManagedStatic<CustomCallShardingRuleRegistry> cc_registry;
}  // namespace

std::optional<CustomCallShardingRuleRegistry::ShardingRuleCallBack>
CustomCallShardingRuleRegistry::GetShardingRuleCallBack(
    const std::string& op_name) {
  llvm::sys::ScopedLock scopedLock(cc_registry->mutex_);
  if (auto iter = cc_registry->name_to_call_back_.find(op_name);
      iter != cc_registry->name_to_call_back_.end()) {
    return *iter->second;
  }
  return std::nullopt;
}

LogicalResult CustomCallShardingRuleRegistry::RegisterShardingRuleCallBack(
    const std::string& op_name, ShardingRuleCallBack call_back_func) {
  llvm::sys::ScopedLock scopedLock(cc_registry->mutex_);
  auto [it, emplaced] = cc_registry->name_to_call_back_.try_emplace(
      op_name,
      std::make_unique<ShardingRuleCallBack>(std::move(call_back_func)));
  if (!emplaced) {
    return failure();
  }
  return success();
}

}  // namespace sdy
}  // namespace mlir
