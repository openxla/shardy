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

#include "shardy/dialect/sdy/transforms/propagation/custom_call_sharding_registry.h"

#include <optional>
#include <string>
#include <utility>

#include "llvm/Support/ManagedStatic.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace sdy {

namespace {
static llvm::ManagedStatic<CustomCallShardingRegistry> cc_registry;
}  // namespace

CustomCallShardingRegistry& CustomCallShardingRegistry::GetRegistry() {
  return *cc_registry;
}

std::optional<CustomCallShardingRegistry::ShardingRuleCallBack>
CustomCallShardingRegistry::GetShardingRuleCallBack(
    const std::string& op_name) {
  return GetCallBack<ShardingRuleCallBack>(op_name);
}

LogicalResult CustomCallShardingRegistry::RegisterShardingRuleCallBack(
    const std::string& op_name, ShardingRuleCallBack call_back_func) {
  return RegisterCallBack<ShardingRuleCallBack>(op_name,
                                                std::move(call_back_func));
}

std::optional<CustomCallShardingRegistry::ShardingPropagationCallBack>
CustomCallShardingRegistry::GetShardingPropagationCallBack(
    const std::string& op_name) {
  return GetCallBack<ShardingPropagationCallBack>(op_name);
}

LogicalResult CustomCallShardingRegistry::RegisterShardingPropagationCallBack(
    const std::string& op_name, ShardingPropagationCallBack call_back_func) {
  return RegisterCallBack<ShardingPropagationCallBack>(
      op_name, std::move(call_back_func));
}

}  // namespace sdy
}  // namespace mlir
