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

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

namespace {

using CustomCallShardingRegistryTest = ::testing::Test;

TEST_F(CustomCallShardingRegistryTest, SimplyRegister) {
  auto callBackFunc = [](mlir::Operation* op) { return OpShardingRuleAttr(); };
  const char kCustomCallOpName[] = "sdy_testonly1";

  std::optional<CustomCallShardingRegistry::ShardingRuleCallBack> queryResult =
      CustomCallShardingRegistry::GetShardingRuleCallBack(kCustomCallOpName);
  EXPECT_FALSE(queryResult.has_value());

  LogicalResult registerResult =
      CustomCallShardingRegistry::RegisterShardingRuleCallBack(
          kCustomCallOpName, callBackFunc);
  EXPECT_TRUE(registerResult.succeeded());

  queryResult =
      CustomCallShardingRegistry::GetShardingRuleCallBack(kCustomCallOpName);
  EXPECT_TRUE(queryResult.has_value());

  registerResult = CustomCallShardingRegistry::RegisterShardingRuleCallBack(
      kCustomCallOpName, callBackFunc);
  EXPECT_TRUE(registerResult.failed());
}

}  // namespace

}  // namespace sdy
}  // namespace mlir
