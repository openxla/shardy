/* Copyright 2026 The Shardy Authors.

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

#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

#include <cstddef>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Threading.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

using ::testing::Each;
using ::testing::ElementsAre;

constexpr StringRef kProgram = R"mlir(
  sdy.mesh @mesh = <["x"=2, "y"=2]>
  sdy.mesh @mesh_2 = <["x"=2, "y"=2]>

  func.func @main(
      %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
      -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
    %0 = stablehlo.add %arg0, %arg0 {
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>
    } : tensor<4x4xf32>
    %1:2 = stablehlo.custom_call @foo(%0) {
      sdy.sharding = #sdy.sharding_per_value<[
        <@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>
      ]>
    } : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
    %2 = sdy.manual_computation(%1#0)
        in_shardings=[<@mesh, [{"x"}, {}]>]
        out_shardings=[<@mesh, [{"x"}, {}]>]
        manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
      sdy.return %arg1 : tensor<2x4xf32>
    } : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = sdy.named_computation<"bar">(%2)
        in_shardings=[<@mesh, [{"x"}, {}]>]
        out_shardings=[<@mesh, [{}, {"y"}]>] (%arg1: tensor<4x4xf32>) {
      sdy.return %arg1 : tensor<4x4xf32>
    } : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %3 : tensor<4x4xf32>
  }
)mlir";

using ShardingWalkerTest = ShardyTestBase;

TEST_F(ShardingWalkerTest, ConcurrentWalkShardingsIsThreadSafe) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  ASSERT_TRUE(module);
  constexpr size_t kNumTasks = 64;
  SmallVector<int> counts(kNumTasks, 0);

  mlir::parallelFor(&context, 0, kNumTasks, [&](size_t i) {
    walkShardings(module.get(), [&](TensorShardingAttr) { ++counts[i]; });
  });

  EXPECT_THAT(counts, Each(9));
}

TEST_F(ShardingWalkerTest, TransformShardingsUpdatesMatchingShardings) {
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  ASSERT_TRUE(module);
  FlatSymbolRefAttr mesh2Ref = FlatSymbolRefAttr::get(&context, "mesh_2");

  transformShardings(module.get(), [&](TensorShardingAttr sharding) {
    if (sharding.getRank() > 1 && !sharding.getDimSharding(1).emptyAxes()) {
      return TensorShardingAttr::get(
          &context, mesh2Ref, sharding.getDimShardings(),
          sharding.getReplicatedAxes(), sharding.getUnreducedAxes());
    }
    return sharding;
  });

  SmallVector<StringRef> meshNames;
  walkShardings(module.get(), [&](TensorShardingAttr sharding) {
    meshNames.push_back(sharding.getMeshName());
  });
  EXPECT_THAT(meshNames, ElementsAre("mesh", "mesh", "mesh", "mesh", "mesh_2",
                                     "mesh", "mesh", "mesh", "mesh_2"));
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
