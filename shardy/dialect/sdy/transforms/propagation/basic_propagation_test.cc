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

#include "shardy/dialect/sdy/transforms/propagation/basic_propagation.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/common/propagation_options.h"
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

using BasicPropagationTest = ShardyTestBase;

TEST_F(BasicPropagationTest, ConcurrentNestedModulesNoActionHandlerRace) {
  constexpr const char* kModuleStr = R"mlir(
    module {
      module @m0 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
      module @m1 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
      module @m2 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
      module @m3 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
      module @m4 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
      module @m5 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
      module @m6 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
      module @m7 {
        sdy.mesh @mesh = <["a"=2, "b"=2]>
        func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                        %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
          %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
          %1 = stablehlo.add %0, %arg1 : tensor<8x8xf32>
          %2 = stablehlo.add %1, %arg1 : tensor<8x8xf32>
          return %2 : tensor<8x8xf32>
        }
      }
    }
  )mlir";

  ASSERT_TRUE(context.isMultithreadingEnabled());
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kModuleStr, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  OpPassManager& nestedPm = pm.nest<ModuleOp>();
  nestedPm.addPass(createApplyShardingConstraintsPass());
  nestedPm.addPass(createBasicPropagationPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  EXPECT_FALSE(context.hasActionHandler());
}

TEST_F(BasicPropagationTest,
       UnregistersActionHandlerWhenDebugShardingOriginsEnabled) {
  constexpr const char* kModuleStr = R"mlir(
    module {
      sdy.mesh @mesh = <["a"=2, "b"=2]>
      func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                      %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
        return %0 : tensor<8x8xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kModuleStr, &context);
  ASSERT_TRUE(module);

  PropagationOptions options;
  options.debugShardingOrigins = true;
  options.debugPropagationEdgeSharding = false;

  PassManager pm(&context);
  pm.addPass(createBasicPropagationPass(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));
  EXPECT_FALSE(context.hasActionHandler());
}

TEST_F(BasicPropagationTest,
       UnregistersActionHandlerWhenDebugPropagationEdgeShardingEnabled) {
  constexpr const char* kModuleStr = R"mlir(
    module {
      sdy.mesh @mesh = <["a"=2, "b"=2]>
      func.func @main(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                      %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
        return %0 : tensor<8x8xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kModuleStr, &context);
  ASSERT_TRUE(module);

  PropagationOptions options;
  options.debugShardingOrigins = false;
  options.debugPropagationEdgeSharding = true;

  PassManager pm(&context);
  pm.addPass(createBasicPropagationPass(options));
  ASSERT_TRUE(succeeded(pm.run(*module)));
  EXPECT_FALSE(context.hasActionHandler());
}

TEST_F(BasicPropagationTest,
       DoesNotReenqueueParentOpsOnArgOrTerminatorShardingUpdate) {
  constexpr const char* kModuleStr = R"mlir(
    module {
      sdy.mesh @mesh = <["a"=2, "b"=2]>
      func.func @main(
          %arg0: tensor<4x6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
          %arg1: tensor<4x6xf32>,
          %arg2: tensor<6x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
          %arg3: tensor<6x8xf32>,
          %arg4: tensor<4x10xf32>,
          %arg5: tensor<4x10xf32>,
          %arg6: tensor<4x12xf32>,
          %arg7: tensor<4x12xf32>)
          -> (tensor<4x6xf32>, tensor<6x8xf32>, tensor<4x10xf32>, tensor<4x12xf32>) {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4x6xf32>
        %1 = stablehlo.add %arg2, %arg3 : tensor<6x8xf32>
        %2:2 = sdy.manual_computation(%arg4, %arg5, %arg6, %arg7)
            in_shardings=[
              <@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {?}]>,
              <@mesh, [{"a"}, {"b"}]>, <@mesh, [{"a"}, {?}]>]
            out_shardings=[
              <@mesh, [{"a"}, {?}]>, <@mesh, [{"a"}, {?}]>]
            manual_axes={"a"} (
                %in0: tensor<2x10xf32>, %in1: tensor<2x10xf32>,
                %in2: tensor<2x12xf32>, %in3: tensor<2x12xf32>) {
          %m0 = stablehlo.add %in0, %in1 : tensor<2x10xf32>
          %m1 = stablehlo.add %in2, %in3 : tensor<2x12xf32>
          sdy.return %m0, %m1 : tensor<2x10xf32>, tensor<2x12xf32>
        } : (tensor<4x10xf32>, tensor<4x10xf32>, tensor<4x12xf32>, tensor<4x12xf32>)
            -> (tensor<4x10xf32>, tensor<4x12xf32>)
        return %0, %1, %2#0, %2#1
            : tensor<4x6xf32>, tensor<6x8xf32>, tensor<4x10xf32>, tensor<4x12xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kModuleStr, &context);
  ASSERT_TRUE(module);

  int funcOpPatternAttempts = 0;
  int manualOpPatternAttempts = 0;
  context.registerActionHandler([&](llvm::function_ref<void()> transform,
                                    const tracing::Action& action) {
    if (action.getTag() == ApplyPatternAction::tag &&
        !action.getContextIRUnits().empty()) {
      if (auto* op =
              llvm::dyn_cast<Operation*>(action.getContextIRUnits().front())) {
        if (llvm::isa<func::FuncOp>(op)) {
          ++funcOpPatternAttempts;
        } else if (llvm::isa<ManualComputationOp>(op)) {
          ++manualOpPatternAttempts;
        }
      }
    }
    transform();
  });

  PassManager pm(&context);
  pm.addPass(createBasicPropagationPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  context.registerActionHandler(nullptr);

  // 1 attempt per greedy iteration (1 in NDEBUG, 2 in debug builds).
  // Re-enqueuing parent ops via modifyOpInPlace would trigger O(K) attempts
  // and O(K * N) DeadnessCache region walks.
  EXPECT_GE(funcOpPatternAttempts, 1);
  EXPECT_LE(funcOpPatternAttempts, 2);
  EXPECT_GE(manualOpPatternAttempts, 1);
  EXPECT_LE(manualOpPatternAttempts, 2);

  func::FuncOp funcOp = *module->getOps<func::FuncOp>().begin();
  for (int argIdx : {1, 3}) {
    TensorShardingAttr sharding = getSharding(funcOp.getArgument(argIdx));
    ASSERT_NE(sharding, nullptr);
    EXPECT_EQ(strippedAttrString(sharding, /*stripMnemonic=*/true),
              "<@mesh, [{\"a\", ?}, {\"b\", ?}]>");
  }
  for (int resIdx : {0, 1}) {
    TensorShardingAttr sharding = getFuncResultSharding(funcOp, resIdx);
    ASSERT_NE(sharding, nullptr);
    EXPECT_EQ(strippedAttrString(sharding, /*stripMnemonic=*/true),
              "<@mesh, [{\"a\", ?}, {\"b\", ?}]>");
  }

  ManualComputationOp manualOp = *funcOp.getOps<ManualComputationOp>().begin();
  for (int inIdx : {1, 3}) {
    TensorShardingAttr sharding = manualOp.getInSharding(inIdx);
    ASSERT_NE(sharding, nullptr);
    EXPECT_EQ(strippedAttrString(sharding, /*stripMnemonic=*/true),
              "<@mesh, [{\"a\"}, {\"b\", ?}]>");
  }
  for (int outIdx : {0, 1}) {
    TensorShardingAttr innerSharding =
        getSharding(getBodyTerminatorOperand(manualOp, outIdx));
    ASSERT_NE(innerSharding, nullptr);
    EXPECT_EQ(strippedAttrString(innerSharding, /*stripMnemonic=*/true),
              "<@mesh, [{?}, {\"b\", ?}]>");
  }
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
