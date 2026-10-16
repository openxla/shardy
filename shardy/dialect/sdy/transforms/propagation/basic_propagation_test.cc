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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
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

}  // namespace
}  // namespace sdy
}  // namespace mlir
