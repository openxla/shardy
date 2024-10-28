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

#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/testing_utils.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

inline constexpr StringRef kMeshName = "mesh";

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

MeshAttr getMeshAttr(ModuleOp module) {
  return cast<MeshOp>(module.lookupSymbol(kMeshName)).getMesh();
}

template <class OpTy>
OpTy getFirstOp(ModuleOp module) {
  auto mainFn = cast<func::FuncOp>(module.lookupSymbol("main"));
  auto ops = mainFn.getBody().front().getOps<OpTy>();
  assert(!ops.empty());
  return *ops.begin();
}

void verifyShardingAttrsMatch(TensorShardingAttr resultSharding,
                              TensorShardingAttr expectedSharding) {
  EXPECT_EQ(resultSharding, expectedSharding)
      << "result: " << llvm::to_string(resultSharding)
      << ", expected: " << llvm::to_string(expectedSharding);
}

void verifyReconstructedShardings(
    ValueRange tensors, ArrayRef<TensorFactorShardings> tensorFactorShardings,
    ArrayRef<TensorMappingAttr> tensorMappings, ArrayRef<int64_t> factorSizes,
    StringRef meshName, MeshAttr mesh) {
  for (auto [tensor, factorShardings, tensorMapping] :
       llvm::zip(tensors, tensorFactorShardings, tensorMappings)) {
    TensorShardingAttr reconstructedSharding =
        factorShardings.createTensorShardingAttr(
            mesh.getContext(), tensorMapping, factorSizes, kMeshName, mesh);
    verifyShardingAttrsMatch(reconstructedSharding,
                             getOrCreateSharding(tensor, kMeshName));
  }
}

// Builds a `ShardingProjection` for the first OpTy in the main function.
//
// In addition, verifies that reconstructing the `TensorShardingAttr` for each
// tensor (using `TensorFactorShardings::createTensorShardingAttr`) from the
// created projection matches the original sharding.
template <class OpTy>
ShardingProjection getShardingProjection(ModuleOp module) {
  OpTy op = getFirstOp<OpTy>(module);
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  assert(shardingRule);
  MeshAttr mesh = getMeshAttr(module);
  ShardingProjection projection =
      ShardingProjection::build(op, shardingRule, mesh);
  verifyReconstructedShardings(op->getOperands(), projection.getOperands(),
                               shardingRule.getOperandMappings(),
                               shardingRule.getFactorSizes(), kMeshName, mesh);
  verifyReconstructedShardings(op->getResults(), projection.getResults(),
                               shardingRule.getResultMappings(),
                               shardingRule.getFactorSizes(), kMeshName, mesh);
  return projection;
}

template <class OpTy>
ShardingProjection getShardingProjection(
    ModuleOp module, AxesPerFactorRef axesPerFactor) {
  OpTy op = getFirstOp<OpTy>(module);
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  assert(shardingRule);
  ShardingProjection projection =
      ShardingProjection::build(axesPerFactor, shardingRule);
  return projection;
}

//===----------------------------------------------------------------------===//
// Tests for ShardingProjection::build
//
// TensorFactorShardings::createTensorShardingAttr is also tested indirectly
// by calling it using the created `ShardingProjection` and verifying that the
// reconstructed `TensorShardingAttr` for each tensor matches the original one.
//===----------------------------------------------------------------------===//

class ShardingProjectionBuildTest : public PropagationTestBase {};

TEST_F(ShardingProjectionBuildTest, DotGeneralSimple) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=2, "c"=2, "d"=2, "e"=2]>

    func.func @main(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"a", ?}]>},
                    %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "c"}, {"d", ?}], replicated={"b"}>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"d", "e"}], replicated={"a", "c"}>]>
      } : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::DotGeneralOp>(module.get());

  EXPECT_THAT(projection.getOperand(0),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                                       /*isMinorMost*/ true,
                                       ElementsAre(AxisRefIs("b"))),
                      FactorShardingIs(/*index*/ 2, /*isClosed*/ false,
                                       /*isMinorMost*/ true,
                                       ElementsAre(AxisRefIs("a")))),
                  /*replicatedAxes*/ IsEmpty()));
  EXPECT_THAT(projection.getOperand(1),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(
                          /*index*/ 1, /*isClosed*/ false, /*isMinorMost*/ true,
                          ElementsAre(AxisRefIs("d"))),
                      FactorShardingIs(
                          /*index*/ 2, /*isClosed*/ true, /*isMinorMost*/ true,
                          ElementsAre(AxisRefIs("a"), AxisRefIs("c")))),
                  /*replicatedAxes*/ ElementsAre(AxisRefIs("b"))));
  EXPECT_THAT(
      projection.getResult(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                               /*isMinorMost*/ true, IsEmpty()),
              FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("d"), AxisRefIs("e")))),
          /*replicatedAxes*/ ElementsAre(AxisRefIs("a"), AxisRefIs("c"))));
}

TEST_F(ShardingProjectionBuildTest, ReshapeSplitDim) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=2, "c"=3]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}], replicated={"c"}>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                               /*isMinorMost*/ false,
                               ElementsAre(SubAxisRefIs("a", 1, 2))),
              FactorShardingIs(
                  /*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                  ElementsAre(SubAxisRefIs("a", 2, 2), AxisRefIs("b")))),
          /*replicatedAxes*/ ElementsAre(AxisRefIs("c"))));
  EXPECT_THAT(projection.getResult(0),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                                       /*isMinorMost*/ true, IsEmpty()),
                      FactorShardingIs(/*index*/ 1, /*isClosed*/ false,
                                       /*isMinorMost*/ true, IsEmpty())),
                  /*replicatedAxes*/ IsEmpty()));
}

TEST_F(ShardingProjectionBuildTest, ReshapeSplitDimAxisAlreadySplit) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=16, "b"=8, "c"=4]>

    func.func @main(%arg0: tensor<16x8xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{"a":(1)2, "b"}, {"a":(2)8}], replicated={"c":(2)2}>})
        -> tensor<16x2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<16x8xf32>) -> tensor<16x2x4xf32>
      return %0 : tensor<16x2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(projection.getOperand(0),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(
                          /*index*/ 0, /*isClosed*/ true, /*isMinorMost*/ true,
                          ElementsAre(SubAxisRefIs("a", 1, 2), AxisRefIs("b"))),
                      FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                                       /*isMinorMost*/ false,
                                       ElementsAre(SubAxisRefIs("a", 2, 2))),
                      FactorShardingIs(/*index*/ 2, /*isClosed*/ true,
                                       /*isMinorMost*/ true,
                                       ElementsAre(SubAxisRefIs("a", 4, 4)))),
                  /*replicatedAxes*/ ElementsAre(SubAxisRefIs("c", 2, 2))));
}

TEST_F(ShardingProjectionBuildTest, ReshapeMergeDim) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2, "b"=2]>

    func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}]>})
        -> tensor<16xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
      return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                           /*isMinorMost*/ true, ElementsAre(AxisRefIs("a"))),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           ElementsAre(AxisRefIs("b")))));
  EXPECT_THAT(
      projection.getResult(0).factorIndexToSharding,
      UnorderedElementsAre(FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                                            /*isMinorMost*/ false, IsEmpty()),
                           FactorShardingIs(/*index*/ 1, /*isClosed*/ false,
                                            /*isMinorMost*/ true, IsEmpty())));
}

TEST_F(ShardingProjectionBuildTest, ReshapeWithSizeOneDims) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2]>
    func.func @main(%arg0: tensor<4x2x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}]>})
        -> tensor<8xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<4x2x1xf32>) -> tensor<8xf32>
      return %0 : tensor<8xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                           /*isMinorMost*/ true, ElementsAre(AxisRefIs("a"))),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           IsEmpty()),
          FactorShardingIs(/*index*/ 2, /*isClosed*/ true, /*isMinorMost*/ true,
                           IsEmpty())));

  EXPECT_THAT(
      projection.getResult(0).factorIndexToSharding,
      UnorderedElementsAre(FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                                            /*isMinorMost*/ false, IsEmpty()),
                           FactorShardingIs(/*index*/ 1, /*isClosed*/ false,
                                            /*isMinorMost*/ true, IsEmpty())));
}

TEST_F(ShardingProjectionBuildTest, AddSingleFactorNonDivisible) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=3]>

    func.func @main(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>},
                    %arg1: tensor<2x4xf32>)
        -> tensor<2x4xf32> {
      %0 = stablehlo.add %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::AddOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ true, /*isMinorMost*/ true,
                           IsEmpty()),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           ElementsAre(AxisRefIs("a")))));
}

TEST_F(ShardingProjectionBuildTest, SingleFactorOverflows) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2, "b"=4]>

    func.func @main(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a", "b"}]>},
                    %arg1: tensor<2x4xf32>)
        -> tensor<2x4xf32> {
      %0 = stablehlo.add %arg0, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::AddOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ true, /*isMinorMost*/ true,
                           IsEmpty()),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           ElementsAre(AxisRefIs("a"), AxisRefIs("b")))));
}

TEST_F(ShardingProjectionBuildTest, FactorWithSmallerSizeThanDimOverflows) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2, "b"=4, "c"=2, "d"=4, "e"=2]>

    func.func @main(%arg0: tensor<32x4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"c", ?}, {"b", "d":(2)2, "e"}]>})
        -> tensor<32x1x2xf32> {
      %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:6] : (tensor<32x4x16xf32>) -> tensor<32x1x2xf32>
      return %0 : tensor<32x1x2xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::SliceOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                           /*isMinorMost*/ true, ElementsAre(AxisRefIs("a"))),
          FactorShardingIs(
              /*index*/ 1, /*isClosed*/ false, /*isMinorMost*/ true,
              /*axisRefs*/ ElementsAre(AxisRefIs("c"))),
          FactorShardingIs(
              /*index*/ 2, /*isClosed*/ true, /*isMinorMost*/ true,
              /*axisRefs*/
              ElementsAre(AxisRefIs("b"), SubAxisRefIs("d", 2, 2),
                          AxisRefIs("e")))));
}

TEST_F(ShardingProjectionBuildTest, ReshapeMinorMostFactorNonDivisible) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=6]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                           /*isMinorMost*/ false,
                           ElementsAre(SubAxisRefIs("a", 1, 2))),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           ElementsAre(SubAxisRefIs("a", 2, 3)))));
}

TEST_F(ShardingProjectionBuildTest, ReshapeMinorMostFactorOverflows) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=16]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                           /*isMinorMost*/ false,
                           ElementsAre(SubAxisRefIs("a", 1, 2))),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           ElementsAre(SubAxisRefIs("a", 2, 8)))));
}

TEST_F(ShardingProjectionBuildTest,
       ReshapeMinorMostFactorOverflowsSizeOneAxes) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=16, "b"=1, "c"=1]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", "c"}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                           /*isMinorMost*/ false,
                           ElementsAre(SubAxisRefIs("a", 1, 2))),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           ElementsAre(SubAxisRefIs("a", 2, 8), AxisRefIs("b"),
                                       AxisRefIs("c")))));
}

TEST_F(ShardingProjectionBuildTest, ReshapeNonMinorMostFactorNonDivisible) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=3]>

    func.func @main(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>})
        -> tensor<4x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
      return %0 : tensor<4x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(FactorShardingWithOverflowIs(
                               /*index*/ 0, /*isClosed*/ false,
                               /*isMinorMost*/ false, /*axisRefs*/ IsEmpty(),
                               /*overflowAxes*/ ElementsAre(AxisRefIs("a"))),
                           FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                                            /*isMinorMost*/ true, IsEmpty())));
}

TEST_F(ShardingProjectionBuildTest,
       ReshapeNonMinorMostFactorNonDivisibleSubAxis) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=6]>

    func.func @main(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
        -> tensor<4x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
      return %0 : tensor<4x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(projection.getOperand(0).factorIndexToSharding,
              UnorderedElementsAre(
                  FactorShardingWithOverflowIs(
                      /*index*/ 0, /*isClosed*/ true, /*isMinorMost*/ false,
                      /*axisRefs*/ ElementsAre(SubAxisRefIs("a", 1, 2)),
                      /*overflowAxes*/ ElementsAre(SubAxisRefIs("a", 2, 3))),
                  FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                                   /*isMinorMost*/ true, IsEmpty())));
}

TEST_F(ShardingProjectionBuildTest,
       ReshapeNonMinorMostFactorNonDivisibleMultipleAxes) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2, "b"=3]>

    func.func @main(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}]>})
        -> tensor<4x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
      return %0 : tensor<4x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(projection.getOperand(0).factorIndexToSharding,
              UnorderedElementsAre(
                  FactorShardingWithOverflowIs(
                      /*index*/ 0, /*isClosed*/ true, /*isMinorMost*/ false,
                      /*axisRefs*/ ElementsAre(AxisRefIs("a")),
                      /*overflowAxes*/ ElementsAre(AxisRefIs("b"))),
                  FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                                   /*isMinorMost*/ true, IsEmpty())));

  EXPECT_THAT(projection.getResult(0).factorIndexToSharding,
              UnorderedElementsAre(
                  FactorShardingIs(
                      /*index*/ 0, /*isClosed*/ false, /*isMinorMost*/ true,
                      /*axisRefs*/ IsEmpty()),
                  FactorShardingIs(/*index*/ 1, /*isClosed*/ false,
                                   /*isMinorMost*/ true, IsEmpty())));
}

TEST_F(ShardingProjectionBuildTest, ReshapeMinorMostFactorSizeOneAxes) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=1, "b"=2, "c"=1]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", "c"}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::ReshapeOp>(module.get());

  EXPECT_THAT(
      projection.getOperand(0).factorIndexToSharding,
      UnorderedElementsAre(
          FactorShardingIs(
              /*index*/ 0, /*isClosed*/ true, /*isMinorMost*/ false,
              ElementsAre(AxisRefIs("a"), AxisRefIs("b"), AxisRefIs("c"))),
          FactorShardingIs(/*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                           IsEmpty())));
}

TEST_F(ShardingProjectionBuildTest, DotGeneralSimpleFromFactorShardings) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=2, "c"=2, "d"=2]>

    func.func @main(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
                    %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>
      } : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  ShardingProjection projection =
      getShardingProjection<stablehlo::DotGeneralOp>(
          module.get(), {{AxisRefAttr::get(module->getContext(), "a")},
                         {AxisRefAttr::get(module->getContext(), "b")},
                         {AxisRefAttr::get(module->getContext(), "c"),
                          AxisRefAttr::get(module->getContext(), "d")}});

  EXPECT_THAT(
      projection.getOperand(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("a"))),
              FactorShardingIs(/*index*/ 2, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("c"), AxisRefIs("d")))),
          /*replicatedAxes*/ IsEmpty()));
  EXPECT_THAT(
      projection.getOperand(1),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 2, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("c"), AxisRefIs("d"))),
              FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("b")))),
          /*replicatedAxes*/ IsEmpty()));
  EXPECT_THAT(projection.getResult(0),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                                       /*isMinorMost*/ true,
                                       ElementsAre(AxisRefIs("a"))),
                      FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                                       /*isMinorMost*/ true,
                                       ElementsAre(AxisRefIs("b")))),
                  /*replicatedAxes*/ IsEmpty()));
}

TEST_F(ShardingProjectionBuildTest, ReshapeFromFactorShardings) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

    func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);

  ShardingProjection projection = getShardingProjection<stablehlo::ReshapeOp>(
      module.get(), {{AxisRefAttr::get(module->getContext(), "a")},
                     {AxisRefAttr::get(module->getContext(), "b"),
                      AxisRefAttr::get(module->getContext(), "c")}});

  EXPECT_THAT(
      projection.getOperand(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                               /*isMinorMost*/ false,
                               ElementsAre(AxisRefIs("a"))),
              FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("b"), AxisRefIs("c")))),
          /*replicatedAxes*/ IsEmpty()));

  EXPECT_THAT(
      projection.getResult(0),
      TensorFactorShardingsIs(
          /*factorIndexToSharding*/ UnorderedElementsAre(
              FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("a"))),
              FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                               /*isMinorMost*/ true,
                               ElementsAre(AxisRefIs("b"), AxisRefIs("c")))),
          /*replicatedAxes*/ IsEmpty()));
}

TEST_F(ShardingProjectionBuildTest,
       BuildFromGreatestCommonPrefixAxes_DotGeneralSimple) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=2, "c"=2, "d"=2, "e"=2]>
    func.func @main(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"a"}]>},
                    %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "c"}, {"d"}], replicated={"b"}>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"d", "e"}], replicated={"a", "c"}>]>
      } : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection initialProjection =
      getShardingProjection<stablehlo::DotGeneralOp>(module.get());

  ASSERT_THAT(initialProjection.getGreatestCommonPrefixAxes(/*numFactors=*/3),
              ElementsAre(IsEmpty(), ElementsAre(AxisRefIs("d")),
                          ElementsAre(AxisRefIs("a"))));

  ShardingProjection projection =
      getShardingProjection<stablehlo::DotGeneralOp>(
          module.get(),
          initialProjection.getGreatestCommonPrefixAxes(/*numFactors=*/3));

  EXPECT_THAT(projection.getOperand(0),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                                       /*isMinorMost*/ true, IsEmpty()),
                      FactorShardingIs(/*index*/ 2, /*isClosed*/ true,
                                       /*isMinorMost*/ true,
                                       ElementsAre(AxisRefIs("a")))),
                  /*replicatedAxes*/ IsEmpty()));
  EXPECT_THAT(projection.getOperand(1),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(
                          /*index*/ 1, /*isClosed*/ true, /*isMinorMost*/ true,
                          ElementsAre(AxisRefIs("d"))),
                      FactorShardingIs(
                          /*index*/ 2, /*isClosed*/ true, /*isMinorMost*/ true,
                          ElementsAre(AxisRefIs("a")))),
                  /*replicatedAxes*/ IsEmpty()));
  EXPECT_THAT(projection.getResult(0),
              TensorFactorShardingsIs(
                  /*factorIndexToSharding*/ UnorderedElementsAre(
                      FactorShardingIs(/*index*/ 0, /*isClosed*/ true,
                                       /*isMinorMost*/ true, IsEmpty()),
                      FactorShardingIs(/*index*/ 1, /*isClosed*/ true,
                                       /*isMinorMost*/ true,
                                       ElementsAre(AxisRefIs("d")))),
                  /*replicatedAxes*/ IsEmpty()));
}

//===----------------------------------------------------------------------===//
// Tests for ShardingProjection::updateSharding
//===----------------------------------------------------------------------===//

class ShardingProjectionUpdateShardingTest : public PropagationTestBase {};

TEST_F(ShardingProjectionUpdateShardingTest, DotGeneralSimple) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=4, "c"=4, "d"=4, "e"=4, "f"=4]>

    func.func @main(%arg0: tensor<256x512xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {"c", ?}]>},
                    %arg1: tensor<512x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"e"}, {"d", ?}], replicated={"f":(2)2}>})
        -> tensor<256x128xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b":(1)2, ?}, {"d", "f", ?}]>]>
      } : (tensor<256x512xf32>, tensor<512x128xf32>) -> tensor<256x128xf32>
      return %0 : tensor<256x128xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::DotGeneralOp>(module.get());

  // Set the sharding of factor 0 (LHS non-contracting dim) to ["a", "b"].
  // - The LHS is mapped to factor 0. The old sharding is the same as the new
  //   one, so it will be skipped.
  // - The RHS is not mapped to factor 0, so it will be skipped.
  // - The result is mapped to factor 0. The old sharding axes ["a", "b":(1)2]
  //   are smaller than the new one, so the sharding axes are updated.
  UpdateTensorShardings ifUpdated = projection.updateSharding(
      /*factorIndex=*/0, {createAxis("a"), createAxis("b")});
  EXPECT_THAT(toSetBitsVector(ifUpdated.updateOperands), IsEmpty());
  EXPECT_THAT(toSetBitsVector(ifUpdated.updateResults), ElementsAre(0));

  // Set the sharding of factor 1 (RHS non-contracting dim) to ["d", "f":(1)2].
  // - The LHS is not mapped to factor 1, so it will be skipped.
  // - The RHS is mapped to factor 1, and can be expaneded to new axes.
  // - The result is mapped to factor 1. The existing axes ["d", "f"] is larger
  //   than the new one, so it will be skipped.
  ifUpdated = projection.updateSharding(
      /*factorIndex=*/1, {createAxis("d"), createSubAxis("f", 1, 2)});
  EXPECT_THAT(toSetBitsVector(ifUpdated.updateOperands), ElementsAre(1));
  EXPECT_THAT(toSetBitsVector(ifUpdated.updateResults), IsEmpty());

  // We set the sharding of factor 2 (contracting dim) to [] (an empty vector of
  // axes). Skip all the tensors.
  ifUpdated = projection.updateSharding(/*factorIndex=*/2, {});
  EXPECT_THAT(toSetBitsVector(ifUpdated.updateOperands), IsEmpty());
  EXPECT_THAT(toSetBitsVector(ifUpdated.updateResults), IsEmpty());

  // Check the new factorIndexToSharding. `updateSharding` should not modify
  // other members (isClosed, isMinorMost, overflowAxes).
  EXPECT_THAT(projection.getOperand(0).factorIndexToSharding,
              UnorderedElementsAre(
                  FactorShardingIs(
                      /*index*/ 0, /*isClosed*/ true, /*isMinorMost*/ true,
                      ElementsAre(AxisRefIs("a"), AxisRefIs("b"))),
                  FactorShardingIs(
                      /*index*/ 2, /*isClosed*/ false, /*isMinorMost*/ true,
                      ElementsAre(AxisRefIs("c")))));

  EXPECT_THAT(projection.getOperand(1).factorIndexToSharding,
              UnorderedElementsAre(
                  FactorShardingIs(
                      /*index*/ 1, /*isClosed*/ false, /*isMinorMost*/ true,
                      ElementsAre(AxisRefIs("d"), SubAxisRefIs("f", 1, 2))),
                  FactorShardingIs(
                      /*index*/ 2, /*isClosed*/ true, /*isMinorMost*/ true,
                      ElementsAre(AxisRefIs("e")))));

  EXPECT_THAT(projection.getResult(0).factorIndexToSharding,
              UnorderedElementsAre(
                  FactorShardingIs(/*index*/ 0, /*isClosed*/ false,
                                   /*isMinorMost*/ true,
                                   ElementsAre(AxisRefIs("a"), AxisRefIs("b"))),
                  FactorShardingIs(
                      /*index*/ 1, /*isClosed*/ false, /*isMinorMost*/ true,
                      ElementsAre(AxisRefIs("d"), AxisRefIs("f")))));
}

//===----------------------------------------------------------------------===//
// Tests for IsAxisListPrefixOfTest
//===----------------------------------------------------------------------===//

class IsAxisListPrefixOfTest : public PropagationTestBase {};

TEST_F(IsAxisListPrefixOfTest, IsAxisListPrefixOfTest) {
  auto sameAxes = [](ArrayRef<AxisRefAttr> axes) {
    EXPECT_EQ(isAxisListPrefixOf(axes, axes), PrefixStatus::EQUAL);
  };
  sameAxes({});
  sameAxes({createAxis("a")});
  sameAxes({createSubAxis("a", 2, 4)});
  sameAxes({createAxis("a"), createSubAxis("b", 1, 2)});

  auto strictPrefix = [](ArrayRef<AxisRefAttr> prefix,
                         ArrayRef<AxisRefAttr> superset) {
    EXPECT_EQ(isAxisListPrefixOf(prefix, superset),
              PrefixStatus::STRICT_PREFIX);
    EXPECT_EQ(isAxisListPrefixOf(superset, prefix), PrefixStatus::NOT_A_PREFIX);
  };
  strictPrefix({}, {createAxis("a")});
  strictPrefix({createAxis("a")}, {createAxis("a"), createAxis("b")});
  strictPrefix({createSubAxis("a", 1, 4)}, {createAxis("a"), createAxis("b")});
  strictPrefix({createAxis("a"), createSubAxis("b", 1, 2)},
               {createAxis("a"), createAxis("b")});
  strictPrefix({createAxis("a"), createSubAxis("b", 1, 2)},
               {createAxis("a"), createSubAxis("b", 1, 8)});

  auto otherReplationship = [](ArrayRef<AxisRefAttr> first,
                               ArrayRef<AxisRefAttr> second) {
    EXPECT_EQ(isAxisListPrefixOf(first, second), PrefixStatus::NOT_A_PREFIX);
    EXPECT_EQ(isAxisListPrefixOf(second, first), PrefixStatus::NOT_A_PREFIX);
  };
  otherReplationship({createAxis("a")}, {createAxis("b")});
  otherReplationship({createAxis("a")}, {createSubAxis("b", 1, 2)});
  otherReplationship({createAxis("a"), createAxis("b")},
                     {createAxis("a"), createAxis("c")});
  otherReplationship({createAxis("a"), createAxis("b")},
                     {createAxis("b"), createAxis("a")});
  otherReplationship({createAxis("a"), createSubAxis("b", 2, 4)},
                     {createAxis("a"), createAxis("b")});
  otherReplationship({createAxis("a")},
                     {createSubAxis("a", 1, 4), createAxis("b")});
}

//===----------------------------------------------------------------------===//
// Tests for TensorFactorShardings::createTensorShardingAttr
//
// Since ShardingProjectionBuildTest also tests this method indirectly in each
// test case, here we only test the special cases that aren't tested above.
//===----------------------------------------------------------------------===//

class CreateTensorShardingAttrTest : public PropagationTestBase {
 protected:
  DimensionShardingAttr openDimSharding(ArrayRef<AxisRefAttr> axes) {
    return DimensionShardingAttr::get(&context, axes, /*isClosed=*/false);
  }

  DimensionShardingAttr closedDimSharding(ArrayRef<AxisRefAttr> axes) {
    return DimensionShardingAttr::get(&context, axes, /*isClosed=*/true);
  }

  TensorShardingAttr createTensorSharding(
      ArrayRef<DimensionShardingAttr> dimShardings,
      ArrayRef<AxisRefAttr> replicatedAxes = {}) {
    return TensorShardingAttr::get(&context, kMeshName, dimShardings,
                                   replicatedAxes);
  }
};

TEST_F(CreateTensorShardingAttrTest, ConsecutiveSubAxesMerged) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=8, "b"=2, "c"=2]>

    func.func @main(%arg0: tensor<4x4xf32>) -> tensor<16xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
      return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  TensorFactorShardings factorShardings{
      .factorIndexToSharding =
          {{0, {.axisRefs = {createAxis("b"), createSubAxis("a", 2, 2)}}},
           {1, {.axisRefs = {createSubAxis("a", 4, 2)}}}},
      .replicatedAxes = {createAxis("c")}};

  TensorShardingAttr shardingAttr = factorShardings.createTensorShardingAttr(
      &context, shardingRule.getResultMapping(0), shardingRule.getFactorSizes(),
      kMeshName, getMeshAttr(module.get()));

  verifyShardingAttrsMatch(
      shardingAttr, createTensorSharding(
                        /*dimShardings=*/{openDimSharding(
                            {createAxis("b"), createSubAxis("a", 2, 4)})},
                        /*replicatedAxes=*/{createAxis("c")}));
}

TEST_F(CreateTensorShardingAttrTest, OverflowSubAxisMerged) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=6, "b"=2]>

    func.func @main(%arg0: tensor<8xf32>) -> tensor<2x4xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  TensorFactorShardings factorShardings{
      .factorIndexToSharding = {{0,
                                 {.axisRefs = {createSubAxis("a", 1, 2)},
                                  .overflowAxes = {createSubAxis("a", 2, 3)}}},
                                {1, {.axisRefs = {}, .isClosed = true}}},
      .replicatedAxes = {createAxis("b")}};

  TensorShardingAttr shardingAttr = factorShardings.createTensorShardingAttr(
      &context, shardingRule.getOperandMapping(0),
      shardingRule.getFactorSizes(), kMeshName, getMeshAttr(module.get()));

  verifyShardingAttrsMatch(
      shardingAttr, createTensorSharding(
                        /*dimShardings=*/{openDimSharding({createAxis("a")})},
                        /*replicatedAxes=*/{createAxis("b")}));
}

TEST_F(CreateTensorShardingAttrTest, NonMinorMostFactorFullySharded) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2, "b"=2, "c"=4, "d"=2]>

    func.func @main(%arg0: tensor<4x4xf32>) -> tensor<16xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
      return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  TensorFactorShardings factorShardings{
      .factorIndexToSharding =
          {{0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
           {1, {.axisRefs = {createAxis("c")}}}},
      .replicatedAxes = {createAxis("d")}};

  TensorShardingAttr shardingAttr = factorShardings.createTensorShardingAttr(
      &context, shardingRule.getResultMapping(0), shardingRule.getFactorSizes(),
      kMeshName, getMeshAttr(module.get()));

  verifyShardingAttrsMatch(
      shardingAttr,
      createTensorSharding(
          /*dimShardings=*/{openDimSharding(
              {createAxis("a"), createAxis("b"), createAxis("c")})},
          /*replicatedAxes=*/{createAxis("d")}));
}

TEST_F(CreateTensorShardingAttrTest, NonMinorMostFactorPartiallySharded) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=2, "b"=2]>

    func.func @main(%arg0: tensor<4x4xf32>) -> tensor<16xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
      return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  TensorFactorShardings factorShardings{
      .factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}},
                                {1, {.axisRefs = {createAxis("b")}}}}};

  TensorShardingAttr shardingAttr = factorShardings.createTensorShardingAttr(
      &context, shardingRule.getResultMapping(0), shardingRule.getFactorSizes(),
      kMeshName, getMeshAttr(module.get()));

  // Axis "b" isn't added to the sharding because it would require strided view.
  verifyShardingAttrsMatch(
      shardingAttr, createTensorSharding(
                        /*dimShardings=*/{openDimSharding({createAxis("a")})}));
}

TEST_F(CreateTensorShardingAttrTest, MinorMostFactorNotDivisible) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=3, "b"=4]>

    func.func @main(%arg0: tensor<4x4xf32>) -> tensor<16xf32> {
      %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
      return %0 : tensor<16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  auto op = getFirstOp<stablehlo::ReshapeOp>(module.get());
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  TensorFactorShardings factorShardings{
      .factorIndexToSharding = {{0, {.axisRefs = {createAxis("b")}}},
                                {1, {.axisRefs = {createAxis("a")}}}}};

  TensorShardingAttr shardingAttr = factorShardings.createTensorShardingAttr(
      &context, shardingRule.getResultMapping(0), shardingRule.getFactorSizes(),
      kMeshName, getMeshAttr(module.get()));

  verifyShardingAttrsMatch(shardingAttr,
                           createTensorSharding(
                               /*dimShardings=*/{openDimSharding(
                                   {createAxis("b"), createAxis("a")})}));
}

//===----------------------------------------------------------------------===//
// Tests for ShardingProjection::getGreatestCommonPrefixAxes
//===----------------------------------------------------------------------===//
class ShardingProjectionGetGreatestCommonPrefixAxesTest
    : public PropagationTestBase {};

TEST_F(ShardingProjectionGetGreatestCommonPrefixAxesTest, DotGeneralSimple) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=2, "c"=2, "d"=2, "e"=2]>
    func.func @main(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"a"}]>},
                    %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "c"}, {"d"}], replicated={"b"}>})
        -> tensor<2x4xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"d", "e"}], replicated={"a", "c"}>]>
      } : (tensor<2x8xf32>, tensor<8x4xf32>) -> tensor<2x4xf32>
      return %0 : tensor<2x4xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::DotGeneralOp>(module.get());

  ASSERT_THAT(projection.getGreatestCommonPrefixAxes(/*numFactors=*/3),
              ElementsAre(IsEmpty(), ElementsAre(AxisRefIs("d")),
                          ElementsAre(AxisRefIs("a"))));
}

TEST_F(ShardingProjectionGetGreatestCommonPrefixAxesTest,
       DotConflictOnNonContractingDimension) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=4, "b"=2]>
    func.func @main(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
                    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {"a"}]>})
        -> tensor<8x16xf32> {
      %0 = stablehlo.dot %arg0, %arg1 {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>,
        sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=32, k=16}>
      } : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
      return %0 : tensor<8x16xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::DotOp>(module.get());

  EXPECT_THAT(projection.getGreatestCommonPrefixAxes(/*numFactors=*/3),
              ElementsAre(ElementsAre(AxisRefIs("a")), IsEmpty(),
                          ElementsAre(AxisRefIs("b"))));
}

TEST_F(ShardingProjectionGetGreatestCommonPrefixAxesTest, SubAxesSharding) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=24, "b"=16, "c"=2, "d"=2]>
    func.func @main(%arg0: tensor<1536xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a":(2)6,"b":(2)4,"c","d"}]>})
        -> tensor<1536xf32> {
      %0 = stablehlo.abs %arg0 {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a":(2)6,"b":(2)8,"c"}]>]>,
        sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i]) {i=1536}>
      } : (tensor<1536xf32>) -> tensor<1536xf32>
      return %0 : tensor<1536xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::AbsOp>(module.get());

  EXPECT_THAT(projection.getGreatestCommonPrefixAxes(/*numFactors=*/1),
              ElementsAre(ElementsAre(SubAxisRefIs("a", 2, 6),
                                      SubAxisRefIs("b", 2, 4))));
}

TEST_F(ShardingProjectionGetGreatestCommonPrefixAxesTest,
       DifferentSubAxisPrefixes) {
  const std::string program = R"mlir(
    sdy.mesh @mesh = <["a"=24, "b"=16, "c"=2, "d"=2]>
    func.func @main(%arg0: tensor<1536xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b","a":(2)4,"c"}]>})
        -> tensor<1536xf32> {
      %0 = stablehlo.abs %arg0 {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b","a":(4)2,"c","d"}]>]>,
        sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i]) {i=1536}>
      } : (tensor<1536xf32>) -> tensor<1536xf32>
      return %0 : tensor<1536xf32>
    })mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  ASSERT_TRUE(module);
  ShardingProjection projection =
      getShardingProjection<stablehlo::AbsOp>(module.get());

  EXPECT_THAT(projection.getGreatestCommonPrefixAxes(/*numFactors=*/1),
              ElementsAre(ElementsAre(AxisRefIs("b"))));
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
