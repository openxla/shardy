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

#include "shardy/dialect/sdy/transforms/export/utils.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

class ExportUtilsTest : public ShardyTestBase {
 protected:
  void SetUp() override {
    ShardyTestBase::SetUp();
    moduleOp = mlir::parseSourceString<ModuleOp>(
        "module {\n"
        "  sdy.mesh @mesh = <[\"x\"=2, \"y\"=4]>\n"
        "}",
        &context);
    f32Type = Builder(&context).getF32Type();
  }

  const SymbolTable& getSymbolTable() {
    return symbolTableCollection.getSymbolTable(moduleOp.get());
  }

  Type f32Type;
  OwningOpRef<ModuleOp> moduleOp;

 private:
  mlir::SymbolTableCollection symbolTableCollection;
};

TEST_F(ExportUtilsTest, GetShardIndexInterleavedSubAxes) {
  // mesh = <["a"=4, "b"=2]>
  // axes = ["a":2(2), "b", "a":1(2)]
  MeshAttr mesh =
      MeshAttr::get(&context, {MeshAxisAttr::get(&context, "a", 4),
                               MeshAxisAttr::get(&context, "b", 2)});

  AxisRefAttr aLow =
      AxisRefAttr::get(&context, "a", /*subAxisPreSize=*/2, /*subAxisSize=*/2);
  AxisRefAttr b = AxisRefAttr::get(&context, "b");
  AxisRefAttr aHigh =
      AxisRefAttr::get(&context, "a", /*subAxisPreSize=*/1, /*subAxisSize=*/2);

  SmallVector<AxisRefAttr> axes = {aLow, b, aHigh};

  // Device IDs in {0..7}
  // Mesh layout: id = a * 2 + b, where a = 2 * a_high + a_low
  //   b = id % 2
  //   a_low = (id / 2) % 2
  //   a_high = id / 4
  // Shard index for axes {a_low, b, a_high}:
  //   shardIndex = 4 * a_low + 2 * b + a_high
  EXPECT_EQ(getShardIndex(0, mesh, axes), 0);
  EXPECT_EQ(getShardIndex(1, mesh, axes), 2);
  EXPECT_EQ(getShardIndex(2, mesh, axes), 4);
  EXPECT_EQ(getShardIndex(3, mesh, axes), 6);
  EXPECT_EQ(getShardIndex(4, mesh, axes), 1);
  EXPECT_EQ(getShardIndex(5, mesh, axes), 3);
  EXPECT_EQ(getShardIndex(6, mesh, axes), 5);
  EXPECT_EQ(getShardIndex(7, mesh, axes), 7);
}

TEST_F(ExportUtilsTest, GetShardIndexWithExplicitDeviceIds) {
  // mesh = <["a"=4, "b"=2], device_ids=[7, 6, 5, 4, 3, 2, 1, 0]>
  // Mapping from physical deviceId to logical device index:
  //   logicalDeviceId = 7 - deviceId
  // For axis "a" (size 4), with suffixSize = 2 (size of axis "b"):
  //   shardIndex = (logicalDeviceId / 2) % 4
  SmallVector<int64_t> deviceIds = {7, 6, 5, 4, 3, 2, 1, 0};
  MeshAttr mesh = MeshAttr::get(&context,
                                {MeshAxisAttr::get(&context, "a", 4),
                                 MeshAxisAttr::get(&context, "b", 2)},
                                deviceIds);

  AxisRefAttr a = AxisRefAttr::get(&context, "a");
  SmallVector<AxisRefAttr> axes = {a};

  EXPECT_EQ(getShardIndex(7, mesh, axes), 0);
  EXPECT_EQ(getShardIndex(6, mesh, axes), 0);
  EXPECT_EQ(getShardIndex(5, mesh, axes), 1);
  EXPECT_EQ(getShardIndex(4, mesh, axes), 1);
  EXPECT_EQ(getShardIndex(3, mesh, axes), 2);
  EXPECT_EQ(getShardIndex(2, mesh, axes), 2);
  EXPECT_EQ(getShardIndex(1, mesh, axes), 3);
  EXPECT_EQ(getShardIndex(0, mesh, axes), 3);
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeAlreadyDivisible) {
  auto origType = RankedTensorType::get({8, 16}, f32Type);
  TensorShardingAttr sharding = TensorShardingAttr::get(
      &context, "mesh",
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "x")},
                                  /*isClosed=*/true),
       DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "y")},
                                  /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  Type paddedType =
      getDivisiblePaddedType(origType, sharding, getSymbolTable());
  EXPECT_EQ(paddedType, origType);
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeIndivisible) {
  auto origType = RankedTensorType::get({7, 15}, f32Type);
  TensorShardingAttr sharding = TensorShardingAttr::get(
      &context, "mesh",
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "x")},
                                  /*isClosed=*/true),
       DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "y")},
                                  /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  Type paddedType =
      getDivisiblePaddedType(origType, sharding, getSymbolTable());
  EXPECT_EQ(paddedType, RankedTensorType::get({8, 16}, f32Type));
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeWithAllowedAxes) {
  auto origType = RankedTensorType::get({7, 15}, f32Type);
  TensorShardingAttr sharding = TensorShardingAttr::get(
      &context, "mesh",
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "x")},
                                  /*isClosed=*/true),
       DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "y")},
                                  /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  llvm::DenseSet<StringRef> allowedAxes = {"x"};
  Type paddedType = getDivisiblePaddedType(origType, sharding, getSymbolTable(),
                                           &allowedAxes);
  // Dim 0 (axis "x", size 2) is padded from 7 to 8.
  // Dim 1 (axis "y") is not in allowedAxes, so it remains 15.
  EXPECT_EQ(paddedType, RankedTensorType::get({8, 15}, f32Type));
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeReplicated) {
  auto origType = RankedTensorType::get({7, 15}, f32Type);
  TensorShardingAttr sharding = TensorShardingAttr::get(
      &context, "mesh",
      {DimensionShardingAttr::get(&context, /*axes=*/{}, /*isClosed=*/true),
       DimensionShardingAttr::get(&context, /*axes=*/{}, /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  Type paddedType =
      getDivisiblePaddedType(origType, sharding, getSymbolTable());
  EXPECT_EQ(paddedType, origType);
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeDynamicShape) {
  auto origType = RankedTensorType::get({ShapedType::kDynamic, 15}, f32Type);
  TensorShardingAttr sharding = TensorShardingAttr::get(
      &context, "mesh",
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "x")},
                                  /*isClosed=*/true),
       DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "y")},
                                  /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  Type paddedType =
      getDivisiblePaddedType(origType, sharding, getSymbolTable());
  EXPECT_EQ(paddedType,
            RankedTensorType::get({ShapedType::kDynamic, 16}, f32Type));
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeUnknownMesh) {
  auto origType = RankedTensorType::get({7, 15}, f32Type);
  TensorShardingAttr sharding = TensorShardingAttr::get(
      &context, "unknown_mesh",
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "x")},
                                  /*isClosed=*/true),
       DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "y")},
                                  /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  Type paddedType =
      getDivisiblePaddedType(origType, sharding, getSymbolTable());
  EXPECT_EQ(paddedType, origType);
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeNonRankedType) {
  TensorShardingAttr sharding = TensorShardingAttr::get(
      &context, "mesh",
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "x")},
                                  /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  Type paddedType = getDivisiblePaddedType(f32Type, sharding, getSymbolTable());
  EXPECT_EQ(paddedType, f32Type);
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolSymbolRef) {
  FlatSymbolRefAttr sym = FlatSymbolRefAttr::get(&context, "mesh");
  SymbolTable symbolTable(moduleOp.get());
  EXPECT_EQ(getOrCreateMeshSymbol(moduleOp.get(), sym, symbolTable), sym);
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolExistingMeshAttr) {
  MeshAttr meshAttr =
      MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 2),
                               MeshAxisAttr::get(&context, "y", 4)});
  SymbolTable symbolTable(moduleOp.get());
  FlatSymbolRefAttr sym =
      getOrCreateMeshSymbol(moduleOp.get(), meshAttr, symbolTable);
  ASSERT_NE(sym, nullptr);
  EXPECT_EQ(sym.getValue(), "mesh");
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolNewMeshAttr) {
  MeshAttr newMeshAttr =
      MeshAttr::get(&context, {MeshAxisAttr::get(&context, "z", 8)});
  SymbolTable symbolTable(moduleOp.get());
  FlatSymbolRefAttr sym =
      getOrCreateMeshSymbol(moduleOp.get(), newMeshAttr, symbolTable);
  ASSERT_NE(sym, nullptr);
  auto meshOp = symbolTable.lookup<MeshOp>(sym.getValue());
  ASSERT_NE(meshOp, nullptr);
  EXPECT_EQ(meshOp.getMesh(), newMeshAttr);
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolNullOrInvalidAttr) {
  SymbolTable symbolTable(moduleOp.get());
  EXPECT_EQ(getOrCreateMeshSymbol(moduleOp.get(), nullptr, symbolTable),
            nullptr);
  EXPECT_EQ(getOrCreateMeshSymbol(moduleOp.get(), UnitAttr::get(&context),
                                  symbolTable),
            nullptr);
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolDirectModule) {
  MeshAttr meshAttr =
      MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 2),
                               MeshAxisAttr::get(&context, "y", 4)});
  SymbolTable symbolTable(moduleOp.get());
  FlatSymbolRefAttr sym = getOrCreateMeshSymbol(
      moduleOp.get()->getLoc(), moduleOp.get(), meshAttr, symbolTable);
  ASSERT_NE(sym, nullptr);
  EXPECT_EQ(sym.getValue(), "mesh");
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolNestedOp) {
  SymbolTable symbolTable(moduleOp.get());
  Operation* nestedOp = *moduleOp.get().getOps<MeshOp>().begin();
  FlatSymbolRefAttr sym = FlatSymbolRefAttr::get(&context, "mesh");
  EXPECT_EQ(getOrCreateMeshSymbol(nestedOp, sym, symbolTable), sym);
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolGlobalMesh) {
  // Create a global mesh (non-maximal, non-empty axes) and add it to the
  // module.
  MeshAttr globalMeshAttr =
      MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 4)});
  SymbolTable symbolTable(moduleOp.get());

  OpBuilder builder(moduleOp.get().getBodyRegion());
  MeshOp globalMeshOp = MeshOp::create(builder, moduleOp.get().getLoc(),
                                       "global_mesh", globalMeshAttr);
  symbolTable.insert(globalMeshOp);

  // Calling getOrCreateMeshSymbol with the matching inlined MeshAttr should
  // return the global_mesh symbol ref.
  FlatSymbolRefAttr sym =
      getOrCreateMeshSymbol(moduleOp.get(), globalMeshAttr, symbolTable);
  ASSERT_NE(sym, nullptr);
  EXPECT_EQ(sym.getValue(), "global_mesh");
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolMaximalMesh) {
  // Create a maximal mesh (e.g. single-device mesh with device_ids)
  MeshAttr maximalMeshAttr =
      MeshAttr::get(&context, /*axes=*/{}, /*deviceIds=*/{0});
  SymbolTable symbolTable(moduleOp.get());

  FlatSymbolRefAttr sym =
      getOrCreateMeshSymbol(moduleOp.get(), maximalMeshAttr, symbolTable);
  ASSERT_NE(sym, nullptr);

  auto meshOp = symbolTable.lookup<MeshOp>(sym.getValue());
  ASSERT_NE(meshOp, nullptr);
  EXPECT_EQ(meshOp.getMesh(), maximalMeshAttr);
  EXPECT_TRUE(meshOp.getMesh().isMaximal());
}

TEST_F(ExportUtilsTest, GetOrCreateMeshSymbolExistingMaximalMesh) {
  MeshAttr maximalMeshAttr =
      MeshAttr::get(&context, /*axes=*/{}, /*deviceIds=*/{0});
  SymbolTable symbolTable(moduleOp.get());

  OpBuilder builder(moduleOp.get().getBodyRegion());
  MeshOp maximalMeshOp = MeshOp::create(builder, moduleOp.get().getLoc(),
                                        "maximal_mesh", maximalMeshAttr);
  symbolTable.insert(maximalMeshOp);

  FlatSymbolRefAttr sym =
      getOrCreateMeshSymbol(moduleOp.get(), maximalMeshAttr, symbolTable);
  ASSERT_NE(sym, nullptr);
  EXPECT_EQ(sym.getValue(), "maximal_mesh");
}

TEST_F(ExportUtilsTest, BuildReshapeGroupInfos) {
  auto inType = RankedTensorType::get({3, 16, 7}, f32Type);
  auto outType = RankedTensorType::get({3, 4, 4, 7}, f32Type);

  SmallVector<ReshapeGroupInfo> groups =
      buildReshapeGroupInfos(inType, outType);
  ASSERT_EQ(groups.size(), 3);

  // Group 0: 3 -> 3 (passthrough)
  EXPECT_EQ(groups[0].inStartDim, 0);
  EXPECT_EQ(groups[0].inLastDim, 1);
  EXPECT_EQ(groups[0].outStartDim, 0);
  EXPECT_EQ(groups[0].outLastDim, 1);
  EXPECT_TRUE(groups[0].isPassthrough());

  // Group 1: 16 -> 4x4 (split)
  EXPECT_EQ(groups[1].inStartDim, 1);
  EXPECT_EQ(groups[1].inLastDim, 2);
  EXPECT_EQ(groups[1].outStartDim, 1);
  EXPECT_EQ(groups[1].outLastDim, 3);
  EXPECT_TRUE(groups[1].isSplit());
  EXPECT_FALSE(groups[1].isPassthrough());

  // Group 2: 7 -> 7 (passthrough)
  EXPECT_EQ(groups[2].inStartDim, 2);
  EXPECT_EQ(groups[2].inLastDim, 3);
  EXPECT_EQ(groups[2].outStartDim, 3);
  EXPECT_EQ(groups[2].outLastDim, 4);
  EXPECT_TRUE(groups[2].isPassthrough());
}

TEST_F(ExportUtilsTest, GetReductionType) {
  OwningOpRef<ModuleOp> module = mlir::parseSourceString<ModuleOp>(
      R"mlir(
        func.func @test_ops(%arg0: tensor<4xf32>, %arg1: tensor<f32>,
                            %arg2: tensor<4xi64>, %arg3: tensor<4xf32>) {
          %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
          %1 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.maximum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
          %2 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.minimum across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
          %3 = "stablehlo.reduce"(%arg0, %arg1) <{dimensions = array<i64: 0>}> ({
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %m = stablehlo.multiply %a, %b : tensor<f32>
            stablehlo.return %m : tensor<f32>
          }) : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
          %4 = "stablehlo.scatter"(%arg0, %arg2, %arg3) ({
          ^bb0(%a: tensor<f32>, %b: tensor<f32>):
            %max = stablehlo.maximum %a, %b : tensor<f32>
            stablehlo.return %max : tensor<f32>
          }) {
            scatter_dimension_numbers = #stablehlo.scatter<
              update_window_dims = [],
              inserted_window_dims = [0],
              input_batching_dims = [],
              scatter_indices_batching_dims = [],
              scatter_dims_to_operand_dims = [0],
              index_vector_dim = 1>
          } : (tensor<4xf32>, tensor<4xi64>, tensor<4xf32>) -> tensor<4xf32>
          return
        }
      )mlir",
      &context);
  ASSERT_TRUE(module);
  auto funcOp = cast<func::FuncOp>(module->lookupSymbol("test_ops"));
  Block& block = funcOp.getBody().front();
  auto it = block.begin();
  Operation* reduceAdd = &*it++;
  Operation* reduceMax = &*it++;
  Operation* reduceMin = &*it++;
  Operation* reduceMul = &*it++;
  Operation* scatterMax = &*it++;

  EXPECT_EQ(getReductionType(reduceAdd), ReductionOp::SUM);
  EXPECT_EQ(getReductionType(reduceMax), ReductionOp::MAX);
  EXPECT_EQ(getReductionType(reduceMin), ReductionOp::MIN);
  EXPECT_EQ(getReductionType(reduceMul), std::nullopt);
  EXPECT_EQ(getReductionType(scatterMax), ReductionOp::MAX);
}

TEST_F(ExportUtilsTest, GetReductionIdentityAttr) {
  OpBuilder builder(&context);
  Type i32Type = builder.getI32Type();
  Type u32Type = IntegerType::get(&context, 32, IntegerType::Unsigned);

  // SUM
  Attribute sumF32 =
      getReductionIdentityAttr(f32Type, ReductionOp::SUM, builder);
  EXPECT_EQ(cast<FloatAttr>(sumF32).getValueAsDouble(), 0.0);
  Attribute sumI32 =
      getReductionIdentityAttr(i32Type, ReductionOp::SUM, builder);
  EXPECT_EQ(cast<IntegerAttr>(sumI32).getInt(), 0);

  // MIN
  Attribute minF32 =
      getReductionIdentityAttr(f32Type, ReductionOp::MIN, builder);
  EXPECT_TRUE(cast<FloatAttr>(minF32).getValue().isInfinity());
  EXPECT_FALSE(cast<FloatAttr>(minF32).getValue().isNegative());
  Attribute minI32 =
      getReductionIdentityAttr(i32Type, ReductionOp::MIN, builder);
  EXPECT_EQ(cast<IntegerAttr>(minI32).getInt(),
            std::numeric_limits<int32_t>::max());
  Attribute minU32 =
      getReductionIdentityAttr(u32Type, ReductionOp::MIN, builder);
  EXPECT_EQ(cast<IntegerAttr>(minU32).getValue().getZExtValue(),
            std::numeric_limits<uint32_t>::max());

  // MAX
  Attribute maxF32 =
      getReductionIdentityAttr(f32Type, ReductionOp::MAX, builder);
  EXPECT_TRUE(cast<FloatAttr>(maxF32).getValue().isInfinity());
  EXPECT_TRUE(cast<FloatAttr>(maxF32).getValue().isNegative());
  Attribute maxI32 =
      getReductionIdentityAttr(i32Type, ReductionOp::MAX, builder);
  EXPECT_EQ(cast<IntegerAttr>(maxI32).getInt(),
            std::numeric_limits<int32_t>::min());
  Attribute maxU32 =
      getReductionIdentityAttr(u32Type, ReductionOp::MAX, builder);
  EXPECT_EQ(cast<IntegerAttr>(maxU32).getValue().getZExtValue(), 0);
}

TEST_F(ExportUtilsTest, GetDivisiblePaddedTypeDualSharding) {
  MeshAttr mesh =
      MeshAttr::get(&context, {MeshAxisAttr::get(&context, "a", 4),
                               MeshAxisAttr::get(&context, "b", 6)});

  auto origType = RankedTensorType::get({14, 15}, f32Type);
  TensorShardingAttr inSharding = TensorShardingAttr::get(
      &context, mesh,
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "a")},
                                  /*isClosed=*/true),
       DimensionShardingAttr::get(&context, /*axes=*/{}, /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});
  TensorShardingAttr outSharding = TensorShardingAttr::get(
      &context, mesh,
      {DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "b")},
                                  /*isClosed=*/true),
       DimensionShardingAttr::get(&context, {AxisRefAttr::get(&context, "a")},
                                  /*isClosed=*/true)},
      /*replicatedAxes=*/{}, /*unreducedAxes=*/{});

  // Dim 0: inShardSize = 4 ("a"), outShardSize = 6 ("b"). LCM(4, 6) = 12.
  // 14 padded to multiple of 12 -> 24.
  // Dim 1: inShardSize = 1, outShardSize = 4 ("a"). LCM(1, 4) = 4.
  // 15 padded to multiple of 4 -> 16.
  RankedTensorType paddedType =
      getDivisiblePaddedType(origType, inSharding, outSharding, mesh);
  EXPECT_EQ(paddedType, RankedTensorType::get({24, 16}, f32Type));
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
