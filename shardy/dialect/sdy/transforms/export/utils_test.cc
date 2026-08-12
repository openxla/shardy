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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"
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

 private:
  OwningOpRef<ModuleOp> moduleOp;
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

}  // namespace
}  // namespace sdy
}  // namespace mlir
