/* Copyright 2025 The Shardy Authors.


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

#include "shardy/dialect/sdy/ir/utils.h"

#include <optional>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

namespace {

class UtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loadAllRequiredDialects(&context);
    moduleOp = mlir::parseSourceString<ModuleOp>(
        "module {\n"
        "  sdy.mesh @mesh_empty = <[]>\n"
        "  sdy.mesh @mesh_ab_23 = <[\"a\"=2, \"b\"=3]>\n"
        "  sdy.mesh @mesh_xy_23 = <[\"x\"=2, \"y\"=3]>\n"
        "  sdy.mesh @mesh_xy_23_non_iota = <[\"x\"=2, \"y\"=3], "
        "device_ids=[5, 4, 3, 2, 1, 0]>\n"
        "  sdy.mesh @mesh_xy_23_non_iota_another = <[\"x\"=2, \"y\"=3], "
        "device_ids=[1, 2, 3, 4, 5, 0]>\n"
        "  func.func @main(%arg0: tensor<24xf32>) -> tensor<24xf32> {\n"
        "    return %arg0 : tensor<24xf32>\n"
        "  }\n"
        "}",
        &context);
  }


  TensorShardingAttr createTensorSharding(const std::string& meshName) {
    return TensorShardingAttr::get(
        &context, meshName,
        {DimensionShardingAttr::get(&context, /*axes=*/{}, /*isClosed=*/true)},
        /*replicatedAxes=*/{}, /*unreducedAxes=*/{});
  }

  const SymbolTable& getSymbolTable() {
    return symbolTableCollection.getSymbolTable(moduleOp.get());
  }

 private:
  MLIRContext context;
  OwningOpRef<ModuleOp> moduleOp;
  mlir::SymbolTableCollection symbolTableCollection;
};

TEST_F(UtilsTest, GetCommonMeshName_AllSame) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_ab_23")},
                              {createTensorSharding("mesh_ab_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_ab_23");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_ab_23")},
                              {createTensorSharding("mesh_ab_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_ab_23");
}

TEST_F(UtilsTest, GetCommonMeshName_AllEmptyMeshes) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_empty")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_empty");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_empty")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_empty");
}

TEST_F(UtilsTest, GetCommonMeshName_AllEmptyShardings) {
  EXPECT_EQ(getCommonMeshName(TensorShardingAttr(), TensorShardingAttr(),
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName(TensorShardingAttr(), TensorShardingAttr(),
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            std::nullopt);
}

TEST_F(UtilsTest, GetCommonMeshName_AllSameIgnoringDeviceIds) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23")},
                              {createTensorSharding("mesh_xy_23_non_iota")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23")},
                              {createTensorSharding("mesh_xy_23_non_iota")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_xy_23");
}

TEST_F(UtilsTest, GetCommonMeshName_DifferentMeshes) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_ab_23")},
                              {createTensorSharding("mesh_xy_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_ab_23")},
                              {createTensorSharding("mesh_xy_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            std::nullopt);
}

TEST_F(UtilsTest, GetCommonMeshName_DoesNotCreateIota) {
  EXPECT_EQ(
      getCommonMeshName({createTensorSharding("mesh_xy_23_non_iota")},
                        {createTensorSharding("mesh_xy_23_non_iota_another")},
                        getSymbolTable(), /*ignoreDeviceIds=*/false),
      std::nullopt);
  EXPECT_EQ(
      getCommonMeshName({createTensorSharding("mesh_xy_23_non_iota")},
                        {createTensorSharding("mesh_xy_23_non_iota_another")},
                        getSymbolTable(), /*ignoreDeviceIds=*/true),
      "mesh_xy_23_non_iota");
}

TEST_F(UtilsTest, GetCommonMeshName_EmptyShardingIsIgnored) {
  EXPECT_EQ(getCommonMeshName(
                {createTensorSharding("mesh_xy_23"), TensorShardingAttr()},
                {createTensorSharding("mesh_xy_23")}, getSymbolTable(),
                /*ignoreDeviceIds=*/false),
            "mesh_xy_23");
  EXPECT_EQ(getCommonMeshName(
                {createTensorSharding("mesh_xy_23"), TensorShardingAttr()},
                {createTensorSharding("mesh_xy_23_non_iota")}, getSymbolTable(),
                /*ignoreDeviceIds=*/true),
            "mesh_xy_23");
}

TEST_F(UtilsTest, GetCommonMeshName_AllSameMultipleMeshes) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_ab_23"),
                               createTensorSharding("mesh_ab_23")},
                              {createTensorSharding("mesh_ab_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_ab_23");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_ab_23"),
                               createTensorSharding("mesh_ab_23")},
                              {createTensorSharding("mesh_ab_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_ab_23");
}

TEST_F(UtilsTest, GetCommonMeshName_AllSameIgnoringDeviceIdsMultipleMeshes) {
  EXPECT_EQ(
      getCommonMeshName({createTensorSharding("mesh_xy_23"),
                         createTensorSharding("mesh_xy_23_non_iota")},
                        {createTensorSharding("mesh_xy_23_non_iota_another")},
                        getSymbolTable(), /*ignoreDeviceIds=*/false),
      std::nullopt);
  EXPECT_EQ(
      getCommonMeshName({createTensorSharding("mesh_xy_23"),
                         createTensorSharding("mesh_xy_23_non_iota")},
                        {createTensorSharding("mesh_xy_23_non_iota_another")},
                        getSymbolTable(), /*ignoreDeviceIds=*/true),
      "mesh_xy_23");
}
}  // namespace

}  // namespace sdy
}  // namespace mlir
