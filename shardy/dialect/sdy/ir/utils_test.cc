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

#include <algorithm>
#include <optional>
#include <random>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class UtilsTest : public ShardyTestBase {
 protected:
  void SetUp() override {
    ShardyTestBase::SetUp();
    moduleOp = mlir::parseSourceString<ModuleOp>(
        "module {\n"
        "  sdy.mesh @mesh_empty = <[]>\n"
        "  sdy.mesh @mesh_empty_another = <[]>\n"
        "  sdy.mesh @mesh_ab_23 = <[\"a\"=2, \"b\"=3]>\n"
        "  sdy.mesh @mesh_xy_23 = <[\"x\"=2, \"y\"=3]>\n"
        "  sdy.mesh @mesh_xy_23_non_iota = <[\"x\"=2, \"y\"=3], "
        "device_ids=[5, 4, 3, 2, 1, 0]>\n"
        "  sdy.mesh @mesh_xy_23_non_iota_another = <[\"x\"=2, \"y\"=3], "
        "device_ids=[1, 2, 3, 4, 5, 0]>\n"
        "  sdy.mesh @mesh_maximal = #sdy.mesh<[], device_ids=[0]>\n"
        "  sdy.mesh @mesh_maximal_copy = #sdy.mesh<[], device_ids=[0]>\n"
        "  sdy.mesh @mesh_maximal_another = #sdy.mesh<[], device_ids=[1]>\n"
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

TEST_F(UtilsTest, GetCommonMeshName_AllIdenticalEmptyMeshesDifferentNames) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_empty_another")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_empty_another");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_empty_another")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_empty_another");
}

TEST_F(UtilsTest, GetCommonMeshName_MixOfEmptyAndNonEmptyMeshEmptyFirst) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_xy_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_xy_23");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_xy_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_xy_23");
}

TEST_F(UtilsTest, GetCommonMeshName_MixOfEmptyAndNonEmptyMeshNonEmptyFirst) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_empty")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_xy_23");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_empty")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_xy_23");
}

TEST_F(UtilsTest, GetCommonMeshName_MixOfEmptyAndMaximalMesh) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_maximal");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_maximal");
}

TEST_F(UtilsTest,
       GetCommonMeshName_MixOfEmptyAndMaximalAndNonEmptyNonMaximalMesh) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_xy_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_xy_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            std::nullopt);
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

TEST_F(UtilsTest,
       GetCommonMeshName_AllSameIgnoringDeviceIdsIncludingEmptyMesh) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_xy_23_non_iota")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_xy_23_non_iota")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_xy_23");
}

TEST_F(
    UtilsTest,
    GetCommonMeshName_AllSameIgnoringDeviceIdsIncludingEmptyMeshAndEmptyMeshIsFirst) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_xy_23")},
                              {createTensorSharding("mesh_xy_23_non_iota")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty"),
                               createTensorSharding("mesh_xy_23")},
                              {createTensorSharding("mesh_xy_23_non_iota")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_xy_23");
}

TEST_F(UtilsTest, GetCommonMeshName_AllSameIgnoringDeviceIdsNonIotaIsFirst) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23_non_iota")},
                              {createTensorSharding("mesh_xy_23")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23_non_iota")},
                              {createTensorSharding("mesh_xy_23")},
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

TEST_F(UtilsTest, GetCommonMeshName_AllSameMaximalMeshes) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_maximal"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_maximal");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_maximal"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_maximal");
}

TEST_F(UtilsTest, GetCommonMeshName_AllIdenticalMaximalMeshes) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_maximal"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_maximal_copy")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_maximal");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_maximal"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_maximal_copy")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_maximal");
}

TEST_F(UtilsTest, GetCommonMeshName_DifferentMaximalMeshes) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_maximal"),
                               createTensorSharding("mesh_maximal_copy")},
                              {createTensorSharding("mesh_maximal_another")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_maximal"),
                               createTensorSharding("mesh_maximal_copy")},
                              {createTensorSharding("mesh_maximal_another")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_maximal");
}

TEST_F(UtilsTest,
       GetCommonMeshName_MixOfMaximalAndNonMaximalMeshesMajorityIsMaximal) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_maximal")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            std::nullopt);
}

TEST_F(UtilsTest,
       GetCommonMeshName_MixOfMaximalAndNonMaximalMeshesMajorityIsNonMaximal) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_xy_23")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            std::nullopt);
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_xy_23"),
                               createTensorSharding("mesh_xy_23")},
                              {createTensorSharding("mesh_maximal")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            std::nullopt);
}

// TODO(enver): Should it return nullopt instead when one mesh does not exist.
TEST_F(UtilsTest,
       GetCommonMeshName_MixOfEmptyMeshAndInexistingMeshEmptyMeshFirst) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_does_not_exist")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_does_not_exist");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_empty")},
                              {createTensorSharding("mesh_does_not_exist")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_does_not_exist");
}

// TODO(enver): Should it return nullopt instead when one mesh does not exist.
TEST_F(UtilsTest,
       GetCommonMeshName_MixOfEmptyMeshAndInexistingMeshInexistingMeshFirst) {
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_does_not_exist")},
                              {createTensorSharding("mesh_empty")},
                              getSymbolTable(), /*ignoreDeviceIds=*/false),
            "mesh_empty");
  EXPECT_EQ(getCommonMeshName({createTensorSharding("mesh_does_not_exist")},
                              {createTensorSharding("mesh_empty")},
                              getSymbolTable(), /*ignoreDeviceIds=*/true),
            "mesh_empty");
}

TEST_F(UtilsTest, GetAxisSetDiff) {
  MeshAttr mesh = createMesh({{"a", 8}, {"b", 4}, {"c", 2}, {"d", 4}});

  EXPECT_THAT(getAxisSetDiff({}, {createAxis("c"), createAxis("d")}, mesh),
              IsEmpty());

  EXPECT_THAT(getAxisSetDiff({createAxis("b"), createAxis("a")}, {}, mesh),
              ElementsAre(AxisRefIs("b"), AxisRefIs("a")));

  EXPECT_THAT(getAxisSetDiff({createAxis("a"), createSubAxis("b", 1, 2)},
                             {createSubAxis("b", 1, 2), createAxis("a")}, mesh),
              IsEmpty());

  EXPECT_THAT(getAxisSetDiff({createAxis("b"), createAxis("a")},
                             {createAxis("c"), createSubAxis("d", 1, 2)}, mesh),
              ElementsAre(AxisRefIs("b"), AxisRefIs("a")));

  EXPECT_THAT(getAxisSetDiff({createAxis("b"), createAxis("a")},
                             {createAxis("b"), createAxis("d")}, mesh),
              ElementsAre(AxisRefIs("a")));

  EXPECT_THAT(getAxisSetDiff({createAxis("b"), createSubAxis("a", 1, 4)},
                             {createAxis("a"), createSubAxis("b", 2, 2)}, mesh),
              ElementsAre(SubAxisRefIs("b", 1, 2)));

  EXPECT_THAT(getAxisSetDiff({createSubAxis("a", 2, 4)},
                             {createSubAxis("a", 1, 4)}, mesh),
              ElementsAre(SubAxisRefIs("a", 4, 2)));

  EXPECT_THAT(getAxisSetDiff(
                  {createSubAxis("b", 1, 2), createAxis("a")},
                  {createSubAxis("a", 2, 2), createSubAxis("b", 2, 2)}, mesh),
              ElementsAre(SubAxisRefIs("b", 1, 2), SubAxisRefIs("a", 1, 2),
                          SubAxisRefIs("a", 4, 2)));

  EXPECT_THAT(getAxisSetDiff(
                  {createAxis("a")},
                  {createSubAxis("a", 1, 2), createSubAxis("a", 4, 2)}, mesh),
              ElementsAre(SubAxisRefIs("a", 2, 2)));
}

TEST_F(UtilsTest, SortAndMergeAxes) {
  MeshAttr mesh = createMesh({{"a", 8}, {"b", 8}, {"c", 8}, {"d", 8}});

  SmallVector<AxisRefAttr> axes = {
      createSubAxis("a", 1, 2), createSubAxis("a", 4, 2),
      createSubAxis("b", 1, 2), createSubAxis("b", 2, 2),
      createSubAxis("b", 4, 2), createAxis("c"),
      createSubAxis("d", 2, 2), createSubAxis("d", 4, 2)};

  // Shuffle the vector
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(axes.begin(), axes.end(), g);

  sortAndMergeAxes(axes, mesh);

  EXPECT_THAT(axes, ElementsAre(SubAxisRefIs("a", 1, 2),
                                SubAxisRefIs("a", 4, 2), AxisRefIs("b"),
                                AxisRefIs("c"), SubAxisRefIs("d", 2, 4)));
}

}  // namespace

}  // namespace sdy
}  // namespace mlir
