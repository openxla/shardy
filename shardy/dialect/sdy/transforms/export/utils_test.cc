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

#include "shardy/dialect/sdy/transforms/export/utils.h"

#include <cstdint>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/testing_utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

class ExportTest : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  AxisRefAttr createAxis(StringRef name) {
    return AxisRefAttr::get(&context, name);
  }

  AxisRefAttr createSubAxis(StringRef name, int64_t preSize, int64_t size) {
    return AxisRefAttr::get(&context, name, preSize, size);
  }

  MLIRContext context;
};

TEST_F(ExportTest, getGreatestCommonPrefix) {
  auto checkGreatestCommonPrefix = [](ArrayRef<AxisRefAttr> a,
                                      ArrayRef<AxisRefAttr> b,
                                      ArrayRef<AxisRefAttr> c) {
    EXPECT_THAT(getGreatestCommonPrefix(a, b), ElementsAreArray(c));
    EXPECT_THAT(getGreatestCommonPrefix(b, a), ElementsAreArray(c));
  };
  checkGreatestCommonPrefix(
      {createAxis("a"), createAxis("b"), createAxis("d"), createAxis("e")},
      {createAxis("a"), createAxis("b"), createAxis("c")},
      {createAxis("a"), createAxis("b")});
  checkGreatestCommonPrefix(
      {createAxis("a")}, {createAxis("a"), createAxis("b")}, {createAxis("a")});
  checkGreatestCommonPrefix(
      {createAxis("a"), createSubAxis("b", 2, 4), createAxis("d"),
       createAxis("e")},
      {createAxis("a"), createSubAxis("b", 2, 8), createAxis("d")},
      {createAxis("a"), createSubAxis("b", 2, 4)});
  checkGreatestCommonPrefix(
      {createSubAxis("a", 2, 6), createSubAxis("b", 2, 4), createAxis("d"),
       createAxis("e")},
      {createSubAxis("a", 2, 6), createSubAxis("b", 2, 8), createAxis("d")},
      {createSubAxis("a", 2, 6), createSubAxis("b", 2, 4)});
}

TEST_F(ExportTest, getGreatestCommonPrefix_AxesOrderMatters) {
  auto samePrefix = [](ArrayRef<AxisRefAttr> a) {
    EXPECT_THAT(getGreatestCommonPrefix(a, a), ElementsAreArray(a));
  };
  samePrefix({createAxis("a"), createAxis("d")});
  samePrefix({createAxis("d"), createAxis("a")});
}

TEST_F(ExportTest, getGreatestCommonPrefix_Empty) {
  auto emptyPrefix = [](ArrayRef<AxisRefAttr> a) {
    EXPECT_THAT(getGreatestCommonPrefix(a, {}), IsEmpty());
    EXPECT_THAT(getGreatestCommonPrefix({}, a), IsEmpty());
  };
  emptyPrefix({createAxis("a"), createAxis("c")});
  emptyPrefix({});
}

TEST_F(ExportTest, GetGreatestCommonPrefixAxes_DotGeneralSimple) {
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
      testing_utils::getShardingProjection<stablehlo::DotGeneralOp>(
          module.get());

  EXPECT_THAT(getGreatestCommonPrefixAxes(projection),
              ElementsAre(IsEmpty(), ElementsAre(AxisRefIs("d")),
                          ElementsAre(AxisRefIs("a"))));
}

TEST_F(ExportTest,
       GetGreatestCommonPrefixAxes_DotConflictOnNonContractingDimension) {
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
      testing_utils::getShardingProjection<stablehlo::DotOp>(module.get());

  EXPECT_THAT(getGreatestCommonPrefixAxes(projection),
              ElementsAre(ElementsAre(AxisRefIs("a")), IsEmpty(),
                          ElementsAre(AxisRefIs("b"))));
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
