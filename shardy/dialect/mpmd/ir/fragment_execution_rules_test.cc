/* Copyright 2025 The MPMD Authors.

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

#include "shardy/dialect/mpmd/ir/fragment_execution_rules.h"

#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::ElementsAreArray;
using ::testing::Eq;

namespace mlir::mpmd {
namespace {

void ExpectFragmentInfoEq(FragmentInfo actual, FragmentInfo expected) {
  // GetFragmentInfo returns a FragmentInfo with sorted origins, so we sort the
  // origins as well.
  llvm::sort(expected.origins);
  EXPECT_THAT(actual.origins, ElementsAreArray(expected.origins));
  EXPECT_EQ(actual.stage_id, expected.stage_id);
  EXPECT_EQ(actual.call_counter, expected.call_counter);
  EXPECT_EQ(actual.mesh_id, expected.mesh_id);
  EXPECT_EQ(actual.is_weight_gradient, expected.is_weight_gradient);
  // Compare full struct in case any fields were missed above.
  EXPECT_EQ(actual, expected);
}

FragmentOrigin MakeFragmentOrigin(const std::string& computation_name,
                                  int transpose_count) {
  return {computation_name, transpose_count};
}

FragmentInfo MakeFragmentInfo(
    const std::vector<FragmentOrigin>& origins, int mesh_id,
    std::optional<int> stage_id = std::nullopt,
    std::optional<int> call_counter = std::nullopt,
    std::optional<bool> is_weight_gradient = std::nullopt) {
  return {origins, stage_id, call_counter, mesh_id, is_weight_gradient};
}

FragmentMergeRule MakeFragmentMergeRule(
    const std::vector<FragmentInfo>& sources, const FragmentInfo& target) {
  return {sources, target};
}

struct GetFragmentInfoTestParams {
  bool split_bwd_fragments;
  std::optional<bool> expected_is_weight_gradient;
};

class GetFragmentInfoTest
    : public ::testing::TestWithParam<GetFragmentInfoTestParams> {};

TEST_P(GetFragmentInfoTest, GetFragmentInfo) {
  const GetFragmentInfoTestParams& params = GetParam();
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(123), "f2"(123)]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      return %0 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  ASSERT_TRUE(module);
  auto main_func = GetMainFunction(*module);
  ASSERT_TRUE(main_func);
  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());
  FragmentInfo fragment_info =
      GetFragmentInfo(fragment_op, params.split_bwd_fragments);

  ExpectFragmentInfoEq(
      fragment_info,
      MakeFragmentInfo(
          {MakeFragmentOrigin("f1", 123), MakeFragmentOrigin("f2", 123)},
          /*mesh_id=*/0, /*stage_id=*/std::nullopt,
          /*call_counter=*/std::nullopt, params.expected_is_weight_gradient));
}

INSTANTIATE_TEST_SUITE_P(
    GetFragmentInfo, GetFragmentInfoTest,
    testing::Values(
        GetFragmentInfoTestParams{/*split_bwd_fragments=*/false,
                                  /*expected_is_weight_gradient=*/std::nullopt},
        GetFragmentInfoTestParams{/*split_bwd_fragments=*/true,
                                  /*expected_is_weight_gradient=*/false}),
    [](const testing::TestParamInfo<GetFragmentInfoTest::ParamType>& info) {
      return info.param.split_bwd_fragments ? "WithSplitBwdFragments"
                                            : "WithoutSplitBwdFragments";
    });

struct SetFragmentInfoTestParams {
  std::string test_name;
  FragmentInfo info;
};

class SetFragmentInfoTest
    : public ::testing::TestWithParam<SetFragmentInfoTestParams> {};

TEST_P(SetFragmentInfoTest, SetAndGetFragmentInfo) {
  const SetFragmentInfoTestParams& params = GetParam();
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"(123), "f2"(123)]> (%arg0)(%arg2: tensor<4x8xf32>) {
        mpmd.return %arg2 : tensor<4x8xf32>
      } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
      return %0 : !mesh_1_tensor_4_8_f32
    }
  )mlir";

  MLIRContext context;
  loadAllRequiredDialects(&context);
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(kProgram, &context);
  ASSERT_TRUE(module);
  auto main_func = GetMainFunction(*module);
  ASSERT_TRUE(main_func);
  ASSERT_TRUE(main_func->hasAttr("topology"));

  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());

  IRRewriter rewriter(&context);
  SetFragmentInfo(fragment_op, params.info, rewriter);
  FragmentInfo fragment_info =
      GetFragmentInfo(fragment_op, /*split_bwd_fragments=*/true);

  ExpectFragmentInfoEq(fragment_info, params.info);
}

INSTANTIATE_TEST_SUITE_P(
    SetFragmentInfo, SetFragmentInfoTest,
    testing::Values(
        SetFragmentInfoTestParams{
            "WithStageAndCallCounter",
            MakeFragmentInfo({MakeFragmentOrigin("f3", 456)}, /*mesh_id=*/0,
                             /*stage_id=*/1, /*call_counter=*/2,
                             /*is_weight_gradient=*/false)},
        SetFragmentInfoTestParams{
            "WithWeightGradient",
            MakeFragmentInfo({MakeFragmentOrigin("f4", 789)}, /*mesh_id=*/0,
                             /*stage_id=*/std::nullopt,
                             /*call_counter=*/std::nullopt,
                             /*is_weight_gradient=*/true)}),
    [](const testing::TestParamInfo<SetFragmentInfoTest::ParamType>& info) {
      return info.param.test_name;
    });

struct PrintFragmentInfoTestParams {
  std::string test_name;
  FragmentInfo info;
  std::string expected_output;
};

class PrintFragmentInfoTest
    : public ::testing::TestWithParam<PrintFragmentInfoTestParams> {};

TEST_P(PrintFragmentInfoTest, PrintFragmentInfo) {
  const auto& params = GetParam();
  std::string str;
  llvm::raw_string_ostream os(str);
  os << params.info;
  EXPECT_THAT(str, Eq(params.expected_output));
}

INSTANTIATE_TEST_SUITE_P(
    PrintFragmentInfo, PrintFragmentInfoTest,
    testing::Values(
        PrintFragmentInfoTestParams{
            "AllFields",
            MakeFragmentInfo({MakeFragmentOrigin("f1", 123),
                              MakeFragmentOrigin("f2", 456)},
                             /*mesh_id=*/1, /*stage_id=*/1, /*call_counter=*/2),
            "FragmentInfo(origins=[\"f1\"(123),\"f2\"(456)],stage=1,"
            "call_counter=2,mesh_id=1)"},
        PrintFragmentInfoTestParams{
            "WithWeightGradientTrue",
            MakeFragmentInfo({MakeFragmentOrigin("f1", 123)},
                             /*mesh_id=*/1, /*stage_id=*/1, /*call_counter=*/2,
                             /*is_weight_gradient=*/true),
            "FragmentInfo(origins=[\"f1\"(123)],stage=1,call_counter=2,mesh_id="
            "1,is_weight_gradient=true)"},
        PrintFragmentInfoTestParams{
            "WithWeightGradientFalse",
            MakeFragmentInfo({MakeFragmentOrigin("f1", 123)},
                             /*mesh_id=*/1, /*stage_id=*/1, /*call_counter=*/2,
                             /*is_weight_gradient=*/false),
            "FragmentInfo(origins=[\"f1\"(123)],stage=1,call_counter=2,mesh_id="
            "1,is_weight_gradient=false)"},
        PrintFragmentInfoTestParams{
            "OnlyRequiredFields",
            MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/1),
            "FragmentInfo(origins=[\"f1\"(123)],mesh_id=1)"}),
    [](const testing::TestParamInfo<PrintFragmentInfoTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(FragmentMergeRule, PrintFragmentMergeRule) {
  FragmentMergeRule rule = MakeFragmentMergeRule(
      {MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/1,
                        /*stage_id=*/1),
       MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_id=*/1,
                        /*stage_id=*/1)},
      MakeFragmentInfo(
          {MakeFragmentOrigin("f1", 123), MakeFragmentOrigin("f2", 456)},
          /*mesh_id=*/1, /*stage_id=*/1));
  std::string str;
  llvm::raw_string_ostream os(str);
  os << rule;
  EXPECT_THAT(str, Eq("FragmentMergeRule(sources=["
                      "FragmentInfo(origins=[\"f1\"(123)],stage=1,mesh_id=1),"
                      "FragmentInfo(origins=[\"f2\"(456)],stage=1,mesh_id=1)],"
                      "target=FragmentInfo(origins=["
                      "\"f1\"(123),\"f2\"(456)],stage=1,mesh_id=1))"));
}

TEST(FragmentInfoMapInfoTest, IsEqual) {
  FragmentInfo info1 =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/0);
  FragmentInfo info2 =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/0);
  FragmentInfo info3 =
      MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_id=*/1);
  FragmentInfo info4 =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/1);

  EXPECT_TRUE(FragmentInfoMapInfo::isEqual(info1, info2));
  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(info1, info3));
  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(info1, info4));
}

TEST(FragmentInfoMapInfoTest, GetHashValue) {
  FragmentInfo info1 =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/0);
  FragmentInfo info2 =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/0);
  FragmentInfo info3 =
      MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_id=*/1);

  EXPECT_EQ(FragmentInfoMapInfo::getHashValue(info1),
            FragmentInfoMapInfo::getHashValue(info2));
  // It's highly likely they are different, though not guaranteed.
  EXPECT_NE(FragmentInfoMapInfo::getHashValue(info1),
            FragmentInfoMapInfo::getHashValue(info3));
}

TEST(FragmentInfoMapInfoTest, SpecialKeys) {
  FragmentInfo emptyKey = FragmentInfoMapInfo::getEmptyKey();
  FragmentInfo tombstoneKey = FragmentInfoMapInfo::getTombstoneKey();
  FragmentInfo info1 =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/0);

  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(emptyKey, info1));
  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(tombstoneKey, info1));
  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(emptyKey, tombstoneKey));
}

TEST(FragmentInfoMapInfoTest, DenseMapIntegration) {
  llvm::DenseMap<FragmentInfo, int, FragmentInfoMapInfo> map;

  FragmentInfo info1 =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/0);
  FragmentInfo info2 =
      MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_id=*/1);

  map[info1] = 1;
  map[info2] = 2;

  EXPECT_EQ(map.size(), 2);
  EXPECT_EQ(map[info1], 1);
  EXPECT_EQ(map[info2], 2);

  FragmentInfo info1_copy =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_id=*/0);
  EXPECT_TRUE(map.count(info1_copy));
  EXPECT_EQ(map[info1_copy], 1);

  map.erase(info1);
  EXPECT_EQ(map.size(), 1);
  EXPECT_FALSE(map.count(info1));
}

}  // namespace
}  // namespace mlir::mpmd
