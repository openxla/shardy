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

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/register.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FieldsAre;

namespace mlir::mpmd {
namespace {

FragmentOrigin MakeFragmentOrigin(const std::string& computation_name,
                                  int transpose_count) {
  return {computation_name, transpose_count};
}

FragmentInfo MakeFragmentInfo(const std::vector<FragmentOrigin>& origins,
                              std::optional<int> stage_id = std::nullopt,
                              std::optional<int> call_counter = std::nullopt) {
  return {origins, stage_id, call_counter};
}

FragmentMergeRule MakeFragmentMergeRule(
    const std::vector<FragmentInfo>& sources, const FragmentInfo& target) {
  return {sources, target};
}

TEST(FragmentInfo, GetFragmentInfo) {
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
  FragmentInfo fragment_info = GetFragmentInfo(fragment_op);
  EXPECT_THAT(fragment_info.origins,
              ElementsAre(FieldsAre("f1", 123), FieldsAre("f2", 123)));
  EXPECT_THAT(fragment_info.stage_id, Eq(std::nullopt));
  EXPECT_THAT(fragment_info.call_counter, Eq(std::nullopt));
}

TEST(FragmentInfo, SetFragmentInfo) {
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
  SDY_CHECK(module);
  auto main_func = GetMainFunction(*module);
  SDY_CHECK(main_func);
  FragmentOp fragment_op = cast<FragmentOp>(*main_func.getOps().begin());

  IRRewriter rewriter(&context);
  SetFragmentInfo(fragment_op,
                  MakeFragmentInfo({MakeFragmentOrigin("f3", 456)},
                                   /*stage_id=*/1, /*call_counter=*/2),
                  rewriter);
  FragmentInfo fragment_info = GetFragmentInfo(fragment_op);
  EXPECT_THAT(fragment_info.origins, ElementsAre(FieldsAre("f3", 456)));
  EXPECT_THAT(fragment_info.stage_id, Eq(1));
  EXPECT_THAT(fragment_info.call_counter, Eq(2));

  SetFragmentInfo(fragment_op,
                  MakeFragmentInfo({MakeFragmentOrigin("f4", 789)}), rewriter);
  fragment_info = GetFragmentInfo(fragment_op);
  EXPECT_THAT(fragment_info.origins, ElementsAre(FieldsAre("f4", 789)));
  EXPECT_THAT(fragment_info.stage_id, Eq(std::nullopt));
  EXPECT_THAT(fragment_info.call_counter, Eq(std::nullopt));
}

TEST(FragmentInfo, PrintFragmentInfo) {
  FragmentInfo fragment_info = MakeFragmentInfo(
      {MakeFragmentOrigin("f1", 123), MakeFragmentOrigin("f2", 456)},
      /*stage_id=*/1, /*call_counter=*/2);
  std::string str;
  llvm::raw_string_ostream os(str);
  os << fragment_info;
  EXPECT_THAT(str, Eq("FragmentInfo(origins=[\"f1\"(123),\"f2\"(456)],stage=1,"
                      "call_counter=2)"));
}

TEST(FragmentMergeRule, PrintFragmentMergeRule) {
  FragmentMergeRule rule = MakeFragmentMergeRule(
      {MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*stage_id=*/1),
       MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*stage_id=*/1)},
      MakeFragmentInfo(
          {MakeFragmentOrigin("f1", 123), MakeFragmentOrigin("f2", 456)},
          /*stage_id=*/1));
  std::string str;
  llvm::raw_string_ostream os(str);
  os << rule;
  EXPECT_THAT(str, Eq("FragmentMergeRule(sources=["
                      "FragmentInfo(origins=[\"f1\"(123)],stage=1),"
                      "FragmentInfo(origins=[\"f2\"(456)],stage=1)],"
                      "target=FragmentInfo(origins=["
                      "\"f1\"(123),\"f2\"(456)],stage=1))"));
}

}  // namespace
}  // namespace mlir::mpmd
