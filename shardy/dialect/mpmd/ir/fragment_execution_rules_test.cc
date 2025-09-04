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
#include "llvm/Support/CommandLine.h"
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
#include "shardy/dialect/mpmd/transforms/common/utils.h"
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
  EXPECT_EQ(actual.split_type, expected.split_type);
  // Compare full struct in case any fields were missed above.
  EXPECT_EQ(actual, expected);
}

FragmentOrigin MakeFragmentOrigin(const std::string& computation_name,
                                  int transpose_count) {
  return {computation_name, transpose_count};
}

FragmentInfo MakeFragmentInfo(
    const std::vector<FragmentOrigin>& origins, const std::string& mesh_name,
    std::optional<int> stage_id = std::nullopt,
    std::optional<int> call_counter = std::nullopt,
    std::optional<SplitFragmentType> split_type = std::nullopt) {
  return {origins, stage_id, call_counter, split_type, mesh_name};
}

FragmentMergeRule MakeFragmentMergeRule(
    const std::vector<FragmentInfo>& sources, const FragmentInfo& target) {
  return {sources, target};
}

FragmentScheduleRule MakeFragmentScheduleRule(
    const std::vector<FragmentInfo>& ordered_fragments) {
  return {ordered_fragments};
}

// LLVM's command line classes (OptionCategory, opt) store StringRef arguments
// directly without copying the underlying string data. When these objects are
// created with temporary string literals in test functions, the backing strings
// go out of scope after the test completes, leaving dangling pointers in the
// static GlobalParser->RegisteredOptionCategories.
//
// The functions below use static storage to ensure string literals have static
// storage duration, avoiding the need for manual cleanup. The parser helper
// functions encapsulate opt/parser creation and provide a clean interface for
// tests without exposing StringRef lifetime concerns.

llvm::cl::OptionCategory& getTestOptionCategory() {
  static llvm::cl::OptionCategory category("Test Options");
  return category;
}

bool parseFragmentMergeRule(llvm::StringRef rule_str, FragmentMergeRule& rule) {
  static llvm::cl::opt<FragmentMergeRule> rule_opt(
      "fragment-merge-rule",
      llvm::cl::desc("Fragment merge rule for testing parser functionality"),
      llvm::cl::cat(getTestOptionCategory()));
  static llvm::cl::parser<FragmentMergeRule> parser(rule_opt);

  return parser.parse(rule_opt, "test-rule", rule_str, rule);
}

bool parseFragmentScheduleRule(llvm::StringRef rule_str,
                               FragmentScheduleRule& rule) {
  static llvm::cl::opt<FragmentScheduleRule> rule_opt(
      "fragment-schedule-rule",
      llvm::cl::desc("Fragment schedule rule for testing parser functionality"),
      llvm::cl::cat(getTestOptionCategory()));
  static llvm::cl::parser<FragmentScheduleRule> parser(rule_opt);

  return parser.parse(rule_opt, "test-rule", rule_str, rule);
}

TEST(GetFragmentInfoTest, GetFragmentInfo) {
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

  ExpectFragmentInfoEq(
      fragment_info,
      MakeFragmentInfo(
          {MakeFragmentOrigin("f1", 123), MakeFragmentOrigin("f2", 123)},
          /*mesh_name=*/"m1",
          /*stage_id=*/std::nullopt,
          /*call_counter=*/std::nullopt, /*split_type=*/std::nullopt));
}

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
  FragmentInfo fragment_info = GetFragmentInfo(fragment_op);

  ExpectFragmentInfoEq(fragment_info, params.info);
}

INSTANTIATE_TEST_SUITE_P(
    SetFragmentInfo, SetFragmentInfoTest,
    testing::Values(
        SetFragmentInfoTestParams{
            "WithStageAndCallCounter",
            MakeFragmentInfo({MakeFragmentOrigin("f3", 456)},
                             /*mesh_name=*/"m1",
                             /*stage_id=*/1, /*call_counter=*/2,
                             /*split_type=*/std::nullopt)},
        SetFragmentInfoTestParams{
            "WithWeightGradient",
            MakeFragmentInfo(
                {MakeFragmentOrigin("f4", 789)}, /*mesh_name=*/"m1",
                /*stage_id=*/std::nullopt,
                /*call_counter=*/std::nullopt,
                /*split_type=*/SplitFragmentType::kDropTransferred)}),
    [](const testing::TestParamInfo<SetFragmentInfoTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(SetFragmentInfoTest, RemovesSplitDropTransferred) {
  const std::string kProgram = R"mlir(
    !mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
    func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
      -> (!mesh_1_tensor_4_8_f32) attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
      %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) {split_drop_transferred} (%arg2: tensor<4x8xf32>) {
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
  ASSERT_TRUE(fragment_op->hasAttr(kSplitDropTransferredAttrName));

  IRRewriter rewriter(&context);
  FragmentInfo info = MakeFragmentInfo({MakeFragmentOrigin("f1", 0)},
                                       /*mesh_name=*/"m1",
                                       /*stage_id=*/std::nullopt,
                                       /*call_counter=*/std::nullopt,
                                       /*split_type=*/std::nullopt);
  SetFragmentInfo(fragment_op, info, rewriter);

  EXPECT_FALSE(fragment_op->hasAttr(kSplitDropTransferredAttrName));
  FragmentInfo fragment_info = GetFragmentInfo(fragment_op);
  ExpectFragmentInfoEq(fragment_info, info);
}

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
            "NoSplitType",
            MakeFragmentInfo({MakeFragmentOrigin("f1", 123),
                              MakeFragmentOrigin("f2", 456)},
                             /*mesh_name=*/"m1", /*stage_id=*/1,
                             /*call_counter=*/2, /*split_type=*/std::nullopt),
            "FragmentInfo(origins=[\"f1\"(123),\"f2\"(456)],stage=1,call_"
            "counter=2,mesh_name=\"m1\")"},
        PrintFragmentInfoTestParams{
            "WithSplitTypeDropTransferred",
            MakeFragmentInfo(
                {MakeFragmentOrigin("f1", 123)}, /*mesh_name=*/"m1",
                /*stage_id=*/1, /*call_counter=*/2,
                /*split_type=*/SplitFragmentType::kDropTransferred),
            "FragmentInfo(origins=[\"f1\"(123)],stage=1,call_counter=2,"
            "split_type=kDropTransferred,mesh_name=\"m1\")"},
        PrintFragmentInfoTestParams{
            "WithSplitTypeKeepTransferred",
            MakeFragmentInfo(
                {MakeFragmentOrigin("f1", 123)}, /*mesh_name=*/"m1",
                /*stage_id=*/1, /*call_counter=*/2,
                /*split_type=*/SplitFragmentType::kKeepTransferred),
            "FragmentInfo(origins=[\"f1\"(123)],stage=1,call_counter=2,"
            "split_type=kKeepTransferred,mesh_name=\"m1\")"},
        PrintFragmentInfoTestParams{
            "OnlyRequiredFields",
            MakeFragmentInfo({MakeFragmentOrigin("f1", 123)},
                             /*mesh_name=*/"m1"),
            "FragmentInfo(origins=[\"f1\"(123)],mesh_name=\"m1\")"}),
    [](const testing::TestParamInfo<PrintFragmentInfoTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(FragmentMergeRule, PrintFragmentMergeRule) {
  FragmentMergeRule rule = MakeFragmentMergeRule(
      {MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_name=*/"m1",
                        /*stage_id=*/1),
       MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_name=*/"m1",
                        /*stage_id=*/1)},
      MakeFragmentInfo(
          {MakeFragmentOrigin("f1", 123), MakeFragmentOrigin("f2", 456)},
          /*mesh_name=*/"m1", /*stage_id=*/1, /*call_counter=*/std::nullopt,
          /*split_type=*/std::nullopt));
  std::string str;
  llvm::raw_string_ostream os(str);
  os << rule;
  EXPECT_THAT(
      str, Eq("FragmentMergeRule(sources=["
              "FragmentInfo(origins=[\"f1\"(123)],stage=1,mesh_name=\"m1\"),"
              "FragmentInfo(origins=[\"f2\"(456)],stage=1,mesh_name=\"m1\")],"
              "target=FragmentInfo(origins=["
              "\"f1\"(123),\"f2\"(456)],stage=1,mesh_name=\"m1\"))"));
}

TEST(FragmentMergeRuleParser, ParseValidRule) {
  FragmentMergeRule expected_rule = MakeFragmentMergeRule(
      {MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_name=*/"m1",
                        /*stage_id=*/1, /*call_counter=*/std::nullopt,
                        /*split_type=*/std::nullopt),
       MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_name=*/"m1",
                        /*stage_id=*/1, /*call_counter=*/std::nullopt,
                        /*split_type=*/SplitFragmentType::kDropTransferred)},
      MakeFragmentInfo(
          {MakeFragmentOrigin("f1", 123), MakeFragmentOrigin("f2", 456)},
          /*mesh_name=*/"m1", /*stage_id=*/1,
          /*call_counter=*/std::nullopt, /*split_type=*/std::nullopt));
  // We first construct the rule and print it to a string. Then we parse that
  // string to ensure that the printed form of a rule is directly compatible
  // with the format the parser expects.
  std::string rule_str;
  llvm::raw_string_ostream os(rule_str);
  os << expected_rule;

  FragmentMergeRule rule;
  bool result = parseFragmentMergeRule(rule_str, rule);

  EXPECT_FALSE(result);

  ASSERT_EQ(rule.sources.size(), 2);
  ExpectFragmentInfoEq(rule.sources[0], expected_rule.sources[0]);
  ExpectFragmentInfoEq(rule.sources[1], expected_rule.sources[1]);
  ExpectFragmentInfoEq(rule.target, expected_rule.target);
}

struct InvalidRuleTestParams {
  std::string test_name;
  std::string invalid_rule_str;
};

class FragmentMergeRuleParserInvalidSyntaxTest
    : public ::testing::TestWithParam<InvalidRuleTestParams> {};

TEST_P(FragmentMergeRuleParserInvalidSyntaxTest, ParseInvalidRule) {
  const auto& params = GetParam();
  FragmentMergeRule rule;
  bool result = parseFragmentMergeRule(params.invalid_rule_str, rule);

  EXPECT_TRUE(result);
}

INSTANTIATE_TEST_SUITE_P(
    FragmentMergeRuleParser, FragmentMergeRuleParserInvalidSyntaxTest,
    testing::Values(
        InvalidRuleTestParams{"MissingPrefix",
                              "sources=[FragmentInfo(origins=[\"f1\"(123)])]"},
        InvalidRuleTestParams{
            "MissingSources",
            "FragmentMergeRule(target=FragmentInfo(origins=[\"f1\"(123)]))"},
        InvalidRuleTestParams{"MissingTarget",
                              "FragmentMergeRule(sources=[FragmentInfo(origins="
                              "[\"f1\"(123)])])"}),
    [](const testing::TestParamInfo<
        FragmentMergeRuleParserInvalidSyntaxTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(FragmentScheduleRule, PrintFragmentScheduleRule) {
  FragmentScheduleRule rule = MakeFragmentScheduleRule(
      {MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_name=*/"m1",
                        /*stage_id=*/1),
       MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_name=*/"m1",
                        /*stage_id=*/2)});
  std::string str;
  llvm::raw_string_ostream os(str);
  os << rule;
  EXPECT_THAT(
      str,
      Eq("FragmentScheduleRule(ordered_fragments=["
         "FragmentInfo(origins=[\"f1\"(123)],stage=1,mesh_name=\"m1\")->"
         "FragmentInfo(origins=[\"f2\"(456)],stage=2,mesh_name=\"m1\")])"));
}

TEST(FragmentScheduleRuleParser, ParseValidRule) {
  FragmentScheduleRule expected_rule = MakeFragmentScheduleRule(
      {MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, /*mesh_name=*/"m1",
                        /*stage_id=*/1, /*call_counter=*/std::nullopt,
                        /*split_type=*/std::nullopt),
       MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, /*mesh_name=*/"m1",
                        /*stage_id=*/1, /*call_counter=*/std::nullopt,
                        /*split_type=*/SplitFragmentType::kDropTransferred)});
  // We first construct the rule and print it to a string. Then we parse that
  // string to ensure that the printed form of a rule is directly compatible
  // with the format the parser expects.
  std::string rule_str;
  llvm::raw_string_ostream os(rule_str);
  os << expected_rule;

  FragmentScheduleRule rule;
  bool result = parseFragmentScheduleRule(rule_str, rule);

  EXPECT_FALSE(result);

  ASSERT_EQ(rule.ordered_fragments.size(), 2);
  ExpectFragmentInfoEq(rule.ordered_fragments[0],
                       expected_rule.ordered_fragments[0]);
  ExpectFragmentInfoEq(rule.ordered_fragments[1],
                       expected_rule.ordered_fragments[1]);
}

class FragmentScheduleRuleParserInvalidSyntaxTest
    : public ::testing::TestWithParam<InvalidRuleTestParams> {};

TEST_P(FragmentScheduleRuleParserInvalidSyntaxTest, ParseInvalidRule) {
  const auto& params = GetParam();
  FragmentScheduleRule rule;
  bool result = parseFragmentScheduleRule(params.invalid_rule_str, rule);

  EXPECT_TRUE(result);
}

INSTANTIATE_TEST_SUITE_P(
    FragmentScheduleRuleParser, FragmentScheduleRuleParserInvalidSyntaxTest,
    testing::Values(
        InvalidRuleTestParams{"MissingPrefix",
                              "[FragmentInfo(origins=[\"f1\"(123)])->"
                              "FragmentInfo(origins=[\"f2\"(456)])])"},
        InvalidRuleTestParams{
            "MissingArrow",
            "FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["
            "\"f1\"(123)]) FragmentInfo(origins=[\"f2\"(456)])])"},
        InvalidRuleTestParams{
            "MissingClosingBrackets",
            "FragmentScheduleRule(ordered_fragments=[FragmentInfo(origins=["
            "\"f1\"(123)])->FragmentInfo(origins=[\"f2\"(456)])"}),
    [](const testing::TestParamInfo<
        FragmentScheduleRuleParserInvalidSyntaxTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(FragmentInfoMapInfoTest, IsEqual) {
  FragmentInfo info1 = MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, "m1");
  FragmentInfo info2 = MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, "m1");
  FragmentInfo info3 = MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, "m1");

  EXPECT_TRUE(FragmentInfoMapInfo::isEqual(info1, info2));
  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(info1, info3));
}

TEST(FragmentInfoMapInfoTest, GetHashValue) {
  FragmentInfo info1 = MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, "m1");
  FragmentInfo info2 = MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, "m1");
  FragmentInfo info3 = MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, "m1");

  EXPECT_EQ(FragmentInfoMapInfo::getHashValue(info1),
            FragmentInfoMapInfo::getHashValue(info2));
  // It's highly likely they are different, though not guaranteed.
  EXPECT_NE(FragmentInfoMapInfo::getHashValue(info1),
            FragmentInfoMapInfo::getHashValue(info3));
}

TEST(FragmentInfoMapInfoTest, SpecialKeys) {
  FragmentInfo emptyKey = FragmentInfoMapInfo::getEmptyKey();
  FragmentInfo tombstoneKey = FragmentInfoMapInfo::getTombstoneKey();
  FragmentInfo info1 = MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, "m1");

  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(emptyKey, info1));
  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(tombstoneKey, info1));
  EXPECT_FALSE(FragmentInfoMapInfo::isEqual(emptyKey, tombstoneKey));
}

TEST(FragmentInfoMapInfoTest, DenseMapIntegration) {
  llvm::DenseMap<FragmentInfo, int, FragmentInfoMapInfo> map;

  FragmentInfo info1 = MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, "m1");
  FragmentInfo info2 = MakeFragmentInfo({MakeFragmentOrigin("f2", 456)}, "m1");

  map[info1] = 1;
  map[info2] = 2;

  EXPECT_EQ(map.size(), 2);
  EXPECT_EQ(map[info1], 1);
  EXPECT_EQ(map[info2], 2);

  FragmentInfo info1_copy =
      MakeFragmentInfo({MakeFragmentOrigin("f1", 123)}, "m1");
  EXPECT_TRUE(map.contains(info1_copy));
  EXPECT_EQ(map[info1_copy], 1);

  map.erase(info1);
  EXPECT_EQ(map.size(), 1);
  EXPECT_FALSE(map.contains(info1));
}

}  // namespace
}  // namespace mlir::mpmd
