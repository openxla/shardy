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

#include "shardy/dialect/mpmd/transforms/export/naming_utils.h"

#include <cstdint>
#include <optional>
#include <string>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::mpmd {

namespace {

using ::testing::EndsWith;

UserOriginAttr GetUserOrigin(MLIRContext* ctx, StringRef name,
                             int64_t transpose_count) {
  return UserOriginAttr::get(ctx, StringAttr::get(ctx, name), transpose_count);
}

FragmentOp CreateMockFragment(MLIRContext* ctx, ArrayRef<Attribute> origins,
                              std::optional<int64_t> stage_id = std::nullopt,
                              StringRef inferred_by = "") {
  OpBuilder builder(ctx);
  Location loc = UnknownLoc::get(ctx);
  ArrayAttr origins_attr = ArrayAttr::get(ctx, origins);
  IntegerAttr stage_id_attr =
      stage_id ? IntegerAttr::get(IntegerType::get(ctx, 64), *stage_id)
               : nullptr;

  auto fragment = FragmentOp::create(builder, loc, TypeRange(), ValueRange(),
                                     origins_attr, "mock_mesh", stage_id_attr);
  if (!inferred_by.empty()) {
    fragment->setAttr("mpmd.inferred_by", StringAttr::get(ctx, inferred_by));
  }
  return fragment;
}

std::string GetFullNameFromMockFragment(FragmentOp fragment,
                                        bool is_all_forward = false) {
  std::string name = GetFullNameFromFragment(fragment, is_all_forward);
  fragment->destroy();
  return name;
}

TEST(InformativeFragmentNameTests, EmptyOrigin) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({}), "inferred");
}

TEST(InformativeFragmentNameTests, SingleOriginUnsplittable) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "f_25_g", 0)}),
            "f_25_g");
}

TEST(InformativeFragmentNameTests, SingleOriginUnsplittableTransposed) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "f_25_g", 3)}),
            R"origin(f_25_g(3))origin");
}

TEST(InformativeFragmentNameTests, SingleOriginSplittable) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "f_2", 0)}), "f_2");
}

TEST(InformativeFragmentNameTests, SingleOriginSplittableTransposed) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "f_2", 1)}),
            R"origin(f(1)_2)origin");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocks) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "block_0", 0),
                                        GetUserOrigin(&ctx, "block_1", 0),
                                        GetUserOrigin(&ctx, "block_2", 0)}),
            "block_0:2");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocksTransposed) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "block_2", 1),
                                        GetUserOrigin(&ctx, "block_1", 1),
                                        GetUserOrigin(&ctx, "block_0", 1)}),
            R"origin(block(1)_0:2)origin");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocksBroken) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "block_0", 0),
                                        GetUserOrigin(&ctx, "block_2", 0),
                                        GetUserOrigin(&ctx, "block_3", 0)}),
            "block_0.block_2:3");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocksBrokenByTranspose) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "block_0", 0),
                                        GetUserOrigin(&ctx, "block_1", 1),
                                        GetUserOrigin(&ctx, "block_2", 1)}),
            R"origin(block_0.block(1)_1:2)origin");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocksBrokenAtEnd) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "block_0", 0),
                                        GetUserOrigin(&ctx, "block_1", 0),
                                        GetUserOrigin(&ctx, "loss", 0)}),
            "block_0:1.loss");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocksBrokenAtStart) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "encoder", 0),
                                        GetUserOrigin(&ctx, "block_0", 0),
                                        GetUserOrigin(&ctx, "block_1", 0)}),
            "encoder.block_0:1");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocksBrokenAtEndByTranspose) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetInformativeFragmentName({GetUserOrigin(&ctx, "block_0", 0),
                                        GetUserOrigin(&ctx, "block_1", 0),
                                        GetUserOrigin(&ctx, "block_2", 0),
                                        GetUserOrigin(&ctx, "block_2", 1)}),
            R"origin(block_0:2.block(1)_2)origin");
}

TEST(InformativeFragmentNameTests, ConsecutiveBlocksBrokenAtMiddle) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(
      GetInformativeFragmentName(
          {GetUserOrigin(&ctx, "block_0", 0), GetUserOrigin(&ctx, "block_1", 0),
           GetUserOrigin(&ctx, "loss", 0), GetUserOrigin(&ctx, "block_1", 1),
           GetUserOrigin(&ctx, "block_0", 1)}),
      R"origin(block_0:1.loss.block(1)_0:1)origin");
}

TEST(TruncateTest, SmallString) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(Truncate("abcdefg", 32), "abcdefg");
}

TEST(TruncateTest, BigString) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  std::string truncated =
      Truncate("12345678123456781234567812345678andsomemore", 32);
  EXPECT_EQ(truncated.size(), 32);
  EXPECT_THAT(truncated, EndsWith("<...>"));
  EXPECT_EQ(truncated, "123456781234567812345678123<...>");
}

TEST(GetFullNameFromMetadataTests, EmptyOriginStage) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(CreateMockFragment(&ctx, {})),
            "inferred");
}

TEST(GetFullNameFromMetadataTests, EmptyOriginWithStage) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(CreateMockFragment(&ctx, {}, 3)),
            "stage3");
}

TEST(GetFullNameFromMetadataTests, SingleOriginWithoutStage) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "foo", 0)})),
            "foo_fwd");
}

TEST(GetFullNameFromMetadataTests, SingleOriginWithMultipleTransposes) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "bar", 5)})),
            "bar_transpose5");
}

TEST(GetFullNameFromMetadataTests,
     ManyOriginWithCommonNamesButSameBlockCounter) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "block_6", 0),
                                          GetUserOrigin(&ctx, "block_6", 0),
                                          GetUserOrigin(&ctx, "block_6", 0)})),
            "block_6_fwd");
}

TEST(GetFullNameFromMetadataTests,
     ManyOriginWithCommonNamesAndDifferentBlockCounters) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "block_6", 0),
                                          GetUserOrigin(&ctx, "block_7", 0),
                                          GetUserOrigin(&ctx, "block_8", 0)})),
            "block_6:8_fwd");
}

TEST(GetFullNameFromMetadataTests,
     ManyOriginWithCommonNamesWithReverseBlockCountersButFwd) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "block_6", 0),
                                          GetUserOrigin(&ctx, "block_5", 0),
                                          GetUserOrigin(&ctx, "block_4", 0)})),
            "block_4:6_fwd");
}

TEST(GetFullNameFromMetadataTests,
     ManyOriginWithCommonNamesAssumesContiguousBlockCounters) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "block_6", 0),
                                          GetUserOrigin(&ctx, "block_4", 0)})),
            "block_4:6_fwd");
}

TEST(GetFullNameFromMetadataTests, ManyOriginWithCommonNamesAndBwd) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "block_6", 1),
                                          GetUserOrigin(&ctx, "block_5", 1),
                                          GetUserOrigin(&ctx, "block_4", 1)})),
            "block_6:4_bwd");
}

TEST(GetFullNameFromMetadataTests, PickMostPopularNameandDropLeastPopular) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(
      GetFullNameFromMockFragment(CreateMockFragment(
          &ctx,
          {GetUserOrigin(&ctx, "block_6", 1), GetUserOrigin(&ctx, "block_5", 1),
           GetUserOrigin(&ctx, "block_4", 1), GetUserOrigin(&ctx, "scan", 1)})),
      "block_6:4..._bwd");
}

TEST(GetFullNameFromMetadataTests, PickMostPopularNameandDropLeastPopular2) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(
      GetFullNameFromMockFragment(CreateMockFragment(
          &ctx,
          {GetUserOrigin(&ctx, "scan_6", 1), GetUserOrigin(&ctx, "scan_5", 1),
           GetUserOrigin(&ctx, "scan_4", 1), GetUserOrigin(&ctx, "block", 1)})),
      "scan_6:4..._bwd");
}

TEST(GetFullNameFromMetadataTests, MixedFwdAndBwd) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  // The block numbers aren't reversed as this fragment isn't completely bwd
  // (i.e. the scan origin has transpose count = 0).
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "block_6", 1),
                                          GetUserOrigin(&ctx, "block_5", 1),
                                          GetUserOrigin(&ctx, "scan", 0)})),
            "block_5:6..._fwd_bwd");
}

TEST(GetFullNameFromMetadataTests, SameNameFwdAndBwd) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "foo", 0),
                                          GetUserOrigin(&ctx, "foo", 1)})),
            "foo_fwd_bwd");
}

TEST(GetFullNameFromMetadataTests, DifferentNamesFwdAndBwd) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  EXPECT_EQ(GetFullNameFromMockFragment(
                CreateMockFragment(&ctx, {GetUserOrigin(&ctx, "foo", 0),
                                          GetUserOrigin(&ctx, "bar", 1)})),
            "bar..._fwd_bwd");
}

TEST(GetCallSitesSummaryNameTests, SingleCallSiteWithCallCounter) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {{"mesh1", {{"foo", 0}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "foo_call0");
}

TEST(GetCallSitesSummaryNameTests, SingleCallSiteWithoutCallCounter) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {{"mesh1", {{"foo", std::nullopt}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "foo");
}

TEST(GetCallSitesSummaryNameTests,
     ManyCallsInSingleMeshWithAscendingAndContiguousCallCounter) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {{"mesh1", {{"x", 0}, {"x", 1}, {"x", 2}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "x_calls0to2");
}

TEST(GetCallSitesSummaryNameTests,
     ManyCallsInSingleMeshWithDescendingAndContiguousCallCounter) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {{"mesh1", {{"x", 2}, {"x", 1}, {"x", 0}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "x_calls2from0");
}

TEST(GetCallSitesSummaryNameTests, ManyCallsInSingleMeshWithInconsistentNames) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {{"mesh1", {{"x", 0}, {"y", 1}, {"z", 2}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "x");
}

TEST(GetCallSitesSummaryNameTests,
     ManyCallsInSingleMeshWithDescendingButNotContiguousCallCounter) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {{"mesh1", {{"x", 3}, {"x", 1}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "x");
}

TEST(GetCallSitesSummaryNameTests, MultipleMeshesWithConsistentCallCounters) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {{"mesh1", {{"x", 0}, {"x", 1}, {"x", 2}}},
                                     {"mesh2", {{"y", 0}, {"y", 1}, {"y", 2}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "x_calls0to2");
}

TEST(GetCallSitesSummaryNameTests, MultipleMeshesWithInconsistentCallCounters) {
  MLIRContext ctx;
  ctx.loadDialect<MpmdDialect>();
  MeshToCallSites mesh_call_sites = {
      {"mesh1", {{"x", 0}, {"x", 1}, {"x", 2}}},
      {"mesh2", {{"y", 0}, {"y", std::nullopt}, {"y", 3}}}};
  EXPECT_EQ(GetCallSitesSummaryName(mesh_call_sites), "x");
}

}  // namespace
}  // namespace mlir::mpmd
