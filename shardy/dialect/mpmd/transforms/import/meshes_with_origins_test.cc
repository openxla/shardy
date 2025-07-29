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

#include "shardy/dialect/mpmd/transforms/import/meshes_with_origins.h"

#include <optional>
#include <utility>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_inference_origins.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir::mpmd {
namespace {

using ::testing::UnorderedElementsAre;

MeshesWithOrigins GetMeshesWithOrigins(
    MLIRContext& context,
    ArrayRef<std::pair<StringRef, std::optional<StringRef>>> mesh_and_origins) {
  SmallVector<MeshWithOriginsAttr> mesh_with_origins_attrs;
  for (auto [mesh, origin] : mesh_and_origins) {
    mesh_with_origins_attrs.push_back(MeshWithOriginsAttr::get(
        &context, mesh,
        origin ? ArrayRef<OriginAttr>(OriginAttr::get(&context, *origin))
               : ArrayRef<OriginAttr>()));
  }
  return MeshesWithOrigins(
      MeshesWithOriginsAttr::get(&context, mesh_with_origins_attrs));
}

MATCHER_P(MeshesWithOriginsAttrEq, expected, "") {
  if (arg != expected) {
    *result_listener << "\n Expected: \n\t" << debugString(expected)
                     << "\n Actual: \n\t" << debugString(arg);
    return false;
  }
  return true;
}

}  // namespace

TEST(MeshesWithOrigins, BasicFunctionality) {
  MLIRContext context;
  context.loadDialect<MpmdDialect>();

  MeshesWithOrigins m1;
  EXPECT_EQ(m1.size(), -1);
  EXPECT_FALSE(m1.empty());
  EXPECT_FALSE(m1.has_meshes_specified());

  MeshesWithOrigins m2 = GetMeshesWithOrigins(context, {{"m1", {}}});
  EXPECT_EQ(m2.size(), 1);
  EXPECT_FALSE(m2.empty());
  EXPECT_TRUE(m2.has_meshes_specified());
  EXPECT_THAT(m2.MeshNames(), UnorderedElementsAre("m1"));

  MeshesWithOrigins m3 = GetMeshesWithOrigins(context, {});
  EXPECT_EQ(m3.size(), 0);
  EXPECT_TRUE(m3.empty());
  EXPECT_TRUE(m3.has_meshes_specified());

  MeshesWithOrigins m4 = MeshesWithOrigins::CreateUseSet({});
  EXPECT_EQ(m4.size(), 0);
  EXPECT_TRUE(m4.empty());
  EXPECT_TRUE(m4.has_meshes_specified());
}

TEST(MeshesWithOrigins, Union) {
  MLIRContext context;
  OpBuilder builder(&context);
  context.loadDialect<MpmdDialect>();

  MeshesWithOrigins m1 = GetMeshesWithOrigins(context, {{"m1", {}}});
  MeshesWithOrigins m2 = GetMeshesWithOrigins(context, {{"m1", "o1"}});
  MeshesWithOrigins m3 = GetMeshesWithOrigins(context, {{"m2", {}}});

  MeshesWithOrigins m4_wildcard =
      GetMeshesWithOrigins(context, {{"m2", "o1"}, {kWildcardMesh, "o2"}});
  MeshesWithOrigins m5_wildcard =
      GetMeshesWithOrigins(context, {{"m2", "o1"}, {kWildcardMesh, "o3"}});

  m1.Union(m2);
  m1.Union(m3);
  EXPECT_FALSE(m1.has_wildcard_mesh());
  EXPECT_TRUE(m4_wildcard.has_wildcard_mesh());

  m1.Union(m4_wildcard);
  EXPECT_THAT(m1.MeshNames(/*include_wildcard_mesh=*/true),
              UnorderedElementsAre("m1", "m2", "*"));
  EXPECT_TRUE(m1.has_wildcard_mesh());

  m1.Union(m5_wildcard);
  EXPECT_THAT(m1.MeshNames(/*include_wildcard_mesh=*/true),
              UnorderedElementsAre("m1", "m2", "*"));
  EXPECT_THAT(
      m1.ToAttr(builder),
      MeshesWithOriginsAttrEq(MeshesWithOriginsAttr::get(
          &context,
          {MeshWithOriginsAttr::get(&context, "m1",
                                    {OriginAttr::get(&context, "o1")}),
           MeshWithOriginsAttr::get(&context, "m2",
                                    {OriginAttr::get(&context, "o1")}),
           MeshWithOriginsAttr::get(&context, kWildcardMesh,
                                    {OriginAttr::get(&context, "o2"),
                                     OriginAttr::get(&context, "o3")})})));

  MeshesWithOrigins m4;
  EXPECT_FALSE(m4.has_meshes_specified());

  MeshesWithOrigins m5;
  m4.Union(m5);
  EXPECT_FALSE(m5.has_meshes_specified());

  m4.Union(m1);
  EXPECT_TRUE(m4.has_meshes_specified());
}

TEST(MeshesWithOrigins, Intersect) {
  MLIRContext context;
  context.loadDialect<MpmdDialect>();

  MeshesWithOrigins m1 = GetMeshesWithOrigins(context, {{"m1", {}}});
  MeshesWithOrigins m2 = GetMeshesWithOrigins(context, {{"m1", "o1"}});
  MeshesWithOrigins m3 = GetMeshesWithOrigins(context, {{"m2", {}}});

  m1.Intersect(m2);
  EXPECT_THAT(m1.MeshNames(), UnorderedElementsAre("m1"));

  m1.Intersect(m3);
  EXPECT_TRUE(m1.empty());

  MeshesWithOrigins m4;
  EXPECT_FALSE(m4.has_meshes_specified());

  MeshesWithOrigins m5;
  m4.Intersect(m5);
  EXPECT_FALSE(m5.has_meshes_specified());

  m4.Intersect(m2);
  EXPECT_TRUE(m4.has_meshes_specified());
  EXPECT_THAT(m4.MeshNames(), UnorderedElementsAre("m1"));
}

TEST(MeshesWithOrigins, IntersectWithWildcard) {
  MLIRContext context;
  OpBuilder builder(&context);
  context.loadDialect<MpmdDialect>();

  MeshesWithOrigins m1 = GetMeshesWithOrigins(context, {{"m1", {}}});
  MeshesWithOrigins m2_wildcard =
      GetMeshesWithOrigins(context, {{"m1", "o1"}, {kWildcardMesh, "o3"}});
  MeshesWithOrigins m3_wildcard =
      GetMeshesWithOrigins(context, {{"m2", "o2"}, {kWildcardMesh, "o4"}});

  m1.Intersect(m2_wildcard);
  EXPECT_THAT(m1.MeshNames(/*include_wildcard_mesh=*/true),
              UnorderedElementsAre("m1"));

  m1.Intersect(m3_wildcard);
  EXPECT_THAT(m1.MeshNames(/*include_wildcard_mesh=*/true),
              UnorderedElementsAre("m1"));

  // Intersect two wildcard sets.
  m2_wildcard.Intersect(m3_wildcard);
  EXPECT_THAT(m2_wildcard.MeshNames(/*include_wildcard_mesh=*/true),
              UnorderedElementsAre("m1", "m2", "*"));
  EXPECT_THAT(
      m2_wildcard.ToAttr(builder),
      MeshesWithOriginsAttrEq(MeshesWithOriginsAttr::get(
          &context,
          {MeshWithOriginsAttr::get(&context, "m1",
                                    {OriginAttr::get(&context, "o1"),
                                     OriginAttr::get(&context, "o4")}),
           MeshWithOriginsAttr::get(&context, "m2",
                                    {OriginAttr::get(&context, "o2"),
                                     OriginAttr::get(&context, "o3")}),
           MeshWithOriginsAttr::get(&context, kWildcardMesh,
                                    {OriginAttr::get(&context, "o3"),
                                     OriginAttr::get(&context, "o4")})})));

  // Intersect with an empty set.
  MeshesWithOrigins m4;
  EXPECT_TRUE(m4.MeshNamesOrEmpty(/*include_wildcard_mesh=*/true).empty());
  m4.Intersect(m3_wildcard);
  EXPECT_THAT(
      m4.ToAttr(builder),
      MeshesWithOriginsAttrEq(MeshesWithOriginsAttr::get(
          &context,
          {MeshWithOriginsAttr::get(&context, "m2",
                                    {OriginAttr::get(&context, "o2")}),
           MeshWithOriginsAttr::get(&context, kWildcardMesh,
                                    {OriginAttr::get(&context, "o4")})})));
}

TEST(MeshesWithOrigins, Insert) {
  MLIRContext context;
  context.loadDialect<MpmdDialect>();
  MeshesWithOrigins m;
  EXPECT_FALSE(m.has_meshes_specified());

  m.insert(MeshWithOriginsAttr::get(&context, "m1", {}));
  EXPECT_TRUE(m.has_meshes_specified());
  EXPECT_THAT(m.MeshNames(), UnorderedElementsAre("m1"));

  m.insert(MeshWithOriginsAttr::get(&context, "m2", {}));
  EXPECT_THAT(m.MeshNames(), UnorderedElementsAre("m1", "m2"));

  m.insert(MeshWithOriginsAttr::get(&context, kWildcardMesh,
                                    OriginAttr::get(&context, "o1")));
  EXPECT_THAT(m.MeshNames(/*include_wildcard_mesh=*/true),
              UnorderedElementsAre("m1", "m2", "*"));

  m.insert(MeshWithOriginsAttr::get(&context, kWildcardMesh,
                                    OriginAttr::get(&context, "o2")));
  OpBuilder builder(&context);
  EXPECT_THAT(m.ToAttr(builder),
              MeshesWithOriginsAttrEq(MeshesWithOriginsAttr::get(
                  &context, {MeshWithOriginsAttr::get(&context, "m1", {}),
                             MeshWithOriginsAttr::get(&context, "m2", {}),
                             MeshWithOriginsAttr::get(
                                 &context, kWildcardMesh,
                                 {OriginAttr::get(&context, "o1"),
                                  OriginAttr::get(&context, "o2")})})));
}

TEST(MeshesWithOrigins, ToAttr) {
  MLIRContext context;
  context.loadDialect<MpmdDialect>();
  MeshesWithOrigins m1 =
      GetMeshesWithOrigins(context, {{"m1", "o1"}, {"m2", "o2"}});
  OpBuilder builder(&context);
  MeshesWithOriginsAttr attr = m1.ToAttr(builder);
  EXPECT_EQ(attr.size(), 2);
  EXPECT_EQ(
      attr,
      MeshesWithOriginsAttr::get(
          &context, {MeshWithOriginsAttr::get(
                         &context, "m1", {OriginAttr::get(&context, "o1")}),
                     MeshWithOriginsAttr::get(
                         &context, "m2", {OriginAttr::get(&context, "o2")})}));

  MeshesWithOrigins m2;
  EXPECT_EQ(m2.ToAttr(builder), nullptr);

  MeshesWithOrigins m3 =
      GetMeshesWithOrigins(context, {{"m1", "o1"}, {kWildcardMesh, "o2"}});
  EXPECT_THAT(
      m3.ToAttr(builder),
      MeshesWithOriginsAttrEq(MeshesWithOriginsAttr::get(
          &context,
          {MeshWithOriginsAttr::get(&context, "m1",
                                    {OriginAttr::get(&context, "o1")}),
           MeshWithOriginsAttr::get(&context, kWildcardMesh,
                                    {OriginAttr::get(&context, "o2")})})));
}

TEST(MeshesWithOrigins, HasSameMeshes) {
  MLIRContext context;
  context.loadDialect<MpmdDialect>();

  MeshesWithOrigins m1 =
      GetMeshesWithOrigins(context, {{"m1", "o1"}, {"m2", "o2"}});
  MeshesWithOrigins m2 =
      GetMeshesWithOrigins(context, {{"m1", {}}, {"m2", {}}});
  MeshesWithOrigins m3 = GetMeshesWithOrigins(context, {{"m1", {}}});

  // Ignores origin and order
  EXPECT_TRUE(m1.HasSameMeshes(m2));
  EXPECT_TRUE(m2.HasSameMeshes(m1));
  EXPECT_FALSE(m1.HasSameMeshes(m3));
  EXPECT_FALSE(m3.HasSameMeshes(m1));

  MeshesWithOrigins m4;
  MeshesWithOrigins m5;
  EXPECT_TRUE(m4.HasSameMeshes(m5));
  EXPECT_FALSE(m4.HasSameMeshes(m1));
  EXPECT_FALSE(m1.HasSameMeshes(m4));

  MeshesWithOrigins m6_wildcard = GetMeshesWithOrigins(
      context, {{"m1", {}}, {"m2", {}}, {kWildcardMesh, "o2"}});
  MeshesWithOrigins m7_wildcard = GetMeshesWithOrigins(
      context, {{"m1", {}}, {"m2", "o1"}, {kWildcardMesh, "o3"}});
  EXPECT_TRUE(m6_wildcard.HasSameMeshes(m7_wildcard));
  EXPECT_FALSE(m6_wildcard.HasSameMeshes(m1));
  EXPECT_FALSE(m6_wildcard.HasSameMeshes(m2));
}

TEST(MeshesWithOrigins, GetPrioritizedMeshName) {
  MLIRContext context;
  context.loadDialect<MpmdDialect>();
  MeshesWithOrigins m1 =
      GetMeshesWithOrigins(context, {{"m2", "o1"}, {"m1", "o2"}});
  EXPECT_EQ(m1.GetPrioritizedMeshName(), "m1");

  MeshesWithOrigins m2 = MeshesWithOrigins::CreateUseSet({});
  EXPECT_EQ(m2.GetPrioritizedMeshName(), std::nullopt);

  SetVector<StringRef> preferred_mesh_names;
  preferred_mesh_names.insert("m3");
  EXPECT_EQ(m1.GetPrioritizedMeshName(preferred_mesh_names), "m1");
  preferred_mesh_names.insert("m2");
  EXPECT_EQ(m1.GetPrioritizedMeshName(preferred_mesh_names), "m2");
}

TEST(MeshesWithOrigins, GetPrioritizedMeshNameHandlingLowPriorityOrigins) {
  MLIRContext context;
  context.loadDialect<MpmdDialect>();

  StringRef low_priority_origin = kBroadcastInputOrigin;
  MeshesWithOrigins m =
      GetMeshesWithOrigins(context, {{"m1", low_priority_origin},
                                     {"m2", "o2"},
                                     {"m3", low_priority_origin},
                                     {"m4", "o4"},
                                     {kWildcardMesh, low_priority_origin}});
  EXPECT_EQ(m.GetPrioritizedMeshName(), "m2");

  SetVector<StringRef> preferred_mesh_names;
  preferred_mesh_names.insert("non_existent_mesh_2");
  EXPECT_EQ(m.GetPrioritizedMeshName(preferred_mesh_names),
            "non_existent_mesh_2");
  preferred_mesh_names.insert("non_existent_mesh_1");
  EXPECT_EQ(m.GetPrioritizedMeshName(preferred_mesh_names),
            "non_existent_mesh_1");

  preferred_mesh_names.insert("m3");
  EXPECT_EQ(m.GetPrioritizedMeshName(preferred_mesh_names), "m3");

  preferred_mesh_names.insert("m1");
  EXPECT_EQ(m.GetPrioritizedMeshName(preferred_mesh_names), "m1");

  preferred_mesh_names.insert("m4");
  EXPECT_EQ(m.GetPrioritizedMeshName(preferred_mesh_names), "m4");
  preferred_mesh_names.insert("m2");
  EXPECT_EQ(m.GetPrioritizedMeshName(preferred_mesh_names), "m2");
}

}  // namespace mlir::mpmd
