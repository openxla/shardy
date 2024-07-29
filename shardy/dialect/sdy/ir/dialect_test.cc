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

#include "shardy/dialect/sdy/ir/dialect.h"

#include <cstdint>
#include <optional>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

namespace {

using ::testing::HasSubstr;

class DialectTest : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  MeshAttr createMesh(ArrayRef<StringRef> axisNames) {
    SmallVector<MeshAxisAttr> meshAxisAttrs;
    meshAxisAttrs.reserve(axisNames.size());
    for (StringRef axisName : axisNames) {
      meshAxisAttrs.push_back(MeshAxisAttr::get(&context, axisName, 1));
    }
    return MeshAttr::get(&context, meshAxisAttrs);
  }

  AxisRefAttr createAxis(StringRef name) {
    return AxisRefAttr::get(&context, name);
  }

  AxisRefAttr createSubAxis(StringRef name, int64_t preSize, int64_t size) {
    return AxisRefAttr::get(&context, name, preSize, size);
  }

  DimensionShardingAttr createDimSharding(ArrayRef<AxisRefAttr> axes,
                                          bool isClosed = false) {
    return DimensionShardingAttr::get(&context, axes, isClosed);
  }

  TensorShardingAttr createTensorSharding(
      ArrayRef<DimensionShardingAttr> dimShardings,
      ArrayRef<AxisRefAttr> replicatedAxes = {}) {
    return TensorShardingAttr::get(&context, "mesh", dimShardings,
                                   replicatedAxes);
  }

  MLIRContext context;
};

TEST_F(DialectTest, AxisRefAttrContains) {
  auto strictlyContains = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a.strictlyContains(b));
    EXPECT_TRUE(a.contains(b));
    EXPECT_FALSE(b.contains(a));
    EXPECT_FALSE(b.strictlyContains(a));
  };
  strictlyContains(createAxis("x"), createSubAxis("x", 2, 4));
  strictlyContains(createSubAxis("x", 1, 4), createSubAxis("x", 1, 2));
  strictlyContains(createSubAxis("x", 2, 4), createSubAxis("x", 4, 2));
  strictlyContains(createSubAxis("x", 1, 8), createSubAxis("x", 2, 2));

  auto equals = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a.contains(b));
    EXPECT_TRUE(b.contains(a));
    EXPECT_FALSE(a.strictlyContains(b));
    EXPECT_FALSE(b.strictlyContains(a));
  };
  equals(createAxis("x"), createAxis("x"));
  equals(createSubAxis("x", 1, 4), createSubAxis("x", 1, 4));

  auto doesNotContain = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_FALSE(a.contains(b));
    EXPECT_FALSE(b.contains(a));
    EXPECT_FALSE(a.strictlyContains(b));
    EXPECT_FALSE(b.strictlyContains(a));
  };
  doesNotContain(createAxis("x"), createAxis("y"));
  doesNotContain(createAxis("x"), createSubAxis("y", 1, 2));
  doesNotContain(createSubAxis("x", 1, 4), createSubAxis("x", 4, 2));
  doesNotContain(createSubAxis("x", 1, 4), createSubAxis("x", 2, 4));
}

TEST_F(DialectTest, AxisRefAttrPrefixOf) {
  auto strictPrefixOf = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a.strictPrefixOf(b));
    EXPECT_TRUE(a.prefixOf(b));
    EXPECT_FALSE(b.prefixOf(a));
    EXPECT_FALSE(b.strictPrefixOf(a));
  };
  strictPrefixOf(createSubAxis("x", 1, 4), createAxis("x"));
  strictPrefixOf(createSubAxis("x", 1, 2), createSubAxis("x", 1, 4));
  strictPrefixOf(createSubAxis("x", 2, 2), createSubAxis("x", 2, 8));

  auto equals = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a.prefixOf(b));
    EXPECT_TRUE(b.prefixOf(a));
    EXPECT_FALSE(a.strictPrefixOf(b));
    EXPECT_FALSE(b.strictPrefixOf(a));
  };
  equals(createAxis("x"), createAxis("x"));
  equals(createSubAxis("x", 2, 4), createSubAxis("x", 2, 4));

  auto isNotPrefix = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_FALSE(a.prefixOf(b));
    EXPECT_FALSE(b.prefixOf(a));
    EXPECT_FALSE(a.strictPrefixOf(b));
    EXPECT_FALSE(b.strictPrefixOf(a));
  };
  isNotPrefix(createAxis("x"), createAxis("y"));
  isNotPrefix(createSubAxis("x", 1, 2), createAxis("y"));
  isNotPrefix(createSubAxis("x", 1, 4), createSubAxis("x", 4, 2));
  isNotPrefix(createSubAxis("x", 1, 4), createSubAxis("x", 2, 4));
}

TEST_F(DialectTest, AxisRefAttrOverlaps) {
  auto checkOverlaps = [](AxisRefAttr a, AxisRefAttr b, bool expected) {
    EXPECT_EQ(a.overlaps(b), expected);
    EXPECT_EQ(b.overlaps(a), expected);
  };

  checkOverlaps(createAxis("x"), createAxis("x"), true);
  checkOverlaps(createSubAxis("x", 2, 2), createSubAxis("x", 2, 2), true);
  checkOverlaps(createSubAxis("x", 1, 4), createSubAxis("x", 1, 2), true);

  checkOverlaps(createAxis("x"), createSubAxis("x", 2, 4), true);
  checkOverlaps(createSubAxis("x", 1, 4), createSubAxis("x", 2, 4), true);
  checkOverlaps(createSubAxis("x", 2, 8), createSubAxis("x", 4, 2), true);

  checkOverlaps(createAxis("x"), createAxis("y"), false);
  checkOverlaps(createAxis("x"), createSubAxis("y", 1, 2), false);
  checkOverlaps(createSubAxis("x", 1, 4), createSubAxis("x", 4, 2), false);
  checkOverlaps(createSubAxis("x", 1, 2), createSubAxis("x", 4, 2), false);
}

// The test cases are the same as DialectTest.AxisRefAttrOverlaps.
TEST_F(DialectTest, AxisRefAttrGetPrefixWithoutOverlap) {
  auto samePrefix = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_EQ(a.getPrefixWithoutOverlap(b), std::nullopt);
    EXPECT_EQ(b.getPrefixWithoutOverlap(a), std::nullopt);
  };
  samePrefix(createAxis("x"), createAxis("x"));
  samePrefix(createSubAxis("x", 2, 2), createSubAxis("x", 2, 2));
  samePrefix(createSubAxis("x", 1, 4), createSubAxis("x", 1, 2));

  // "x":(2)4 and "x"
  EXPECT_EQ(createSubAxis("x", 2, 4).getPrefixWithoutOverlap(createAxis("x")),
            std::nullopt);
  EXPECT_EQ(createAxis("x").getPrefixWithoutOverlap(createSubAxis("x", 2, 4)),
            createSubAxis("x", 1, 2));

  // "x":(2)4 and "x":(1)4
  EXPECT_EQ(createSubAxis("x", 2, 4).getPrefixWithoutOverlap(
                createSubAxis("x", 1, 4)),
            std::nullopt);
  EXPECT_EQ(createSubAxis("x", 1, 4).getPrefixWithoutOverlap(
                createSubAxis("x", 2, 4)),
            createSubAxis("x", 1, 2));

  // "x"(4)2 and "x":(2)8
  EXPECT_EQ(createSubAxis("x", 4, 2).getPrefixWithoutOverlap(
                createSubAxis("x", 2, 8)),
            std::nullopt);
  EXPECT_EQ(createSubAxis("x", 2, 8).getPrefixWithoutOverlap(
                createSubAxis("x", 4, 2)),
            createSubAxis("x", 2, 2));

  auto checkNoOverlap = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_EQ(a.getPrefixWithoutOverlap(b), a);
    EXPECT_EQ(b.getPrefixWithoutOverlap(a), b);
  };
  checkNoOverlap(createAxis("x"), createAxis("y"));
  checkNoOverlap(createAxis("x"), createSubAxis("y", 1, 2));
  checkNoOverlap(createSubAxis("x", 1, 4), createSubAxis("x", 4, 2));
  checkNoOverlap(createSubAxis("x", 1, 2), createSubAxis("x", 4, 2));
}

TEST_F(DialectTest, AxisRefAttrCanMerge) {
  EXPECT_TRUE(createSubAxis("x", 1, 4).canMerge(createSubAxis("x", 4, 2)));
  EXPECT_TRUE(createSubAxis("x", 2, 2).canMerge(createSubAxis("x", 4, 4)));

  EXPECT_FALSE(createAxis("x").canMerge(createAxis("x")));
  EXPECT_FALSE(createAxis("x").canMerge(createAxis("y")));
  EXPECT_FALSE(createSubAxis("x", 1, 2).canMerge(createSubAxis("y", 2, 2)));
  EXPECT_FALSE(createAxis("x").canMerge(createSubAxis("x", 4, 2)));
  EXPECT_FALSE(createSubAxis("x", 1, 2).canMerge(createSubAxis("x", 4, 2)));
  EXPECT_FALSE(createSubAxis("x", 4, 2).canMerge(createSubAxis("x", 1, 4)));
}

TEST_F(DialectTest, AxisRefAttrMerge) {
  auto mesh = MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 16)});
  EXPECT_EQ(createSubAxis("x", 1, 4).merge(createSubAxis("x", 4, 2), mesh),
            createSubAxis("x", 1, 8));
  EXPECT_EQ(createSubAxis("x", 2, 2).merge(createSubAxis("x", 4, 4), mesh),
            createSubAxis("x", 2, 8));
  EXPECT_EQ(createSubAxis("x", 1, 8).merge(createSubAxis("x", 8, 2), mesh),
            createAxis("x"));
}

TEST_F(DialectTest, TensorShardingAttrCanShardOrReplicate) {
  TensorShardingAttr sharding = createTensorSharding(
      {createDimSharding({createAxis("x"), createSubAxis("z", 2, 2)},
                         /*isClosed=*/true),
       createDimSharding(createAxis("w"), /*isClosed=*/false)},
      /*replicatedAxes=*/{createAxis("u"), createSubAxis("v", 1, 2)});

  EXPECT_FALSE(sharding.canShard(0, "x"));
  EXPECT_FALSE(sharding.canShard(0, "y"));
  EXPECT_FALSE(sharding.canShard(0, "u"));

  EXPECT_FALSE(sharding.canShard(1, "x"));
  EXPECT_FALSE(sharding.canShard(1, "z"));
  EXPECT_FALSE(sharding.canShard(1, "w"));
  EXPECT_FALSE(sharding.canShard(1, "v"));

  EXPECT_FALSE(sharding.canReplicate("x"));
  EXPECT_FALSE(sharding.canReplicate("z"));
  EXPECT_FALSE(sharding.canReplicate("u"));
  EXPECT_FALSE(sharding.canReplicate("v"));

  EXPECT_TRUE(sharding.canShard(1, "y"));
  EXPECT_TRUE(sharding.canShard(1, "a"));

  EXPECT_TRUE(sharding.canReplicate("y"));
  EXPECT_TRUE(sharding.canReplicate("a"));
}

TEST_F(DialectTest, TensorShardingAttrGetSharded) {
  DimensionShardingAttr dimSharding0 =
      createDimSharding(createSubAxis("x", 2, 2));
  DimensionShardingAttr dimSharding1 = createDimSharding({});
  SmallVector<AxisRefAttr> replicatedAxes{createAxis("u")};
  TensorShardingAttr sharding =
      createTensorSharding({dimSharding0, dimSharding1}, replicatedAxes);

  EXPECT_EQ(sharding.getSharded(0, "y"),
            createTensorSharding(
                {createDimSharding({createSubAxis("x", 2, 2), createAxis("y")}),
                 dimSharding1},
                replicatedAxes));
  EXPECT_EQ(
      sharding.getSharded(1, "y"),
      createTensorSharding({dimSharding0, createDimSharding(createAxis("y"))},
                           replicatedAxes));
}

TEST_F(DialectTest, TensorShardingAttrGetReplicated) {
  SmallVector<DimensionShardingAttr> dimShardings = {
      createDimSharding(createSubAxis("x", 2, 2), /*isClosed=*/true),
      createDimSharding({})};
  TensorShardingAttr sharding = createTensorSharding(
      dimShardings,
      /*replicatedAxes=*/{createSubAxis("v", 2, 2), createAxis("y")});

  MeshAttr mesh = createMesh({"u", "v", "w", "x", "y", "z"});

  EXPECT_EQ(sharding.getReplicated("u", mesh),
            createTensorSharding(
                dimShardings,
                {createAxis("u"), createSubAxis("v", 2, 2), createAxis("y")}));
  EXPECT_EQ(
      sharding.getReplicated("w", mesh),
      createTensorSharding(dimShardings, {createSubAxis("v", 2, 2),
                                          createAxis("w"), createAxis("y")}));
  EXPECT_EQ(
      sharding.getReplicated("z", mesh),
      createTensorSharding(dimShardings, {createSubAxis("v", 2, 2),
                                          createAxis("y"), createAxis("z")}));
}

TEST_F(DialectTest, DimensionShardingAttrGetSharded) {
  DimensionShardingAttr dimSharding = createDimSharding({},
                                                        /*isClosed=*/false);
  ASSERT_TRUE(dimSharding.emptyAxes());
  EXPECT_EQ(dimSharding.getSharded("y"),
            createDimSharding({createAxis("y")}, /*isClosed=*/false));
}

TEST_F(DialectTest,
       DimensionShardingAttrGetShardedCannotDoubleShardOnSameAxis) {
  DimensionShardingAttr dimSharding = createDimSharding(createAxis("y"),
                                                        /*isClosed=*/false);
  EXPECT_DEBUG_DEATH(dimSharding.getSharded("y"),
                     HasSubstr("cannot shard along an already bound axis"));
}

TEST_F(DialectTest, DimensionShardingAttrGetShardedCannotShardClosed) {
  DimensionShardingAttr dimSharding = createDimSharding({createAxis("y")},
                                                        /*isClosed=*/true);
  ASSERT_TRUE(dimSharding.getIsClosed());
  EXPECT_DEBUG_DEATH(dimSharding.getSharded("x"),
                     HasSubstr("cannot shard a closed dimension"));
}

TEST_F(DialectTest, DimensionShardingAttrGetShardedSize) {
  auto mesh = MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 4),
                                       MeshAxisAttr::get(&context, "y", 2),
                                       MeshAxisAttr::get(&context, "z", 3)});

  SmallVector<DimensionShardingAttr> dimShardings = {
      createDimSharding({createAxis("x"), createAxis("y")}),
      createDimSharding({createAxis("z")})};

  EXPECT_EQ(dimShardings[0].getShardedSize(mesh), 8);
  EXPECT_EQ(dimShardings[1].getShardedSize(mesh), 3);
}
}  // namespace

}  // namespace sdy
}  // namespace mlir
