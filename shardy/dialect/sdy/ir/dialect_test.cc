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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/testing_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

namespace {

using ::testing::HasSubstr;

class DialectTest : public ShardyTestBase {
 protected:
  DimensionShardingAttr createDimSharding(ArrayRef<AxisRefAttr> axes,
                                          bool isClosed = false) {
    return DimensionShardingAttr::get(&context, axes, isClosed);
  }

  TensorShardingAttr createTensorSharding(
      ArrayRef<DimensionShardingAttr> dimShardings,
      ArrayRef<AxisRefAttr> replicatedAxes = {},
      ArrayRef<AxisRefAttr> unreducedAxes = {}) {
    return TensorShardingAttr::get(&context, "mesh", dimShardings,
                                   replicatedAxes, unreducedAxes);
  }

  DimMappingAttr createDimMapping(ArrayRef<int64_t> factorIndices) {
    return DimMappingAttr::get(&context, factorIndices);
  }

  TensorMappingAttr createTensorMapping(ArrayRef<DimMappingAttr> dimMappings) {
    return TensorMappingAttr::get(&context, dimMappings);
  }

  OpShardingRuleAttr createOpShardingRule(
      ArrayRef<int64_t> factorSizes,
      ArrayRef<TensorMappingAttr> operandMappings,
      ArrayRef<TensorMappingAttr> resultMappings,
      ArrayRef<int64_t> reductionFactors = {},
      ArrayRef<int64_t> needReplicationFactors = {},
      ArrayRef<int64_t> permutationFactors = {}, bool isCustomRule = false) {
    return OpShardingRuleAttr::get(&context, factorSizes, operandMappings,
                                   resultMappings, reductionFactors,
                                   needReplicationFactors, permutationFactors,
                                   isCustomRule);
  }
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

TEST_F(DialectTest, AxisRefAttrSuffixOf) {
  auto mesh = MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 8),
                                       MeshAxisAttr::get(&context, "y", 4)});
  auto strictSuffixOf = [mesh](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a.strictSuffixOf(b, mesh));
    EXPECT_TRUE(a.suffixOf(b, mesh));
    EXPECT_FALSE(b.suffixOf(a, mesh));
    EXPECT_FALSE(b.strictSuffixOf(a, mesh));
  };
  strictSuffixOf(createSubAxis("x", 2, 4), createAxis("x"));
  strictSuffixOf(createSubAxis("x", 4, 2), createSubAxis("x", 2, 4));
  strictSuffixOf(createSubAxis("x", 2, 2), createSubAxis("x", 1, 4));

  auto equals = [mesh](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a == b);
    EXPECT_TRUE(a.suffixOf(b, mesh));
    EXPECT_TRUE(b.suffixOf(a, mesh));
    EXPECT_FALSE(a.strictSuffixOf(b, mesh));
    EXPECT_FALSE(b.strictSuffixOf(a, mesh));
  };
  equals(createAxis("x"), createAxis("x"));
  equals(createSubAxis("x", 2, 4), createSubAxis("x", 2, 4));

  auto isNotSuffix = [mesh](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_FALSE(a.suffixOf(b, mesh));
    EXPECT_FALSE(b.suffixOf(a, mesh));
    EXPECT_FALSE(a.strictSuffixOf(b, mesh));
    EXPECT_FALSE(b.strictSuffixOf(a, mesh));
  };
  isNotSuffix(createAxis("x"), createAxis("y"));
  isNotSuffix(createSubAxis("x", 1, 2), createSubAxis("x", 2, 4));
  isNotSuffix(createSubAxis("x", 1, 4), createSubAxis("x", 2, 4));
  isNotSuffix(createSubAxis("x", 1, 4), createAxis("x"));
  isNotSuffix(createSubAxis("x", 2, 2), createAxis("x"));
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

TEST_F(DialectTest, AxisRefAttrCoexists) {
  auto canCoexist = [](AxisRefAttr a, AxisRefAttr b, bool expected) {
    EXPECT_EQ(a.canCoexist(b), expected);
    EXPECT_EQ(b.canCoexist(a), expected);
  };

  canCoexist(createAxis("x"), createAxis("y"), true);
  canCoexist(createAxis("x"), createSubAxis("y", 2, 2), true);
  canCoexist(createAxis("x"), createAxis("x"), true);
  canCoexist(createAxis("x"), createSubAxis("x", 2, 2), true);
  canCoexist(createSubAxis("x", 2, 2), createSubAxis("x", 2, 2), true);
  canCoexist(createSubAxis("x", 1, 2), createSubAxis("x", 1, 4), true);
  canCoexist(createSubAxis("x", 1, 2), createSubAxis("x", 2, 4), true);
  canCoexist(createSubAxis("x", 1, 2), createSubAxis("x", 6, 2), true);
  canCoexist(createSubAxis("x", 1, 4), createSubAxis("x", 2, 2), true);
  canCoexist(createSubAxis("x", 1, 4), createSubAxis("x", 2, 4), true);

  canCoexist(createSubAxis("x", 1, 2), createSubAxis("x", 1, 3), false);
  canCoexist(createSubAxis("x", 1, 2), createSubAxis("x", 3, 2), false);
  canCoexist(createSubAxis("x", 1, 3), createSubAxis("x", 2, 3), false);
}

TEST_F(DialectTest, AxisRefAttrCompare) {
  auto compare = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < a);
    EXPECT_FALSE(b < b);
  };

  compare(createAxis("x"), createAxis("y"));
  compare(createSubAxis("x", 1, 4), createSubAxis("y", 2, 4));
  compare(createAxis("x"), createSubAxis("y", 2, 4));
  compare(createSubAxis("x", 1, 4), createAxis("y"));
  compare(createSubAxis("x", 1, 4), createSubAxis("x", 2, 4));
  compare(createSubAxis("x", 1, 4), createSubAxis("x", 1, 8));
  compare(createSubAxis("x", 1, 4), createAxis("x"));
  compare(createSubAxis("x", 1, 4), createSubAxis("x", 2, 2));
}

TEST_F(DialectTest, AxisRefAttrGetOverlap) {
  auto contained = [](AxisRefAttr small, AxisRefAttr large) {
    EXPECT_TRUE(large.contains(small));
    EXPECT_EQ(large.getOverlap(small), small);
    EXPECT_EQ(small.getOverlap(large), small);
  };
  contained(createAxis("x"), createAxis("x"));
  contained(createSubAxis("x", 1, 4), createAxis("x"));
  contained(createSubAxis("x", 4, 8), createAxis("x"));
  contained(createSubAxis("x", 2, 2), createSubAxis("x", 2, 2));
  contained(createSubAxis("x", 1, 2), createSubAxis("x", 1, 4));
  contained(createSubAxis("x", 2, 2), createSubAxis("x", 1, 4));
  contained(createSubAxis("x", 2, 2), createSubAxis("x", 1, 8));

  auto overlaps = [](AxisRefAttr a, AxisRefAttr b, AxisRefAttr expected) {
    EXPECT_EQ(a.getOverlap(b), expected);
    EXPECT_EQ(b.getOverlap(a), expected);
  };
  overlaps(createSubAxis("x", 1, 4), createSubAxis("x", 2, 4),
           createSubAxis("x", 2, 2));
  overlaps(createSubAxis("x", 4, 4), createSubAxis("x", 2, 4),
           createSubAxis("x", 4, 2));

  auto checkNoOverlap = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_FALSE(a.overlaps(b));
    EXPECT_EQ(a.getOverlap(b), std::nullopt);
    EXPECT_EQ(b.getOverlap(a), std::nullopt);
  };
  checkNoOverlap(createAxis("x"), createAxis("y"));
  checkNoOverlap(createAxis("x"), createSubAxis("y", 1, 2));
  checkNoOverlap(createSubAxis("x", 1, 4), createSubAxis("x", 4, 2));
  checkNoOverlap(createSubAxis("x", 1, 2), createSubAxis("x", 4, 2));

  auto checkCannotCoexist = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_FALSE(a.canCoexist(b));
    EXPECT_EQ(a.getOverlap(b), std::nullopt);
    EXPECT_EQ(b.getOverlap(a), std::nullopt);
  };
  checkCannotCoexist(createSubAxis("x", 1, 2), createSubAxis("x", 3, 2));
  checkCannotCoexist(createSubAxis("x", 1, 3), createSubAxis("x", 2, 3));
  checkCannotCoexist(createSubAxis("x", 2, 3), createSubAxis("x", 3, 2));
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

  auto checkCannotCoexist = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_EQ(a.getPrefixWithoutOverlap(b), std::nullopt);
    EXPECT_EQ(b.getPrefixWithoutOverlap(a), std::nullopt);
  };
  checkCannotCoexist(createSubAxis("x", 1, 2), createSubAxis("x", 3, 2));
  checkCannotCoexist(createSubAxis("x", 1, 3), createSubAxis("x", 2, 3));
  checkCannotCoexist(createSubAxis("x", 2, 3), createSubAxis("x", 3, 2));
}

TEST_F(DialectTest, AxisRefAttrGetSuffixWithoutOverlap) {
  auto mesh = MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 16),
                                       MeshAxisAttr::get(&context, "y", 4)});
  auto sameSuffix = [&](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_EQ(a.getSuffixWithoutOverlap(b, mesh), std::nullopt);
    EXPECT_EQ(b.getSuffixWithoutOverlap(a, mesh), std::nullopt);
  };
  sameSuffix(createAxis("x"), createAxis("x"));
  sameSuffix(createSubAxis("x", 2, 2), createSubAxis("x", 2, 2));
  sameSuffix(createSubAxis("x", 1, 4), createSubAxis("x", 2, 2));

  // "x":(2)4 and "x"
  EXPECT_EQ(
      createSubAxis("x", 2, 4).getSuffixWithoutOverlap(createAxis("x"), mesh),
      std::nullopt);
  EXPECT_EQ(
      createAxis("x").getSuffixWithoutOverlap(createSubAxis("x", 2, 4), mesh),
      createSubAxis("x", 8, 2));

  // "x":(1)4 and "x":(2)4
  EXPECT_EQ(createSubAxis("x", 1, 4).getSuffixWithoutOverlap(
                createSubAxis("x", 2, 4), mesh),
            std::nullopt);
  EXPECT_EQ(createSubAxis("x", 2, 4).getSuffixWithoutOverlap(
                createSubAxis("x", 1, 4), mesh),
            createSubAxis("x", 4, 2));

  // "x"(4)2 and "x":(2)8
  EXPECT_EQ(createSubAxis("x", 4, 2).getSuffixWithoutOverlap(
                createSubAxis("x", 2, 8), mesh),
            std::nullopt);
  EXPECT_EQ(createSubAxis("x", 2, 8).getSuffixWithoutOverlap(
                createSubAxis("x", 4, 2), mesh),
            createSubAxis("x", 8, 2));

  auto checkNoOverlap = [&](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_EQ(a.getSuffixWithoutOverlap(b, mesh), a);
    EXPECT_EQ(b.getSuffixWithoutOverlap(a, mesh), b);
  };
  checkNoOverlap(createAxis("x"), createAxis("y"));
  checkNoOverlap(createAxis("x"), createSubAxis("y", 1, 2));
  checkNoOverlap(createSubAxis("x", 1, 4), createSubAxis("x", 4, 2));
  checkNoOverlap(createSubAxis("x", 1, 2), createSubAxis("x", 4, 2));

  auto checkCannotCoexist = [&](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_EQ(a.getSuffixWithoutOverlap(b, mesh), std::nullopt);
    EXPECT_EQ(b.getSuffixWithoutOverlap(a, mesh), std::nullopt);
  };
  checkCannotCoexist(createSubAxis("x", 1, 3), createSubAxis("x", 2, 3));
  checkCannotCoexist(createSubAxis("x", 3, 2), createSubAxis("x", 1, 2));
}

TEST_F(DialectTest, AxisRefAttrCanMerge) {
  auto checkCanMerge = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_TRUE(a.canMerge(b));
    EXPECT_FALSE(b.canMerge(a));
  };
  checkCanMerge(createSubAxis("x", 1, 4), createSubAxis("x", 4, 2));
  checkCanMerge(createSubAxis("x", 2, 2), createSubAxis("x", 4, 4));

  EXPECT_FALSE(createAxis("x").canMerge(createAxis("x")));
  EXPECT_FALSE(createSubAxis("x", 1, 2).canMerge(createSubAxis("x", 1, 2)));
  EXPECT_FALSE(createAxis("x").canMerge(createAxis("y")));
  EXPECT_FALSE(createSubAxis("x", 1, 2).canMerge(createSubAxis("y", 2, 2)));
  EXPECT_FALSE(createAxis("x").canMerge(createSubAxis("x", 4, 2)));
  EXPECT_FALSE(createSubAxis("x", 1, 2).canMerge(createSubAxis("x", 4, 2)));
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

TEST_F(DialectTest, AxisRefAttrGetGreatestCommonPrefix) {
  auto isNotPrefix = [](AxisRefAttr a, AxisRefAttr b) {
    EXPECT_EQ(a.getGreatestCommonPrefix(b), std::nullopt);
    EXPECT_EQ(b.getGreatestCommonPrefix(a), std::nullopt);
  };
  isNotPrefix(createAxis("x"), createAxis("y"));
  isNotPrefix(createSubAxis("x", 1, 2), createSubAxis("x", 2, 4));
  isNotPrefix(createSubAxis("x", 1, 2), createSubAxis("y", 1, 2));

  auto equals = [](AxisRefAttr a) {
    EXPECT_EQ(a.getGreatestCommonPrefix(a), a);
  };
  equals(createAxis("x"));
  equals(createSubAxis("x", 2, 4));

  auto prefix = [](AxisRefAttr small, AxisRefAttr large) {
    EXPECT_EQ(small.getGreatestCommonPrefix(large), small);
    EXPECT_EQ(large.getGreatestCommonPrefix(small), small);
  };
  prefix(createSubAxis("x", 1, 4), createAxis("x"));
  prefix(createSubAxis("x", 1, 2), createSubAxis("x", 1, 4));
  prefix(createSubAxis("x", 2, 4), createSubAxis("x", 2, 8));
}

TEST_F(DialectTest, AxisRefAttrGetFirstOverlapping) {
  SmallVector<AxisRefAttr> orderedAxes = {
      createAxis("x"), createSubAxis("y", 1, 2), createSubAxis("y", 4, 4),
      createSubAxis("y", 64, 2), createSubAxis("z", 2, 2)};

  // No overlapping.
  EXPECT_EQ(createAxis("a").getFirstOverlapping(orderedAxes),
            orderedAxes.end());
  EXPECT_EQ(createSubAxis("a", 1, 2).getFirstOverlapping(orderedAxes),
            orderedAxes.end());
  EXPECT_EQ(createSubAxis("y", 2, 2).getFirstOverlapping(orderedAxes),
            orderedAxes.end());
  EXPECT_EQ(createSubAxis("y", 16, 4).getFirstOverlapping(orderedAxes),
            orderedAxes.end());

  // First overlapping "x"
  EXPECT_EQ(createAxis("x").getFirstOverlapping(orderedAxes),
            orderedAxes.begin());
  EXPECT_EQ(createSubAxis("x", 1, 2).getFirstOverlapping(orderedAxes),
            orderedAxes.begin());

  // First overlapping "y":(1)2
  EXPECT_EQ(createAxis("y").getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 1);
  EXPECT_EQ(createSubAxis("y", 1, 2).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 1);
  EXPECT_EQ(createSubAxis("y", 1, 4).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 1);
  EXPECT_EQ(createSubAxis("y", 1, 32).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 1);

  // First overlapping "y":(4)4
  EXPECT_EQ(createSubAxis("y", 2, 4).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 2);
  EXPECT_EQ(createSubAxis("y", 2, 16).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 2);
  EXPECT_EQ(createSubAxis("y", 4, 2).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 2);
  EXPECT_EQ(createSubAxis("y", 8, 2).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 2);

  // First overlapping "y":(64)2
  EXPECT_EQ(createSubAxis("y", 32, 4).getFirstOverlapping(orderedAxes),
            orderedAxes.begin() + 3);
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

  MeshAttr mesh =
      createMesh({{"u", 1}, {"v", 1}, {"w", 1}, {"x", 1}, {"y", 1}, {"z", 1}});

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

TEST_F(DialectTest, TensorShardingAttrGetLocalTypeUnranked) {
  OpBuilder builder(&context);
  auto mesh = MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 2),
                                       MeshAxisAttr::get(&context, "y", 3),
                                       MeshAxisAttr::get(&context, "z", 4)});

  TensorShardingAttr shardingWithAxes =
      createTensorSharding({createDimSharding({createAxis("x")}),
                            createDimSharding({createAxis("y")})});

  Type i32Type = builder.getI32Type();
  EXPECT_EQ(shardingWithAxes.getLocalType(i32Type, mesh), i32Type);

  Type f32Type = builder.getF32Type();
  EXPECT_EQ(shardingWithAxes.getLocalType(f32Type, mesh), f32Type);

  Type indexType = builder.getIndexType();
  EXPECT_EQ(shardingWithAxes.getLocalType(indexType, mesh), indexType);

  Type unrankedType = UnrankedTensorType::get(i32Type);
  EXPECT_EQ(shardingWithAxes.getLocalType(unrankedType, mesh), unrankedType);

  Type noneType = builder.getNoneType();
  EXPECT_EQ(shardingWithAxes.getLocalType(noneType, mesh), noneType);

  Type complexType = ComplexType::get(f32Type);
  EXPECT_EQ(shardingWithAxes.getLocalType(complexType, mesh), complexType);

  Type tupleType = TupleType::get(&context, {i32Type, f32Type});
  EXPECT_EQ(shardingWithAxes.getLocalType(tupleType, mesh), tupleType);
}

TEST_F(DialectTest, TensorShardingAttrGetLocalTypeVector) {
  OpBuilder builder(&context);
  auto globalType = VectorType::get({8, 4}, builder.getI32Type());
  auto mesh = MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 2),
                                       MeshAxisAttr::get(&context, "y", 3),
                                       MeshAxisAttr::get(&context, "z", 4)});

  TensorShardingAttr sharding1 = createTensorSharding({});
  EXPECT_EQ(sharding1.getLocalType(globalType, mesh), globalType);

  TensorShardingAttr sharding2 = createTensorSharding(
      {createDimSharding({createAxis("x")}), createDimSharding({})});
  EXPECT_EQ(sharding2.getLocalType(globalType, mesh),
            VectorType::get({4, 4}, builder.getI32Type()));

  TensorShardingAttr sharding3 = createTensorSharding(
      {createDimSharding({createAxis("y")}), createDimSharding({})});
  EXPECT_EQ(sharding3.getLocalType(globalType, mesh),
            VectorType::get({3, 4}, builder.getI32Type()));

  TensorShardingAttr sharding4 = createTensorSharding(
      {createDimSharding({createAxis("x"), createAxis("z")}),
       createDimSharding({createAxis("y")})});
  EXPECT_EQ(sharding4.getLocalType(globalType, mesh),
            VectorType::get({1, 2}, builder.getI32Type()));
}

TEST_F(DialectTest, TensorShardingAttrGetLocalTensorType) {
  OpBuilder builder(&context);
  auto globalType = RankedTensorType::get({8, 4}, builder.getI32Type());
  auto mesh = MeshAttr::get(&context, {MeshAxisAttr::get(&context, "x", 2),
                                       MeshAxisAttr::get(&context, "y", 3),
                                       MeshAxisAttr::get(&context, "z", 4)});

  TensorShardingAttr sharding1 = createTensorSharding({});
  EXPECT_EQ(sharding1.getLocalTensorType(globalType, mesh), globalType);

  TensorShardingAttr sharding2 = createTensorSharding(
      {createDimSharding({createAxis("x")}), createDimSharding({})});
  EXPECT_EQ(sharding2.getLocalTensorType(globalType, mesh),
            RankedTensorType::get({4, 4}, builder.getI32Type()));

  TensorShardingAttr sharding3 = createTensorSharding(
      {createDimSharding({createAxis("y")}), createDimSharding({})});
  EXPECT_EQ(sharding3.getLocalTensorType(globalType, mesh),
            RankedTensorType::get({3, 4}, builder.getI32Type()));

  TensorShardingAttr sharding4 = createTensorSharding(
      {createDimSharding({createAxis("x"), createAxis("z")}),
       createDimSharding({createAxis("y")})});
  EXPECT_EQ(sharding4.getLocalTensorType(globalType, mesh),
            RankedTensorType::get({1, 2}, builder.getI32Type()));
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

TEST_F(DialectTest, TensorMappingAttrContainsFactor) {
  TensorMappingAttr tensorMapping =
      createTensorMapping({createDimMapping({0, 3}), createDimMapping({2})});

  EXPECT_TRUE(tensorMapping.containsFactor(0));
  EXPECT_TRUE(tensorMapping.containsFactor(2));
  EXPECT_TRUE(tensorMapping.containsFactor(3));

  EXPECT_FALSE(tensorMapping.containsFactor(-1));
  EXPECT_FALSE(tensorMapping.containsFactor(1));
  EXPECT_FALSE(tensorMapping.containsFactor(4));
}

TEST_F(DialectTest, OpShardingRuleAttrElementWiseOperation) {
  // An element-wise operation with two operands, e.g., a + b where operand has
  // shape 8x16x32.
  TensorMappingAttr tensorMapping = createTensorMapping(
      {createDimMapping({0}), createDimMapping({1}), createDimMapping({2})});
  OpShardingRuleAttr rule = createOpShardingRule(
      /*factorSizes=*/{8, 16, 32},
      /*operandMappings=*/{tensorMapping, tensorMapping},
      /*resultMappings=*/{tensorMapping});

  EXPECT_EQ(rule.getNumFactors(), 3);
  EXPECT_EQ(rule.getNumOperands(), 2);
  EXPECT_EQ(rule.getNumResults(), 1);

  auto verifyBatchingFactor = [&](int64_t factorIndex) {
    EXPECT_FALSE(rule.isReductionFactor(factorIndex));
    EXPECT_FALSE(rule.isNeedReplicationFactor(factorIndex));
    EXPECT_FALSE(rule.isPermutationFactor(factorIndex));
    EXPECT_TRUE(rule.isFactorInAllNonScalarTensors(factorIndex));
    EXPECT_TRUE(rule.isBatchingFactor(factorIndex));
  };
  verifyBatchingFactor(0);
  verifyBatchingFactor(1);
  verifyBatchingFactor(2);
}

TEST_F(DialectTest, OpShardingRuleAttrDotGeneralOperation) {
  // An einsum with sharding rule ([i, j, l], [i, l, k])->([i, j, k]), where l
  // is the contracting dimension.
  TensorMappingAttr lhs = createTensorMapping(
      {createDimMapping({0}), createDimMapping({1}), createDimMapping({3})});
  TensorMappingAttr rhs = createTensorMapping(
      {createDimMapping({0}), createDimMapping({3}), createDimMapping({2})});
  TensorMappingAttr result = createTensorMapping(
      {createDimMapping({0}), createDimMapping({1}), createDimMapping({2})});
  OpShardingRuleAttr rule = createOpShardingRule(
      /*factorSizes=*/{8, 16, 32, 64}, /*operandMappings=*/{lhs, rhs},
      /*resultMappings=*/{result}, /*reductionFactors=*/{3});

  EXPECT_EQ(rule.getNumFactors(), 4);
  EXPECT_EQ(rule.getNumOperands(), 2);
  EXPECT_EQ(rule.getNumResults(), 1);

  // Verify the first factor is a batching factor.
  EXPECT_FALSE(rule.isReductionFactor(0));
  EXPECT_FALSE(rule.isNeedReplicationFactor(0));
  EXPECT_FALSE(rule.isPermutationFactor(0));
  EXPECT_TRUE(rule.isFactorInAllNonScalarTensors(0));
  EXPECT_TRUE(rule.isBatchingFactor(0));

  auto verifyNonContractingDimension = [&](int64_t factorIndex) {
    EXPECT_FALSE(rule.isReductionFactor(factorIndex));
    EXPECT_FALSE(rule.isNeedReplicationFactor(factorIndex));
    EXPECT_FALSE(rule.isPermutationFactor(factorIndex));
    EXPECT_FALSE(rule.isFactorInAllNonScalarTensors(factorIndex));
    EXPECT_FALSE(rule.isBatchingFactor(factorIndex));
  };
  verifyNonContractingDimension(1);
  verifyNonContractingDimension(2);

  // Verify the contracting dimension is a reduction factor.
  EXPECT_TRUE(rule.isReductionFactor(3));
  EXPECT_FALSE(rule.isNeedReplicationFactor(3));
  EXPECT_FALSE(rule.isPermutationFactor(3));
  EXPECT_FALSE(rule.isFactorInAllNonScalarTensors(3));
  EXPECT_FALSE(rule.isBatchingFactor(3));
}

TEST_F(DialectTest, OpShardingRuleAttrDynamicSlice) {
  // An dynamic_slice with sharding rule ([i, j, k], [], [], [])->([i, l, m]),
  // {i=32, j=1, k=1, l=1, m=1}.
  TensorMappingAttr operand = createTensorMapping(
      {createDimMapping({0}), createDimMapping({1}), createDimMapping({2})});
  TensorMappingAttr index = createTensorMapping({});
  TensorMappingAttr result = createTensorMapping(
      {createDimMapping({0}), createDimMapping({3}), createDimMapping({4})});
  OpShardingRuleAttr rule =
      createOpShardingRule(/*factorSizes=*/{8, 1, 1, 1, 1},
                           /*operandMappings=*/{operand, index, index, index},
                           /*resultMappings=*/{result});

  EXPECT_EQ(rule.getNumFactors(), 5);
  EXPECT_EQ(rule.getNumOperands(), 4);
  EXPECT_EQ(rule.getNumResults(), 1);

  // Verify the first factor is a batching factor.
  EXPECT_FALSE(rule.isReductionFactor(0));
  EXPECT_FALSE(rule.isNeedReplicationFactor(0));
  EXPECT_FALSE(rule.isPermutationFactor(0));
  EXPECT_TRUE(rule.isFactorInAllNonScalarTensors(0));
  EXPECT_TRUE(rule.isBatchingFactor(0));

  auto verifyNonBatchingFactor = [&](int64_t factorIndex) {
    EXPECT_FALSE(rule.isReductionFactor(factorIndex));
    EXPECT_FALSE(rule.isNeedReplicationFactor(factorIndex));
    EXPECT_FALSE(rule.isPermutationFactor(factorIndex));
    EXPECT_FALSE(rule.isFactorInAllNonScalarTensors(factorIndex));
    EXPECT_FALSE(rule.isBatchingFactor(factorIndex));
  };
  verifyNonBatchingFactor(1);
  verifyNonBatchingFactor(2);
}

TEST_F(DialectTest, MeshAttrEquals) {
  auto checkEquality = [](MeshAttr a, MeshAttr b,
                          bool ignoreDeviceIds = false) {
    EXPECT_TRUE(a.equals(b, ignoreDeviceIds));
    EXPECT_TRUE(b.equals(a, ignoreDeviceIds));
  };
  auto checkInequality = [](MeshAttr a, MeshAttr b,
                            bool ignoreDeviceIds = false) {
    EXPECT_FALSE(a.equals(b, ignoreDeviceIds));
    EXPECT_FALSE(b.equals(a, ignoreDeviceIds));
  };
  {
    MeshAttr mesh = createMesh({});
    checkEquality(mesh, mesh);
    // Do not ignore device order.
    checkInequality(mesh, createMesh({{"x", 4}}));
    checkInequality(mesh, createMesh({{"x", 1}}));
    checkInequality(mesh, createMesh({}, /*deviceIds=*/{0}));
    checkInequality(mesh, createMesh({}, /*deviceIds=*/{5}));
    // Ignore device order.
    checkInequality(mesh, createMesh({{"x", 4}}), true);
    checkInequality(mesh, createMesh({{"x", 1}}), true);
    checkInequality(mesh, createMesh({}, /*deviceIds=*/{0}), true);
    checkInequality(mesh, createMesh({}, /*deviceIds=*/{5}), true);
  }
  {
    MeshAttr mesh = createMesh({{"x", 3}, {"y", 2}});
    checkEquality(mesh, mesh);
    // Do not ignore device order.
    checkInequality(mesh, createMesh({{"x", 2}, {"y", 3}}));
    checkInequality(mesh, createMesh({{"x", 3}, {"z", 2}}));
    checkInequality(mesh, createMesh({{"y", 2}, {"x", 3}}));
    checkInequality(mesh, createMesh({{"x", 4}, {"y", 2}}));
    checkInequality(mesh, createMesh({{"x", 3}, {"y", 2}},
                                     /*deviceIds=*/{5, 4, 3, 2, 1, 0}));
    // Ignore device order.
    checkInequality(mesh, createMesh({{"x", 2}, {"y", 3}}), true);
    checkInequality(mesh, createMesh({{"x", 3}, {"z", 2}}), true);
    checkInequality(mesh, createMesh({{"y", 2}, {"x", 3}}), true);
    checkInequality(mesh, createMesh({{"x", 4}, {"y", 2}}), true);
    checkEquality(
        mesh,
        createMesh({{"x", 3}, {"y", 2}}, /*deviceIds=*/{5, 4, 3, 2, 1, 0}),
        true);
  }
  {
    MeshAttr mesh = createMesh({{"x", 4}}, /*deviceIds=*/{1, 0, 3, 2});
    checkInequality(mesh, createMesh({{"x", 4}}, /*deviceIds=*/{3, 2, 1, 0}));
    checkEquality(mesh, createMesh({{"x", 4}}, /*deviceIds=*/{3, 2, 1, 0}),
                  /*ignoreDeviceIds=*/true);
  }
  {
    MeshAttr mesh = createMesh({}, /*deviceIds=*/{3});
    checkInequality(mesh, createMesh({}, /*deviceIds=*/{5}));
    checkEquality(mesh, createMesh({}, /*deviceIds=*/{5}),
                  /*ignoreDeviceIds=*/true);
  }
  {
    MeshAttr mesh = createMesh({}, /*deviceIds=*/{0});
    checkInequality(mesh, createMesh({{"x", 1}}));
    checkInequality(mesh, createMesh({{"x", 1}}),
                    /*ignoreDeviceIds=*/true);
  }
}

}  // namespace

}  // namespace sdy
}  // namespace mlir
