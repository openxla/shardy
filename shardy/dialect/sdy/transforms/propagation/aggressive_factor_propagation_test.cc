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

#include "shardy/dialect/sdy/transforms/propagation/aggressive_factor_propagation.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/basic_factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/testing_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class GetCompatibleMajorShardingAxesTest : public PropagationTestBase {
 protected:
  AxesPerFactor getCompatibleMajorShardingAxesForAllFactors(
      ShardingProjection projection,
      const BasicFactorPropagation& factorPropagation, int64_t numFactors) {
    AxesPerFactor result =
        factorPropagation.getCompatibleMajorShardingAxesForAllFactors(
            projection, PropagationDirection::BOTH,
            SmallVector<int64_t>(numFactors, 1), /*mesh=*/nullptr,
            /*op=*/nullptr, /*conservativePropagation=*/false);
    EXPECT_EQ(result.size(), numFactors);
    return result;
  }

  bool basicAndAggressiveFactorPropagationSameResult(
      ShardingProjection projection, int64_t numFactors,
      PropagationDirection direction = PropagationDirection::BOTH,
      ArrayRef<int64_t> factorSizes = ArrayRef<int64_t>(),
      ArrayRef<std::pair<StringRef, int64_t>> meshAxes = {}) {
    const AxesPerFactor result1 = getCompatibleMajorShardingAxesForAllFactors(
        projection, BasicFactorPropagation(), numFactors);
    const AxesPerFactor result2 = getCompatibleMajorShardingAxesForAllFactors(
        projection, AggressiveFactorPropagation(), numFactors);
    return result1 == result2;
  }
};

TEST_F(GetCompatibleMajorShardingAxesTest, RealAndFakeConflicts) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
                   {2, {.axisRefs = {createAxis("b")}}},
                   {4, {.axisRefs = {createAxis("c")}}},
                   {6, {.axisRefs = {createAxis("e")}}},
                   {7, {.axisRefs = {}}},
                   {8, {.axisRefs = {createAxis("f")}}},
                   {10, {.axisRefs = {createAxis("g")}}},
               }},
          {.factorIndexToSharding =
               {
                   {1, {.axisRefs = {createAxis("a")}}},
                   {2, {.axisRefs = {}}},
                   {3, {.axisRefs = {createAxis("b")}}},
                   {4, {.axisRefs = {}}},
                   {5, {.axisRefs = {createAxis("c")}}},
                   {6, {.axisRefs = {}}},
                   {7, {.axisRefs = {createAxis("e")}}},
                   {9, {.axisRefs = {createAxis("g")}}},
                   {10, {.axisRefs = {createAxis("f")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
                   {1, {.axisRefs = {}}},
                   {2, {.axisRefs = {}, .isClosed = true}},
                   {3, {.axisRefs = {}}},
                   {4, {.axisRefs = {}, .overflowAxes = {createAxis("d")}}},
                   {5, {.axisRefs = {}}},
                   {6, {.axisRefs = {}, .isClosed = true}},
                   {7, {.axisRefs = {}}},
                   {8, {.axisRefs = {}}},
                   {9, {.axisRefs = {}}},
               }},
      });

  // Basic strategy does not propagate anything for this case.
  AxesPerFactor resultWithBasicStrategy =
      getCompatibleMajorShardingAxesForAllFactors(projection,
                                                  BasicFactorPropagation(), 11);
  for (ArrayRef<AxisRefAttr> element : resultWithBasicStrategy) {
    EXPECT_THAT(element, IsEmpty());
  }

  AxesPerFactor resultWithAggressiveStrategy =
      getCompatibleMajorShardingAxesForAllFactors(
          projection, AggressiveFactorPropagation(), 11);

  // Axis "a" is in factors 0 and 1, which co-exists in the result. We can
  // propagate "a" along factor 0 or 1 to the result. The real conflicts
  // prohibit further propagation along these two factors.
  EXPECT_THAT(resultWithAggressiveStrategy[0], IsEmpty());
  EXPECT_THAT(resultWithAggressiveStrategy[1], IsEmpty());

  // Axis "b" is in factors 2 and 3. Since we cannot propagate "b" along
  // factor 2 (the factor sharding is closed in the result), we can propagate
  // "b" along factor 3.
  EXPECT_THAT(resultWithAggressiveStrategy[2], IsEmpty());
  EXPECT_THAT(resultWithAggressiveStrategy[3], ElementsAre(AxisRefIs("b")));

  // Axis "c" is in factors 4 and 5. Since we cannot propagate "c" along
  // factor 4 (the factor sharding has overflow axes in the result), we can
  // propagate "c" along factor 5.
  EXPECT_THAT(resultWithAggressiveStrategy[4], IsEmpty());
  EXPECT_THAT(resultWithAggressiveStrategy[5], ElementsAre(AxisRefIs("c")));

  // Axis "e" is in factors 6 and 7. We cannot propagate "e" along factor 6
  // since the factor sharding is closed in the result. We cannot propagate
  // "e" along factor 7 since operand 0 contains factor 7 and is already
  // sharded along factor 7.
  EXPECT_THAT(resultWithAggressiveStrategy[6], IsEmpty());
  EXPECT_THAT(resultWithAggressiveStrategy[7], IsEmpty());

  // Factor 10 already contains axes "f" and "g". Factor does not appear in
  // the result. Hence, we can propagate "f" and "g" to result along factor 8
  // and 9, respectively.
  EXPECT_THAT(resultWithAggressiveStrategy[8], ElementsAre(AxisRefIs("f")));
  EXPECT_THAT(resultWithAggressiveStrategy[9], ElementsAre(AxisRefIs("g")));
  EXPECT_THAT(resultWithAggressiveStrategy[10], IsEmpty());
}

TEST_F(GetCompatibleMajorShardingAxesTest, TwoFactorsDoNotCoExistInAnyTensor) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
               }},
          {.factorIndexToSharding =
               {
                   {1, {.axisRefs = {createAxis("a")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
               }},
          {.factorIndexToSharding =
               {
                   {1, {.axisRefs = {}}},
               }},
      });

  // Basic strategy does not propagate anything for this case.
  AxesPerFactor resultWithBasicStrategy =
      getCompatibleMajorShardingAxesForAllFactors(projection,
                                                  BasicFactorPropagation(), 2);
  for (ArrayRef<AxisRefAttr> element : resultWithBasicStrategy) {
    EXPECT_THAT(element, IsEmpty());
  }

  // Factors 0 and 1 do not co-exist in any tensor. Hence, we can propagate
  // axis "a" along both factors.
  AxesPerFactor resultWithAggressiveStrategy =
      getCompatibleMajorShardingAxesForAllFactors(
          projection, AggressiveFactorPropagation(), 2);
  EXPECT_THAT(resultWithAggressiveStrategy[0], ElementsAre(AxisRefIs("a")));
  EXPECT_THAT(resultWithAggressiveStrategy[1], ElementsAre(AxisRefIs("a")));
}

TEST_F(GetCompatibleMajorShardingAxesTest,
       ConflictsBetweenDifferentFactorsAndReplicated) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
                   {3,
                    {.axisRefs = {createSubAxis("h", 1, 2), createAxis("i")}}},
               }},
          {.factorIndexToSharding =
               {
                   {1, {.axisRefs = {createAxis("b")}}},
                   {2, {.axisRefs = {createAxis("f")}, .isClosed = true}},
               },
           .replicatedAxes = {createSubAxis("h", 1, 2)}},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
                   {4,
                    {.axisRefs = {createSubAxis("j", 1, 8),
                                  createSubAxis("k", 2, 4)}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c"), createAxis("d")}}},
                   {3, {.axisRefs = {}}},
               },
           .replicatedAxes = {createSubAxis("h", 2, 4),
                              createSubAxis("i", 2, 2)}},
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c"), createAxis("e")}}},
                   {2,
                    {.axisRefs = {createAxis("f"), createAxis("g")},
                     .isClosed = true}},
               }},
          {.factorIndexToSharding =
               {
                   {4, {.axisRefs = {createSubAxis("j", 1, 4)}}},
                   {5, {.axisRefs = {createSubAxis("k", 1, 4)}}},
               }},
      });

  int64_t numFactors = 6;
  AxesPerFactor resultWithBasicStrategy =
      getCompatibleMajorShardingAxesForAllFactors(
          projection, BasicFactorPropagation(), numFactors);
  AxesPerFactor resultWithAggressiveStrategy =
      getCompatibleMajorShardingAxesForAllFactors(
          projection, AggressiveFactorPropagation(), numFactors);

  // The compatible major axes for factor 0 are ["a", "b", "c"], but the 2nd
  // operand, which isn't mapped to factor 0, has a different factor (1) that
  // is sharded along "b".
  EXPECT_THAT(resultWithBasicStrategy[0], ElementsAre(AxisRefIs("a")));
  // Since factors 0 and 1 does not co-exist in the same operand, we can
  // ignore their conflicts.
  EXPECT_THAT(resultWithAggressiveStrategy[0],
              ElementsAre(AxisRefIs("a"), AxisRefIs("b"), AxisRefIs("c")));

  // Axis "b" appears in factors 0 and 1, which does not co-exist in the same
  // operand. Hence, we can propagate axis "b" along factors 0 and 1,
  // respectively. `getAxesWithConservativeStrategy` treat it as a conflict,
  // while `getAxesWithAggressiveStrategy` ignore
  // this fake conflict.
  EXPECT_THAT(resultWithBasicStrategy[1], IsEmpty());
  EXPECT_THAT(resultWithAggressiveStrategy[1], ElementsAre(AxisRefIs("b")));

  // For other factors, the results are the same.
  for (int64_t i = 2; i < numFactors; i++) {
    EXPECT_TRUE(resultWithBasicStrategy[i] == resultWithAggressiveStrategy[i]);
  }
}

TEST_F(GetCompatibleMajorShardingAxesTest, FullAxesConflictsOnlyForSameFactor) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
                   {3, {.axisRefs = {createAxis("h")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
                   {2, {.axisRefs = {createAxis("g")}}},
                   {3, {.axisRefs = {createAxis("i")}}},
               }},
          {.factorIndexToSharding = {{1, {.axisRefs = {createAxis("e")}}}}},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
                   {1, {.axisRefs = {createAxis("e"), createAxis("f")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("d")}}},
                   {2, {.axisRefs = {}}},
                   {3, {.axisRefs = {createAxis("i"), createAxis("j")}}},
               }},
      });
  // Two strategies have the same criterion on the conflicts within a single
  // factor.
  EXPECT_TRUE(basicAndAggressiveFactorPropagationSameResult(projection, 4));
}

TEST_F(GetCompatibleMajorShardingAxesTest, SubAxesConflictsOnlyForSameFactor) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {1,
                    {.axisRefs = {createSubAxis("c", 2, 2), createAxis("d")}}},
                   {4, {.axisRefs = {createSubAxis("i", 1, 4)}}},
                   {5, {.axisRefs = {createSubAxis("k", 1, 8)}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
                   {1, {.axisRefs = {createSubAxis("c", 2, 4)}}},
                   {3,
                    {.axisRefs = {createAxis("g"), createSubAxis("h", 1, 8)}}},
                   {5,
                    {.axisRefs = {createSubAxis("k", 1, 4)}, .isClosed = true}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createSubAxis("a", 1, 2)}}},
                   {2,
                    {.axisRefs = {createSubAxis("e", 2, 4),
                                  createSubAxis("f", 4, 2)}}},
                   {3,
                    {.axisRefs = {createAxis("g"), createSubAxis("h", 1, 4),
                                  createAxis("i")}}},
                   {5, {.axisRefs = {createSubAxis("k", 1, 2)}}},
               }},
          {.factorIndexToSharding =
               {
                   {1,
                    {.axisRefs = {createSubAxis("c", 2, 4), createAxis("d")}}},
                   {2,
                    {.axisRefs = {createSubAxis("e", 2, 4),
                                  createSubAxis("f", 4, 8)}}},
                   {4, {.axisRefs = {createSubAxis("j", 2, 4)}}},
                   {5, {.axisRefs = {}}},
               }},
      });
  // Two strategies have the same criterion on the conflicts within a single
  // factor.
  EXPECT_TRUE(basicAndAggressiveFactorPropagationSameResult(projection, 6));
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
