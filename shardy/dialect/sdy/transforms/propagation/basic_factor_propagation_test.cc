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

#include "shardy/dialect/sdy/transforms/propagation/basic_factor_propagation.h"

#include <cassert>
#include <cstdint>
#include <utility>

#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/testing_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

//===----------------------------------------------------------------------===//
// Tests for BasicFactorPropagation::getCompatibleMajorShardingAxes
//===----------------------------------------------------------------------===//

class GetCompatibleMajorShardingAxesTest : public PropagationTestBase {
 protected:
  SmallVector<AxisRefAttr> getCompatibleMajorShardingAxes(
      ShardingProjection projection, int64_t factorIndex,
      PropagationDirection direction = PropagationDirection::BOTH,
      int64_t factorSize = 1,
      ArrayRef<std::pair<StringRef, int64_t>> meshAxes = {},
      bool conservativePropagation = false) {
    MeshAttr mesh = nullptr;
    if (!meshAxes.empty()) {
      SmallVector<MeshAxisAttr> meshAxisAttrs;
      meshAxisAttrs.reserve(meshAxes.size());
      for (auto [axisName, size] : meshAxes) {
        meshAxisAttrs.push_back(MeshAxisAttr::get(&context, axisName, size));
      }
      mesh = MeshAttr::get(&context, meshAxisAttrs);
    }
    return BasicFactorPropagation().getCompatibleMajorShardingAxes(
        projection, factorIndex, direction, factorSize, mesh, /*op=*/nullptr,
        conservativePropagation);
  }
};

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

  // Conflict between ["a", "b", "c"] and ["a", "b", "d"], no conflict with
  // ["a"] and [].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0),
              ElementsAre(AxisRefIs("a"), AxisRefIs("b")));

  // No conflict between ["e"] and ["e", "f"].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 1),
              ElementsAre(AxisRefIs("e"), AxisRefIs("f")));

  // No conflict between ["g"] and [].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 2),
              ElementsAre(AxisRefIs("g")));

  // Conflict between ["h"], ["i"], and ["i", "j"].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 3), IsEmpty());

  // No tensors are mapped to factor 4.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 4), IsEmpty());
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

  // No conflict between ["a", "b"], [], and ["a":(1)2].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0),
              ElementsAre(AxisRefIs("a"), AxisRefIs("b")));

  // Conflict between ["c":(2)2, "d"] and ["c":(2)4].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 1),
              ElementsAre(SubAxisRefIs("c", 2, 2)));

  // No conflict between ["e":(2)4, "f":(4)2] and ["e":(2)4, "f":(4)8"].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 2),
              ElementsAre(SubAxisRefIs("e", 2, 4), SubAxisRefIs("f", 4, 8)));

  // Conflict between ["g", "h":(1)8] and ["g", "h":(1)4, "i"].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 3),
              ElementsAre(AxisRefIs("g"), SubAxisRefIs("h", 1, 4)));

  // Conflict between ["j":(1)4] and ["j":(2)4].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 4), IsEmpty());

  // The compatible axes of ["k":(1)8], ["k":(1)4], ["k":(1)2], and [] are
  // ["k":(1)8]. Since ["k":(1)4] is closed, the final compatible axes to
  // ["k":(1)4].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 5),
              ElementsAre(SubAxisRefIs("k", 1, 4)));
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

  // The compatible major axes for factor 0 are ["a", "b", "c"], but the 2nd
  // operand, which isn't mapped to factor 0, has a different factor (1) that
  // is sharded along "b".
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0),
              ElementsAre(AxisRefIs("a")));

  // The compatible major axes for factor 2 are ["f", "g"], but factor 2 is
  // closed for the 2nd operand (with partial sharding ["f"]).
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 2),
              ElementsAre(AxisRefIs("f")));

  // The compatible major axes for factor 3 are ["h":(1)2, "i"], but the 1st
  // result (with partial sharding []) is replicated along "i":(2)2.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 3),
              ElementsAre(SubAxisRefIs("h", 1, 2), SubAxisRefIs("i", 1, 2)));

  // The compatible major axes for factor 4 are ["j":(1)8, "k":(2)4], but the
  // 3rd result (with partial sharding ["j":(1)4]) has a different factor (5)
  // that is sharded along a sub-axis of "k":(2)4.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 4),
              ElementsAre(SubAxisRefIs("j", 1, 8)));
}

TEST_F(GetCompatibleMajorShardingAxesTest, ConflictsWithOverflowAxes) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {1, {.axisRefs = {createAxis("c")}}},
               }},
      },
      /*results=*/{
          {
              .factorIndexToSharding =
                  {
                      {0, {.axisRefs = {}}},
                      {1,
                       {.axisRefs = {},
                        .overflowAxes = {createSubAxis("b", 1, 2)}}},
                  },
          },
      });

  // The compatible major axes for factor 0 are ["a", "b"], but the result has
  // "b" in the overflow axes of factor 1.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0),
              ElementsAre(AxisRefIs("a")));

  // The compatible major axes for factor 1 are ["c"], but factor 1 has overflow
  // axes (in which case it doesn't matter that it's open) for the result (with
  // partial sharding []).
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 1), IsEmpty());
}

TEST_F(GetCompatibleMajorShardingAxesTest, MinorMostFactorNotDivisible) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b")},
                     .isMinorMost = true}},
                   {1, {.axisRefs = {createAxis("c")}, .isMinorMost = true}},
               }},
      },
      /*results=*/{
          {
              .factorIndexToSharding =
                  {
                      {0, {.axisRefs = {}, .isMinorMost = false}},
                      {1, {.axisRefs = {}, .isMinorMost = true}},
                  },
          },
      });
  SmallVector<std::pair<StringRef, int64_t>> meshAxes = {
      {"a", 3}, {"b", 3}, {"c", 3}};

  // The compatible major axes for factor 0 are ["a", "b"]. This factor is
  // non-minor-most in the result (with partial sharding []).

  // The factor size (9) is divisible by the size of ["a", "b"] (9).
  EXPECT_THAT(
      getCompatibleMajorShardingAxes(projection, 0, PropagationDirection::BOTH,
                                     /*factorSize=*/9, meshAxes),
      ElementsAre(AxisRefIs("a"), AxisRefIs("b")));

  // The factor size (6) is divisible by the size of ["a"] (3), but not by the
  // size of ["a", "b"] (9).
  EXPECT_THAT(
      getCompatibleMajorShardingAxes(projection, 0, PropagationDirection::BOTH,
                                     /*factorSize=*/6, meshAxes),
      ElementsAre(AxisRefIs("a")));

  // The factor size (4) isn't divisible by the size of ["a"] (3).
  EXPECT_THAT(
      getCompatibleMajorShardingAxes(projection, 0, PropagationDirection::BOTH,
                                     /*factorSize=*/4, meshAxes),
      IsEmpty());

  // The compatible major axes for factor 1 are ["c"]. This factor is minor-most
  // in the result (with partial sharding []), therefore it's fine that the
  // factor size (4) isn't divisible by the size of ["c"] (3).
  EXPECT_THAT(
      getCompatibleMajorShardingAxes(projection, 1, PropagationDirection::BOTH,
                                     /*factorSize=*/4, meshAxes),
      ElementsAre(AxisRefIs("c")));
}

TEST_F(GetCompatibleMajorShardingAxesTest, BackwardPropagationDirection) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {1, {.axisRefs = {createAxis("d"), createAxis("e")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
                   {1, {.axisRefs = {createAxis("d")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
                   {1, {.axisRefs = {createAxis("d")}}},
               }},
      });

  // Since we are only propagating backwards, we can expand both operands to
  // have [a, b, c].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0,
                                             PropagationDirection::BACKWARD),
              ElementsAre(AxisRefIs("a"), AxisRefIs("b"), AxisRefIs("c")));

  // Since we are only propagating backwards, we do not push [e] forwards.
  // NOTE: we do not propagate sideways to each operand as our current behavior
  // with BACKWARD closes all operands for factor expansion.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 1,
                                             PropagationDirection::BACKWARD),
              ElementsAre(AxisRefIs("d")));
}

TEST_F(GetCompatibleMajorShardingAxesTest, ForwardPropagationDirection) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
                   {1, {.axisRefs = {createAxis("d")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {1, {.axisRefs = {createAxis("d")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
                   {1, {.axisRefs = {createAxis("d"), createAxis("e")}}},
               }},
      });

  // Since we are only propagating forwards, we can expand both results to
  // have [a, b, c].
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0,
                                             PropagationDirection::FORWARD),
              ElementsAre(AxisRefIs("a"), AxisRefIs("b"), AxisRefIs("c")));

  // Since we are only propagating forwards, we do not push [e] backwards.
  // NOTE: we do not propagate sideways to each result as our current behavior
  // with FORWARD closes all results for factor expansion.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 1,
                                             PropagationDirection::FORWARD),
              ElementsAre(AxisRefIs("d")));
}

TEST_F(GetCompatibleMajorShardingAxesTest,
       BackwardPropagationDirectionConflict) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("z"), createAxis("a"),
                                  createAxis("b")}}},
               }},
      });

  // Even though we are propagating backwards, we still need to account for
  // conflicts in the results. The "z" blocks any sharding.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0,
                                             PropagationDirection::BACKWARD),
              IsEmpty());
}

TEST_F(GetCompatibleMajorShardingAxesTest,
       ForwardPropagationDirectionConflict) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("z"), createAxis("a"),
                                  createAxis("b")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
               }},
      });

  // Even though we are propagating forward, we still need to account for
  // conflicts in the operands. The "z" blocks any sharding.
  EXPECT_THAT(getCompatibleMajorShardingAxes(projection, 0,
                                             PropagationDirection::FORWARD),
              IsEmpty());
}

TEST_F(GetCompatibleMajorShardingAxesTest, NonePropagationDirection) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
               }},
      });

  // Even though [a, b, c] is the most compatible, since we aren't propagating,
  // we don't update any operands or results.
  EXPECT_THAT(
      getCompatibleMajorShardingAxes(projection, 0, PropagationDirection::NONE),
      IsEmpty());
}

TEST_F(GetCompatibleMajorShardingAxesTest,
       ConservativePropagationStopsSplitAxes) {
  ShardingProjection projection(
      /*operands=*/
      {{.factorIndexToSharding =
            {
                {0,
                 {.axisRefs = {createAxis("a"), createSubAxis("b", 1, 2),
                               createAxis("c")}}},
                {1, {.axisRefs = {createSubAxis("b", 2, 4)}}},
            }}},
      /*results=*/{{.factorIndexToSharding = {{0, {}}, {1, {}}}}});

  // In conservative mode, sub-axes are not propagated. So only a full "a"
  // will be propagated to the result, with the operand not being updated.
  EXPECT_THAT(
      getCompatibleMajorShardingAxes(
          projection, 0, /*direction=*/PropagationDirection::BOTH,
          /*factorSize=*/1, /*meshAxes=*/{}, /*conservativePropagation=*/true),
      ElementsAre(AxisRefIs("a")));
  EXPECT_THAT(
      getCompatibleMajorShardingAxes(
          projection, 1, /*direction=*/PropagationDirection::BOTH,
          /*factorSize=*/1, /*meshAxes=*/{}, /*conservativePropagation=*/true),
      IsEmpty());
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
