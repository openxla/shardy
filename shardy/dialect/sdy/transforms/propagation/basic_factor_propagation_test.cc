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
#include "shardy/dialect/sdy/transforms/propagation/factor_propagation.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include "shardy/dialect/sdy/transforms/propagation/testing_utils.h"
#include "shardy/dialect/sdy/transforms/propagation/utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

PropagationDirectionAlongFactor propagateAnything() {
  return [](int64_t) { return PropagationDirection::BOTH; };
}

class BasicFactorPropagationTest : public PropagationTestBase {
 protected:
  UpdateTensorShardings propagateFactorShardings(
      ShardingProjection& projection, int64_t numFactors,
      PropagationDirectionAlongFactor directionAlongFactor =
          propagateAnything(),
      MeshAttr mesh = nullptr, bool conservativePropagation = false) {
    return BasicFactorPropagation().propagateFactorShardings(
        projection, directionAlongFactor, SmallVector<int64_t>(numFactors, 1),
        mesh, /*op=*/nullptr, conservativePropagation);
  }

  UpdateTensorShardings propagateFactorShardings(
      ShardingProjection& projection, ArrayRef<int64_t> factorSizes,
      PropagationDirectionAlongFactor directionAlongFactor =
          propagateAnything(),
      MeshAttr mesh = nullptr) {
    return BasicFactorPropagation().propagateFactorShardings(
        projection, directionAlongFactor, factorSizes, mesh, /*op=*/nullptr,
        /*conservativePropagation=*/false);
  }
};

// NOLINTBEGIN(clang-diagnostic-pre-c++20-compat-pedantic)

TEST_F(BasicFactorPropagationTest, FullAxesConflictsOnlyForSameFactor) {
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

  // Propagate the following axes along each factor to all tensors.
  // * Factor 0: ["a", "b"] is the compatible major sharding axes of ["a"], [],
  //   ["a", "b", "c"], and ["a", "b", "d"].
  // * Factor 1: ["e", "f"] is the compatible major sharding axes of ["e"] and
  //   ["e", "f"].
  // * Factor 2: ["g"] is the compatible major sharding axes of ["g"] and [].
  // * Factor 3: [] is the compatible major sharding axes of ["h"], ["i"], and
  //   ["i", "j"].

  ShardingProjection projectionExpected(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {3, {.axisRefs = {createAxis("h")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {2, {.axisRefs = {createAxis("g")}}},
                   {3, {.axisRefs = {createAxis("i")}}},
               }},
          {.factorIndexToSharding =
               {{1, {.axisRefs = {createAxis("e"), createAxis("f")}}}}},
      },
      /*results=*/{
          projection.getResult(0),
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("d")}}},
                   {2, {.axisRefs = {createAxis("g")}}},
                   {3, {.axisRefs = {createAxis("i"), createAxis("j")}}},
               }},
      });

  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 4);
  EXPECT_THAT(toSetBitsVector(updateOperands), ElementsAre(0, 1, 2));
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(1));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(BasicFactorPropagationTest, SubAxesConflictsOnlyForSameFactor) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {1,
                    {.axisRefs = {createSubAxis("c", 2, 2), createAxis("d")}}},
                   {4, {.axisRefs = {createSubAxis("j", 1, 4)}}},
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

  // Propagate the following axes along each factor to all tensors.
  // * Factor 0: ["a", "b"] is the compatible major sharding axes of ["a", "b"],
  //   [], and ["a":(1)2].
  // * Factor 1: ["c":(2)2] is the compatible major sharding axes of ["c":(2)2,
  //   "d"], ["c":(2)4], and ["c":(2)4, "d"].
  // * Factor 2: ["e":(2)4, "f":(4)8"] is the compatible major sharding axes of
  //   ["e":(2)4, "f":(4)2] and ["e":(2)4, "f":(4)8"].
  // * Factor 3: ["g", "h":(1)4] is the compatible major sharding axes of ["g",
  //   "h":(1)8] and ["g", "h":(1)4, "i"].
  // * Factor 4: [] is the compatible major sharding axes of ["j":(1)4] and
  //   ["j":(2)4].
  // * Factor 5: The compatible axes of ["k":(1)8], ["k":(1)4], ["k":(1)2], and
  //   [] are ["k":(1)8]. Since ["k":(1)4] is closed, the final axes to be
  //   propagated are ["k":(1)4].

  ShardingProjection projectionExpected(
      /*operands=*/
      {
          projection.getOperand(0),
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
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
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {2,
                    {.axisRefs = {createSubAxis("e", 2, 4),
                                  createSubAxis("f", 4, 8)}}},
                   {3,
                    {.axisRefs = {createAxis("g"), createSubAxis("h", 1, 4),
                                  createAxis("i")}}},
                   {5, {.axisRefs = {createSubAxis("k", 1, 4)}}},
               }},
          {.factorIndexToSharding =
               {
                   {1,
                    {.axisRefs = {createSubAxis("c", 2, 4), createAxis("d")}}},
                   {2,
                    {.axisRefs = {createSubAxis("e", 2, 4),
                                  createSubAxis("f", 4, 8)}}},
                   {4, {.axisRefs = {createSubAxis("j", 2, 4)}}},
                   {5, {.axisRefs = {createSubAxis("k", 1, 4)}}},
               }},
      });

  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 6);
  EXPECT_THAT(toSetBitsVector(updateOperands), ElementsAre(1));
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0, 1));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(BasicFactorPropagationTest,
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

  // Propagate the following axes along each factor to all tensors.
  // * Factor 0: The compatible major axes are ["a", "b", "c"]. However, "b" is
  //   also used by Factor 1. We only propagate ["a"] along Factor 0.
  // * Factor 1: The compatible major axes are ["b"]. However, "b" is also used
  //   by Factor 0. We propagate [] along Factor 1.
  // * Factor 2: The compatible major axes are ["f", "g"]. However, Factor 2 is
  //   closed for the 2nd operand (with partial sharding ["f"]). We propagate
  //   ["f"] along Factor 2.
  // * Factor 3: The compatible major axes are ["h":(1)2, "i"]. However, the 1st
  //   result is replicated along "i":(2)2. We propagate ["h":(1)2, "i":(1)2]
  //   along Factor 3.
  // * Factor 4: The compatible major axes are ["j":(1)8, "k":(2)4]. However,
  //   "k"(1)4 is also used by Factor 5. We propagate ["j":(1)8] along Factor 4.
  // * Factor 5: The compatible major axes are ["k":(1)4]. However, "k":(2)4 is
  //   also used by Factor 4. We propagate ["k":(1)2] along Factor 5.

  ShardingProjection projectionExpected(
      /*operands=*/
      {
          projection.getOperand(0),
          projection.getOperand(1),
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
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
                   {3,
                    {.axisRefs = {createSubAxis("h", 1, 2),
                                  createSubAxis("i", 1, 2)}}},
               },
           .replicatedAxes = {createSubAxis("h", 2, 4),
                              createSubAxis("i", 2, 2)}},
          projection.getResult(1),
          {.factorIndexToSharding =
               {
                   {4, {.axisRefs = {createSubAxis("j", 1, 8)}}},
                   {5, {.axisRefs = {createSubAxis("k", 1, 4)}}},
               }},
      });

  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 6);
  EXPECT_THAT(toSetBitsVector(updateOperands), ElementsAre(2));
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0, 2));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(BasicFactorPropagationTest, ConflictsWithOverflowAxes) {
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
  //
  // The compatible major axes for factor 1 are ["c"], but factor 1 has overflow
  // axes (in which case it doesn't matter that it's open) for the result (with
  // partial sharding []).
  //
  // Finally, we propagate ["a"] and [] along factors 0 and 1, respectively.
  ShardingProjection projectionExpected(
      /*operands=*/{projection.getOperand(0)},
      /*results=*/{
          {
              .factorIndexToSharding =
                  {
                      {0, {.axisRefs = {createAxis("a")}}},
                      {1,
                       {.axisRefs = {},
                        .overflowAxes = {createSubAxis("b", 1, 2)}}},
                  },
          },
      });

  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 2);
  EXPECT_THAT(toSetBitsVector(updateOperands), IsEmpty());
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(BasicFactorPropagationTest, MinorMostFactorNotDivisible) {
  SmallVector<std::pair<StringRef, int64_t>> meshAxes = {
      {"a", 3}, {"b", 3}, {"c", 3}};
  SmallVector<MeshAxisAttr> meshAxisAttrs;
  meshAxisAttrs.reserve(meshAxes.size());
  for (auto [axisName, size] : meshAxes) {
    meshAxisAttrs.push_back(MeshAxisAttr::get(&context, axisName, size));
  }
  MeshAttr mesh = MeshAttr::get(&context, meshAxisAttrs);

  TensorFactorShardings operand = {
      .factorIndexToSharding = {
          {0,
           {.axisRefs = {createAxis("a"), createAxis("b")},
            .isMinorMost = true}},
          {1, {.axisRefs = {createAxis("c")}, .isMinorMost = true}},
      }};
  TensorFactorShardings resultBefore = {
      .factorIndexToSharding = {
          {0, {.axisRefs = {}, .isMinorMost = false}},
          {1, {.axisRefs = {}, .isMinorMost = true}},
      }};

  // The compatible major axes for factor 0 are ["a", "b"]. This factor is
  // non-minor-most in the result (with partial sharding []).
  //
  // The compatible major axes for factor 1 are ["c"]. This factor is minor-most
  // in the result (with partial sharding []), therefore it's fine that the
  // factor size (4) isn't divisible by the size of ["c"] (3).

  auto test = [&](ArrayRef<int64_t> factorSizes,
                  const TensorFactorShardings& resultAfter) {
    ShardingProjection projectionBefore({operand}, {resultBefore});
    ShardingProjection projectionAfter({operand}, {resultAfter});
    auto [updateOperands, updateResults] = propagateFactorShardings(
        projectionBefore, factorSizes, propagateAnything(), mesh);
    EXPECT_THAT(toSetBitsVector(updateOperands), IsEmpty());
    EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0));
    EXPECT_EQ(projectionBefore, projectionAfter);
  };

  {
    // The factor size (9) is divisible by the size of ["a", "b"] (9).
    SmallVector<int64_t> factorSizes = {9, 4};
    TensorFactorShardings resultAfter = {
        .factorIndexToSharding = {
            {0,
             {.axisRefs = {createAxis("a"), createAxis("b")},
              .isMinorMost = false}},
            {1, {.axisRefs = {createAxis("c")}, .isMinorMost = true}},
        }};
    test(factorSizes, resultAfter);
  }

  {
    // The factor size (6) is divisible by the size of ["a"] (3), but not by the
    // size of ["a", "b"] (9).
    SmallVector<int64_t> factorSizes = {6, 19};
    TensorFactorShardings resultAfter = {
        .factorIndexToSharding = {
            {0, {.axisRefs = {createAxis("a")}, .isMinorMost = false}},
            {1, {.axisRefs = {createAxis("c")}, .isMinorMost = true}},
        }};
    test(factorSizes, resultAfter);
  }

  {
    // The factor size (4) isn't divisible by the size of ["a"] (3).
    SmallVector<int64_t> factorSizes = {4, 1};
    TensorFactorShardings resultAfter = {
        .factorIndexToSharding = {
            {0, {.axisRefs = {}, .isMinorMost = false}},
            {1, {.axisRefs = {createAxis("c")}, .isMinorMost = true}},
        }};
    test(factorSizes, resultAfter);
  }
}

TEST_F(BasicFactorPropagationTest, DifferentDirectionsForDifferentFactors) {
  ShardingProjection projection(
      /*operands=*/
      {{.factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}},
                                  {1, {.axisRefs = {createAxis("b")}}},
                                  {2, {.axisRefs = {createAxis("c")}}},
                                  {3, {.axisRefs = {createAxis("d")}}},
                                  {4, {.axisRefs = {}}},
                                  {5, {.axisRefs = {}}},
                                  {6, {.axisRefs = {}}},
                                  {7, {.axisRefs = {}}}}},
       {.factorIndexToSharding = {{0, {.axisRefs = {}}},
                                  {1, {.axisRefs = {}}},
                                  {2, {.axisRefs = {}}},
                                  {3, {.axisRefs = {}}},
                                  {4, {.axisRefs = {}}},
                                  {5, {.axisRefs = {}}},
                                  {6, {.axisRefs = {}}},
                                  {7, {.axisRefs = {}}}}}},
      /*results=*/
      {{.factorIndexToSharding = {{0, {.axisRefs = {}}},
                                  {1, {.axisRefs = {}}},
                                  {2, {.axisRefs = {}}},
                                  {3, {.axisRefs = {}}},
                                  {4, {.axisRefs = {createAxis("e")}}},
                                  {5, {.axisRefs = {createAxis("f")}}},
                                  {6, {.axisRefs = {}}},
                                  {7, {.axisRefs = {createAxis("h")}}}}},
       {.factorIndexToSharding = {{0, {.axisRefs = {}}},
                                  {1, {.axisRefs = {}}},
                                  {2, {.axisRefs = {}}},
                                  {3, {.axisRefs = {}}},
                                  {4, {.axisRefs = {}}},
                                  {5, {.axisRefs = {}}},
                                  {6, {.axisRefs = {createAxis("g")}}},
                                  {7, {.axisRefs = {}}}}}});

  PropagationDirectionAlongFactor directionAlongFactor =
      [](int64_t factorIndex) {
        if (factorIndex == 0 || factorIndex == 4) {
          return PropagationDirection::BOTH;
        }
        if (factorIndex == 1 || factorIndex == 5) {
          return PropagationDirection::FORWARD;
        }
        if (factorIndex == 2 || factorIndex == 6) {
          return PropagationDirection::BACKWARD;
        }
        return PropagationDirection::NONE;
      };

  ShardingProjection projectionExpected(
      /*operands=*/
      {{.factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}},
                                  {1, {.axisRefs = {createAxis("b")}}},
                                  {2, {.axisRefs = {createAxis("c")}}},
                                  {3, {.axisRefs = {createAxis("d")}}},
                                  {4, {.axisRefs = {createAxis("e")}}},
                                  {5, {.axisRefs = {}}},
                                  {6, {.axisRefs = {createAxis("g")}}},
                                  {7, {.axisRefs = {}}}}},
       {.factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}},
                                  {1, {.axisRefs = {}}},
                                  {2, {.axisRefs = {}}},
                                  {3, {.axisRefs = {}}},
                                  {4, {.axisRefs = {createAxis("e")}}},
                                  {5, {.axisRefs = {}}},
                                  {6, {.axisRefs = {createAxis("g")}}},
                                  {7, {.axisRefs = {}}}}}},
      /*results=*/
      {{.factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}},
                                  {1, {.axisRefs = {createAxis("b")}}},
                                  {2, {.axisRefs = {}}},
                                  {3, {.axisRefs = {}}},
                                  {4, {.axisRefs = {createAxis("e")}}},
                                  {5, {.axisRefs = {createAxis("f")}}},
                                  {6, {.axisRefs = {}}},
                                  {7, {.axisRefs = {createAxis("h")}}}}},
       {.factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}},
                                  {1, {.axisRefs = {createAxis("b")}}},
                                  {2, {.axisRefs = {}}},
                                  {3, {.axisRefs = {}}},
                                  {4, {.axisRefs = {createAxis("e")}}},
                                  {5, {.axisRefs = {}}},
                                  {6, {.axisRefs = {createAxis("g")}}},
                                  {7, {.axisRefs = {}}}}}});

  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 8, directionAlongFactor);
  EXPECT_THAT(toSetBitsVector(updateOperands), ElementsAre(0, 1));
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0, 1));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(BasicFactorPropagationTest, UniDirectionalPropagationWithConflict) {
  ShardingProjection projection(
      /*operands=*/
      {{.factorIndexToSharding =
            {{0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
             {1, {.axisRefs = {}}},
             {2, {.axisRefs = {createAxis("d")}}}}}},
      /*results=*/
      {{.factorIndexToSharding = {
            {0, {.axisRefs = {createAxis("b")}}},
            {1, {.axisRefs = {createAxis("c"), createAxis("d")}}},
            {2, {.axisRefs = {}}}}}});

  PropagationDirectionAlongFactor directionAlongFactor =
      [](int64_t factorIndex) {
        if (factorIndex == 0) {
          return PropagationDirection::FORWARD;
        }
        if (factorIndex == 1) {
          return PropagationDirection::BACKWARD;
        }
        return PropagationDirection::NONE;
      };

  ShardingProjection projectionExpected(
      /*operands=*/
      {{.factorIndexToSharding =
            {{0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
             {1, {.axisRefs = {createAxis("c")}}},
             {2, {.axisRefs = {createAxis("d")}}}}}},
      /*results=*/{projection.getResult(0)});

  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 3, directionAlongFactor);
  EXPECT_THAT(toSetBitsVector(updateOperands), ElementsAre(0));
  EXPECT_THAT(toSetBitsVector(updateResults), IsEmpty());
}

TEST_F(BasicFactorPropagationTest,
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

  // In conservative mode, sub-axes are not propagated. Hence, only the full "a"
  // will be propagated to the result, while other axes are not propagated.

  ShardingProjection projectionExpected(
      /*operands=*/{projection.getOperand(0)},
      /*results=*/{{.factorIndexToSharding = {
                        {0, {.axisRefs = {createAxis("a")}}}, {1, {}}}}});

  auto [updateOperands, updateResults] = propagateFactorShardings(
      projection, 2, propagateAnything(), /*mesh=*/nullptr,
      /*conservativePropagation=*/true);
  EXPECT_THAT(toSetBitsVector(updateOperands), IsEmpty());
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0));
  EXPECT_EQ(projection, projectionExpected);
}

// NOLINTEND(clang-diagnostic-pre-c++20-compat-pedantic)

}  // namespace
}  // namespace sdy
}  // namespace mlir
