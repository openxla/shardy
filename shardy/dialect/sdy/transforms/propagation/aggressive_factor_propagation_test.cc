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

class AggressiveFactorPropagationTest : public PropagationTestBase {
 protected:
  UpdateTensorShardings propagateFactorShardings(
      ShardingProjection& projection, int64_t numFactors,
      PropagateAlongFactorPred propagateAlongFactor = [](int64_t) {
        return true;
      }) {
    return AggressiveFactorPropagation().propagateFactorShardings(
        projection, /*direction=*/PropagationDirection::BOTH,
        propagateAlongFactor, SmallVector<int64_t>(numFactors, 1),
        /*mesh=*/nullptr, /*op=*/nullptr, /*conservativePropagation*/ false);
  }
};

TEST_F(AggressiveFactorPropagationTest, RealAndFakeConflicts) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
                   {2, {.axisRefs = {createAxis("c")}}},
                   {4, {.axisRefs = {createAxis("b")}}},
                   {6, {.axisRefs = {createAxis("e")}}},
                   {7, {.axisRefs = {}}},
                   {8, {.axisRefs = {createAxis("f")}}},
                   {10, {.axisRefs = {createAxis("g")}}},
               }},
          {.factorIndexToSharding =
               {
                   {1, {.axisRefs = {createAxis("a")}}},
                   {2, {.axisRefs = {}}},
                   {3, {.axisRefs = {createAxis("c")}}},
                   {4, {.axisRefs = {}}},
                   {5, {.axisRefs = {createAxis("b")}}},
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
                   {2, {.axisRefs = {}, .overflowAxes = {createAxis("d")}}},
                   {3, {.axisRefs = {}}},
                   {4, {.axisRefs = {}, .isClosed = true}},
                   {5, {.axisRefs = {}}},
                   {6, {.axisRefs = {}, .isClosed = true}},
                   {7, {.axisRefs = {}}},
                   {8, {.axisRefs = {}}},
                   {9, {.axisRefs = {}}},
               }},
      });
  ShardingProjection projectionExpected(
      /*operands=*/{projection.getOperand(0), projection.getOperand(1)},
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a")}}},
                   {1, {.axisRefs = {}}},
                   {2, {.axisRefs = {}, .overflowAxes = {createAxis("d")}}},
                   {3, {.axisRefs = {createAxis("c")}}},
                   {4, {.axisRefs = {}, .isClosed = true}},
                   {5, {.axisRefs = {createAxis("b")}}},
                   {6, {.axisRefs = {}, .isClosed = true}},
                   {7, {.axisRefs = {createAxis("e")}}},
                   {8, {.axisRefs = {createAxis("f")}}},
                   {9, {.axisRefs = {createAxis("g")}}},
               }},
      });

  // Axis "a" may be propagated to the result along factors 0 or 1, which forms
  // a real conflict. We prefer factor 0 because its source is the first operand
  // (all tensors have the same size).
  //
  // Other conflicts are fake. We can propagate other axes as much as possible.
  // Axes "c", "b", "e", "f", "g" can be propagated to the result along factors
  // 3, 5, 7, 8, 9, respectively, since these axes cannot propagated to the
  // result along other factors. Also the sharding after propagation is valid
  // (sharding axes are not overlapped with each other).
  //
  // Propagation on different factors are independent. Although we cannot
  // propagate "e" to the Operand 0 along factor 7, we still propagate "e" to
  // the result along factor 7.
  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 11);
  EXPECT_THAT(toSetBitsVector(updateOperands), IsEmpty());
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(AggressiveFactorPropagationTest, TwoFactorsDoNotCoExistInAnyTensor) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}}}},
          {.factorIndexToSharding = {{1, {.axisRefs = {createAxis("a")}}}}},
      },
      /*results=*/{
          {.factorIndexToSharding = {{0, {.axisRefs = {}}}}},
          {.factorIndexToSharding = {{1, {.axisRefs = {}}}}},
      });
  ShardingProjection projectionExpected(
      /*operands=*/{projection.getOperand(0), projection.getOperand(1)},
      /*results=*/{projection.getOperand(0), projection.getOperand(1)});

  // We can propagate axis "a" along both factors since the two factors do not
  // co-exist in any tensor.
  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 2);
  EXPECT_THAT(toSetBitsVector(updateOperands), IsEmpty());
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0, 1));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(AggressiveFactorPropagationTest,
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

  ShardingProjection projectionExpected(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
                   {3,
                    {.axisRefs = {createSubAxis("h", 1, 2), createAxis("i")}}},
               }},
          projection.getOperand(1),
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
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
  EXPECT_THAT(toSetBitsVector(updateOperands), ElementsAre(0, 2));
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0, 2));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(AggressiveFactorPropagationTest, NewAxesConflict) {
  ShardingProjection projection(
      /*operands=*/
      {
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("c")}}},
                   {1, {.axisRefs = {}}},
                   {2, {.axisRefs = {}, .isClosed = true}},
                   {3, {.axisRefs = {}}},
               },
           .replicatedAxes = {createAxis("d")}},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
                   {1, {.axisRefs = {createAxis("b"), createAxis("a")}}},
                   {2, {.axisRefs = {}}},
                   {3, {.axisRefs = {createAxis("d")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}, .isClosed = true}},
                   {1, {.axisRefs = {}}},
                   {2, {.axisRefs = {createAxis("c"), createAxis("a")}}},
                   {3, {.axisRefs = {}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
                   {1, {.axisRefs = {}, .isClosed = true}},
                   {2, {.axisRefs = {}}},
                   {3, {.axisRefs = {}}},
               }},
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("d")}}},
                   {1, {.axisRefs = {}}},
                   {2, {.axisRefs = {}}},
                   {3, {.axisRefs = {}}},
               }},
      });

  ShardingProjection projectionExpected(
      /*operands=*/
      {
          projection.getOperand(0),
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}}},
                   {1, {.axisRefs = {createAxis("b"), createAxis("a")}}},
                   {2, {.axisRefs = {createAxis("c")}}},
                   {3, {.axisRefs = {createAxis("d")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {}, .isClosed = true}},
                   {1, {.axisRefs = {createAxis("b")}}},
                   {2, {.axisRefs = {createAxis("c"), createAxis("a")}}},
                   {3, {.axisRefs = {createAxis("d")}}},
               }},
      },
      /*results=*/{
          {.factorIndexToSharding =
               {
                   {0, {.axisRefs = {createAxis("a"), createAxis("b")}}},
                   {1, {.axisRefs = {}, .isClosed = true}},
                   {2, {.axisRefs = {createAxis("c")}}},
                   {3, {.axisRefs = {createAxis("d")}}},
               }},
          {.factorIndexToSharding =
               {
                   {0,
                    {.axisRefs = {createAxis("a"), createAxis("b"),
                                  createAxis("d")}}},
                   {1, {.axisRefs = {}}},
                   {2, {.axisRefs = {createAxis("c")}}},
                   {3, {.axisRefs = {}}},
               }},
      });

  // “a” can be propagated to the Result 0 along either Factor 0 or Factor 2.
  // This strategy prefers factor 0 because its source is the first operand
  // (all tensors have the same size).
  auto [updateOperands, updateResults] =
      propagateFactorShardings(projection, 4);
  EXPECT_THAT(toSetBitsVector(updateOperands), ElementsAre(1, 2));
  EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0, 1));
  EXPECT_EQ(projection, projectionExpected);
}

TEST_F(AggressiveFactorPropagationTest, PropagateAlongSpecificFactor) {
  TensorFactorShardings factor0IsSharded = {
      .factorIndexToSharding = {{0, {.axisRefs = {createAxis("a")}}}, {1, {}}}};
  TensorFactorShardings factor1IsSharded = {
      .factorIndexToSharding = {{0, {}}, {1, {.axisRefs = {createAxis("a")}}}}};
  TensorFactorShardings replicated = {
      .factorIndexToSharding = {{0, {}}, {1, {}}}};

  auto propagateAlongFactor =
      [&](PropagateAlongFactorPred propagateAlongFactor,
          const ShardingProjection& projectionExpected) {
        ShardingProjection projection(
            /*operands=*/{factor0IsSharded, factor1IsSharded},
            /*results=*/{replicated});

        auto [updateOperands, updateResults] =
            propagateFactorShardings(projection, 2, propagateAlongFactor);
        EXPECT_THAT(toSetBitsVector(updateOperands), IsEmpty());
        EXPECT_THAT(toSetBitsVector(updateResults), ElementsAre(0));
        EXPECT_EQ(projection, projectionExpected);
      };

  ShardingProjection propagateAlongFactor0Expected(
      /*operands=*/{factor0IsSharded, factor1IsSharded},
      /*results=*/{factor0IsSharded});
  propagateAlongFactor([](int64_t factorIndex) { return factorIndex == 0; },
                       propagateAlongFactor0Expected);
  propagateAlongFactor([](int64_t factorIndex) { return factorIndex != 1; },
                       propagateAlongFactor0Expected);
  // When we propagate along all factors, we propagate "a" to the result along
  // factor 0.
  propagateAlongFactor([](int64_t factorIndex) { return true; },
                       propagateAlongFactor0Expected);

  ShardingProjection propagateAlongFactor1Expected(
      /*operands=*/{factor0IsSharded, factor1IsSharded},
      /*results=*/{factor1IsSharded});
  propagateAlongFactor([](int64_t factorIndex) { return factorIndex == 1; },
                       propagateAlongFactor1Expected);
  propagateAlongFactor([](int64_t factorIndex) { return factorIndex != 0; },
                       propagateAlongFactor1Expected);
}

}  // namespace
}  // namespace sdy
}  // namespace mlir
