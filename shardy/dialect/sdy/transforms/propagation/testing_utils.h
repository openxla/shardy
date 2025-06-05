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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_TESTING_UTILS_H_
#define SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_TESTING_UTILS_H_

#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

using ::testing::DescribeMatcher;
using ::testing::IsEmpty;
using ::testing::PrintToString;

MATCHER_P5(FactorShardingWithOverflowIs, index, isClosed, isMinorMost,
           axisRefsMatcher, overflowAxesMatcher,
           "factor " + PrintToString(index) + " sharding that is " +
               (isClosed || negation ? "closed" : "open") +
               (negation ? " or " : " and ") +
               (isMinorMost || negation ? "minor-most" : "non-minor-most") +
               (negation ? " or " : " and ") +
               DescribeMatcher<ArrayRef<AxisRefAttr>>(axisRefsMatcher,
                                                      negation) +
               "\n" + (negation ? " or" : " and") + " has overflow axes that " +
               DescribeMatcher<ArrayRef<AxisRefAttr>>(overflowAxesMatcher,
                                                      negation)) {
  *result_listener << "where factor " << arg.first << " sharding is "
                   << (arg.second.isClosed ? "closed" : "open") << " and ";
  if (arg.first != index || arg.second.isClosed != isClosed ||
      arg.second.isMinorMost != isMinorMost ||
      !ExplainMatchResult(axisRefsMatcher, arg.second.axisRefs,
                          result_listener)) {
    return false;
  }
  *result_listener << "\nand overflow axes ";
  return ExplainMatchResult(overflowAxesMatcher, arg.second.overflowAxes,
                            result_listener);
}

MATCHER_P4(FactorShardingIs, index, isClosed, isMinorMost, axisRefsMatcher,
           DescribeMatcher<FactorIndexToSharding::value_type>(
               FactorShardingWithOverflowIs(index, isClosed, isMinorMost,
                                            axisRefsMatcher, IsEmpty()),
               negation)) {
  return ExplainMatchResult(
      FactorShardingWithOverflowIs(index, isClosed, isMinorMost,
                                   axisRefsMatcher, IsEmpty()),
      arg, result_listener);
}

MATCHER_P2(FactorShardingIsClosedIs, index, isClosed,
           "factor " + PrintToString(index) + " sharding that is " +
               (isClosed || negation ? "closed" : "open")) {
  *result_listener << "where factor " << arg.first << " sharding is "
                   << (arg.second.isClosed ? "closed" : "open");
  if (arg.first != index || arg.second.isClosed != isClosed) {
    return false;
  }
  return true;
}

MATCHER_P2(FactorShardingIsMinorMostIs, index, isMinorMost,
           "factor " + PrintToString(index) + " sharding that is " +
               (isMinorMost || negation ? "closed" : "open")) {
  *result_listener << "where factor " << arg.first << " sharding is "
                   << (arg.second.isMinorMost ? "closed" : "open");
  if (arg.first != index || arg.second.isMinorMost != isMinorMost) {
    return false;
  }
  return true;
}

MATCHER_P3(TensorFactorShardingsIs, factorIndexToShardingMatcher,
           replicatedAxesMatcher, unreducedAxesMatcher,
           "tensor factor shardings that:\n" +
               DescribeMatcher<FactorIndexToSharding>(
                   factorIndexToShardingMatcher, negation) +
               "\n" + (negation ? "or" : "and") + " replicated axes that " +
               DescribeMatcher<ArrayRef<AxisRefAttr>>(replicatedAxesMatcher,
                                                      negation) +
               "\n" + (negation ? "or" : "and") + " unreduced axes that " +
               DescribeMatcher<ArrayRef<AxisRefAttr>>(unreducedAxesMatcher,
                                                      negation)) {
  if (!ExplainMatchResult(factorIndexToShardingMatcher,
                          arg.factorIndexToSharding, result_listener)) {
    return false;
  }
  *result_listener << "\nand replicated axes ";
  if (!ExplainMatchResult(replicatedAxesMatcher, arg.replicatedAxes,
                          result_listener)) {
    return false;
  }
  *result_listener << "\nand unreduced axes ";
  return ExplainMatchResult(unreducedAxesMatcher, arg.unreducedAxes,
                            result_listener);
}

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_TESTING_UTILS_H_
