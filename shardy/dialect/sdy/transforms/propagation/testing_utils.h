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

#include <cstdint>

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

using ::testing::DescribeMatcher;
using ::testing::IsEmpty;
using ::testing::PrintToString;

MATCHER_P(AxisRefIs, axisName,
          (negation ? "axis isn't " : "axis is ") + PrintToString(axisName)) {
  *result_listener << "where axis is " << arg.toString();
  return arg.getName() == axisName;
}

MATCHER_P3(SubAxisRefIs, axisName, preSize, size,
           (negation ? "sub-axis isn't " : "sub-axis is ") +
               PrintToString(axisName) + ":(" + PrintToString(preSize) + ")" +
               PrintToString(size)) {
  *result_listener << "where sub-axis is " << arg.toString();
  return arg.getName() == axisName && arg.getSubAxisInfo() &&
         arg.getSubAxisInfo().getPreSize() == preSize &&
         arg.getSubAxisInfo().getSize() == size;
}

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

MATCHER_P2(TensorFactorShardingsIs, factorIndexToShardingMatcher,
           replicatedAxesMatcher,
           "tensor factor shardings that:\n" +
               DescribeMatcher<FactorIndexToSharding>(
                   factorIndexToShardingMatcher, negation) +
               "\n" + (negation ? "or" : "and") + " replicated axes that " +
               DescribeMatcher<ArrayRef<AxisRefAttr>>(replicatedAxesMatcher,
                                                      negation)) {
  if (!ExplainMatchResult(factorIndexToShardingMatcher,
                          arg.factorIndexToSharding, result_listener)) {
    return false;
  }
  *result_listener << "\nand replicated axes ";
  return ExplainMatchResult(replicatedAxesMatcher, arg.replicatedAxes,
                            result_listener);
}

class PropagationTestBase : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  AxisRefAttr createAxis(StringRef name) {
    return AxisRefAttr::get(&context, name);
  }

  AxisRefAttr createSubAxis(StringRef name, int64_t preSize, int64_t size) {
    return AxisRefAttr::get(&context, name, preSize, size);
  }

  MLIRContext context;
};

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_TRANSFORMS_PROPAGATION_TESTING_UTILS_H_
