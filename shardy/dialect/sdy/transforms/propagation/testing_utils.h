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

#include <cassert>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_registry.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {

namespace testing_utils {

inline constexpr StringRef kMeshName = "mesh";

inline void verifyShardingAttrsMatch(TensorShardingAttr resultSharding,
                                     TensorShardingAttr expectedSharding) {
  EXPECT_EQ(resultSharding, expectedSharding)
      << "result: " << llvm::to_string(resultSharding)
      << ", expected: " << llvm::to_string(expectedSharding);
}

inline void verifyReconstructedShardings(
    ValueRange tensors, ArrayRef<TensorFactorShardings> tensorFactorShardings,
    ArrayRef<TensorMappingAttr> tensorMappings, ArrayRef<int64_t> factorSizes,
    StringRef meshName, MeshAttr mesh) {
  for (auto [tensor, factorShardings, tensorMapping] :
       llvm::zip(tensors, tensorFactorShardings, tensorMappings)) {
    TensorShardingAttr reconstructedSharding =
        factorShardings.createTensorShardingAttr(
            mesh.getContext(), tensorMapping, factorSizes, kMeshName, mesh);
    verifyShardingAttrsMatch(reconstructedSharding,
                             getOrCreateSharding(tensor, kMeshName));
  }
}

inline MeshAttr getMeshAttr(ModuleOp module) {
  return cast<MeshOp>(module.lookupSymbol(kMeshName)).getMesh();
}

template <class OpTy>
OpTy getFirstOp(ModuleOp module) {
  auto mainFn = cast<func::FuncOp>(module.lookupSymbol("main"));
  auto ops = mainFn.getBody().front().getOps<OpTy>();
  assert(!ops.empty());
  return *ops.begin();
}

// Builds a `ShardingProjection` for the first OpTy in the main function.
//
// In addition, verifies that reconstructing the `TensorShardingAttr` for each
// tensor (using `TensorFactorShardings::createTensorShardingAttr`) from the
// created projection matches the original sharding.
template <class OpTy>
ShardingProjection getShardingProjection(ModuleOp module) {
  OpTy op = getFirstOp<OpTy>(module);
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  assert(shardingRule);
  MeshAttr mesh = getMeshAttr(module);
  ShardingProjection projection =
      ShardingProjection::build(op, shardingRule, mesh);
  verifyReconstructedShardings(op->getOperands(), projection.getOperands(),
                               shardingRule.getOperandMappings(),
                               shardingRule.getFactorSizes(), kMeshName, mesh);
  verifyReconstructedShardings(op->getResults(), projection.getResults(),
                               shardingRule.getResultMappings(),
                               shardingRule.getFactorSizes(), kMeshName, mesh);
  return projection;
}

template <class OpTy>
ShardingProjection getShardingProjection(
    ModuleOp module, ArrayRef<ArrayRef<AxisRefAttr>> axisRefsList) {
  OpTy op = getFirstOp<OpTy>(module);
  OpShardingRuleAttr shardingRule = getOrCreateShardingRule(op);
  assert(shardingRule);
  ShardingProjection projection =
      ShardingProjection::build(axisRefsList, shardingRule);
  return projection;
}
}  // namespace testing_utils

namespace {
using ::testing::DescribeMatcher;
using ::testing::IsEmpty;
using ::testing::PrintToString;
}  // namespace

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
