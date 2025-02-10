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

#include "shardy/dialect/sdy/transforms/propagation/debugging/source_sharding.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

namespace {

// The map from factor to edge source for operands and results.
struct FactorsToEdgeMap {
  llvm::SmallVector<AxisToEdgeMap> operands, results;
};

// Finds what operand/result the new sharding axes came from for a given
// `factorIndex`.
std::optional<int64_t> findNewAxisRefMatch(
    ArrayRef<AxisRefAttr> newAxisRefs, int64_t factorIndex,
    ArrayRef<TensorFactorShardings> oldTensorFactorShardings) {
  for (auto [valueIndex, tensorFactorSharding] :
       llvm::enumerate(oldTensorFactorShardings)) {
    auto oldAxisRefsIt =
        tensorFactorSharding.factorIndexToSharding.find(factorIndex);
    if (oldAxisRefsIt != tensorFactorSharding.factorIndexToSharding.end() &&
        isAxisListPrefixOf(newAxisRefs, oldAxisRefsIt->second.axisRefs) !=
            PrefixStatus::NOT_A_PREFIX) {
      return valueIndex;
    }
  }
  return std::nullopt;
}

// Creates a map from factor to edge source for operands and results.
//
// This only saves any newly introduced factor shardings, not any pre-existing
// ones. So if no operand/result sharding changes, the map will be empty.
FactorsToEdgeMap createSourceMap(
    const ShardingProjection& oldShardingProjection,
    const ShardingProjection& newShardingProjection,
    OpShardingRuleAttr shardingRule, MeshAttr mesh,
    const int64_t propagationStep) {
  FactorsToEdgeMap axisToEdgeMap{
      llvm::SmallVector<AxisToEdgeMap>(oldShardingProjection.getNumOperands(),
                                       AxisToEdgeMap()),
      llvm::SmallVector<AxisToEdgeMap>(oldShardingProjection.getNumResults(),
                                       AxisToEdgeMap())};

  // Saves the `axisRefs` to the specified `valueSourceMap` of
  // `axisToEdgeMap`.
  auto saveEdges = [&](ArrayRef<AxisRefAttr> newAxisRefs,
                       ArrayRef<AxisRefAttr> oldAxisRefs, EdgeNode source,
                       EdgeNode target, AxisToEdgeMap& valueSourceMap) {
    // To avoid iterating over all the new axes, only compare the very last old
    // axis (since there could have been a sub-axis update) and then the
    // trailing new axes.
    int64_t oldAxisIndex = oldAxisRefs.size() - 1;
    if (!oldAxisRefs.empty() &&
        oldAxisRefs[oldAxisIndex] != newAxisRefs[oldAxisIndex]) {
      valueSourceMap.try_emplace(
          newAxisRefs[oldAxisIndex],
          PropagationEdge{source, target, propagationStep});
    }
    for (AxisRefAttr axisRef : newAxisRefs.drop_front(oldAxisRefs.size())) {
      valueSourceMap.try_emplace(
          axisRef, PropagationEdge{source, target, propagationStep});
    }
  };

  MLIRContext* context = mesh.getContext();
  ArrayRef<int64_t> factorSizes = shardingRule.getFactorSizes();
  auto visitValue = [&](const TensorFactorShardings& oldValue,
                        const TensorFactorShardings& newValue,
                        EdgeNodeType valueType, int64_t valueIndex,
                        TensorMappingAttr tensorMapping,
                        llvm::SmallVector<AxisToEdgeMap>& valueSourceMap) {
    DenseSet<AxisRefAttr> oldAxes;
    for (const auto& [_, oldFactorSharding] : oldValue.factorIndexToSharding) {
      oldAxes.insert(oldFactorSharding.axisRefs.begin(),
                     oldFactorSharding.axisRefs.end());
    }
    for (const auto& [oldFactorSharding, newFactorSharding] : llvm::zip_equal(
             oldValue.factorIndexToSharding, newValue.factorIndexToSharding)) {
      const auto& newAxisRefs = newFactorSharding.second.axisRefs;
      if (newAxisRefs == oldFactorSharding.second.axisRefs) {
        continue;
      }
      SmallVector<AxisRefAttr> newlyIntroducedAxes;
      // If multiple sub axes can be merged due to a dimension sharding having
      // multiple factors, each sharded on a sub axis, make sure we only save
      // the merged one. This can happen during an `(A, B) -> (AB,)` reshape.
      TensorShardingAttr tensorSharding = newValue.createTensorShardingAttr(
          context, tensorMapping, factorSizes, "", mesh);
      for (DimensionShardingAttr dimSharding :
           tensorSharding.getDimShardings()) {
        llvm::copy_if(
            dimSharding.getAxes(), std::back_inserter(newlyIntroducedAxes),
            [&](const AxisRefAttr& axisRef) {
              // Don't add any axes that were already in the
              // old sharding. We just want new axes.
              if (oldAxes.contains(axisRef)) {
                return false;
              }
              // We need to avoid any axes that already existed in the old
              // sharding, but aren't in the new projection as the conflicted.
              // E.g. for a contracting dim matmul, if both the LHS/RHS are
              // sharded on the same axis on their respective non-contracting
              // dims, the dimension sharding will contain the conflicting axes,
              // but the factor sharding will not. And we don't want this axis
              // as it isn't a newly introduced axis.
              for (AxisRefAttr newAxisRef : newAxisRefs) {
                if (newAxisRef.prefixOf(axisRef)) {
                  return true;
                }
              }
              return false;
            });
      }
      // This factor sharding has changed, let's find who changed it.
      if (std::optional<int64_t> operandSource =
              findNewAxisRefMatch(newAxisRefs, oldFactorSharding.first,
                                  oldShardingProjection.getOperands())) {
        saveEdges(newlyIntroducedAxes, oldFactorSharding.second.axisRefs,
                  EdgeNode{EdgeNodeType::OPERAND, *operandSource},
                  EdgeNode{valueType, valueIndex}, valueSourceMap[valueIndex]);
      } else if (std::optional<int64_t> resultSource =
                     findNewAxisRefMatch(newAxisRefs, oldFactorSharding.first,
                                         oldShardingProjection.getResults())) {
        saveEdges(newlyIntroducedAxes, oldFactorSharding.second.axisRefs,
                  EdgeNode{EdgeNodeType::RESULT, *resultSource},
                  EdgeNode{valueType, valueIndex}, valueSourceMap[valueIndex]);
      }
    }
  };

  for (auto [i, packedOperands] :
       llvm::enumerate(llvm::zip_equal(oldShardingProjection.getOperands(),
                                       newShardingProjection.getOperands()))) {
    auto [oldOperand, newOperand] = packedOperands;
    visitValue(oldOperand, newOperand, EdgeNodeType::OPERAND, i,
               shardingRule.getOperandMapping(i), axisToEdgeMap.operands);
  }
  for (auto [i, packedResults] :
       llvm::enumerate(llvm::zip_equal(oldShardingProjection.getResults(),
                                       newShardingProjection.getResults()))) {
    auto [oldResult, newResult] = packedResults;
    visitValue(oldResult, newResult, EdgeNodeType::RESULT, i,
               shardingRule.getResultMapping(i), axisToEdgeMap.results);
  }

  return axisToEdgeMap;
}

std::string manualComputationOriginName(OriginShardingType type, StringRef name,
                                        int64_t index) {
  switch (type) {
    case OriginShardingType::MC_INPUT: {
      return llvm::formatv("{0}_input: {1}", name, index);
    }
    case OriginShardingType::MC_OUTPUT: {
      return llvm::formatv("{0}_output: {1}", name, index);
    }
    default:
      unreachableFormatv(
          "Passed in type {0} for manual computation origin name. Only "
          "MC_INPUT and MC_OUTPUT are supported.",
          type);
  }
  llvm_unreachable("unknown OriginShardingType");
}

std::string manualComputationOriginName(OriginShardingType type,
                                        int64_t sourceId, int64_t index) {
  return manualComputationOriginName(
      type, llvm::formatv("mc_{0}", sourceId).str(), index);
}

// Converts the `source` to a `StringAttr`.
StringAttr shardingOriginToString(OriginSharding source, MLIRContext* context) {
  std::string typeString;
  switch (source.type) {
    case OriginShardingType::INPUT: {
      typeString = "input";
      break;
    }
    case OriginShardingType::CONSTRAINT: {
      return StringAttr::get(context,
                             llvm::formatv("constraint_{0}", source.sourceId));
    }
    case OriginShardingType::OUTPUT: {
      typeString = "output";
      break;
    }
    case OriginShardingType::MC_INPUT:
    case OriginShardingType::MC_OUTPUT: {
      return StringAttr::get(
          context, manualComputationOriginName(source.type, source.sourceId,
                                               source.index));
    }
  }
  return StringAttr::get(context,
                         llvm::formatv("{0}: {1}", typeString, source.index));
}

// Avoid printing the string with escaping quotes, aka "\22".
void eraseDoubleQuotesInAxisRefString(std::string& axisRefString) {
  axisRefString.erase(remove(axisRefString.begin(), axisRefString.end(), '"'),
                      axisRefString.end());
}

// Create a list of entries from the `axisToOriginSharding` map to save as a
// `DictionaryAttr`.
SmallVector<NamedAttribute> createOriginShardingEntries(
    const AxisToOriginShardingMap& axisToOriginSharding, MLIRContext* context) {
  SmallVector<NamedAttribute> entries;
  entries.reserve(axisToOriginSharding.size());
  for (const auto& [axisRef, shardingOrigin] : axisToOriginSharding) {
    std::string axisRefString = axisRef.toString();
    eraseDoubleQuotesInAxisRefString(axisRefString);
    entries.emplace_back(
        NamedAttribute(StringAttr::get(context, axisRefString),
                       shardingOriginToString(shardingOrigin, context)));
  }
  return entries;
}

// Gets the existing `kShardingOriginsAttr` on the `op` or creates a new
// `ArrayAttr` with `op->getNumResults()` dictionaries.
SmallVector<Attribute> getOriginShardingDicts(Operation* op, Builder& builder) {
  auto resultDicts = op->getAttrOfType<ArrayAttr>(kShardingOriginsAttr);
  if (!resultDicts) {
    SmallVector<Attribute> newResultDicts(op->getNumResults(),
                                          builder.getDictionaryAttr({}));
    resultDicts = builder.getArrayAttr(newResultDicts);
  }
  return SmallVector<Attribute>(resultDicts.getValue());
}

// Gets the `OpOperand` of the `value` in the `funcOp` terminator if the Value
// is used in the terminator. Else returns `nullptr`.
OpOperand* getTerminatorOperand(Value value, func::FuncOp funcOp) {
  ArrayRef<OpOperand> terminatorOperands =
      getBodyTerminator(funcOp)->getOpOperands();
  if (auto it = llvm::find_if(value.getUses(),
                              [&](const OpOperand& use) {
                                return llvm::is_contained(terminatorOperands,
                                                          use);
                              });
      it != value.getUses().end()) {
    return it.getOperand();
  }
  return nullptr;
}

// Saves the originating sharding debug information on each `Value` in
// `valueToOriginShardingMap`.
void saveShardingOriginsOnModule(
    MLIRContext* context,
    const ValueToOriginShardingMap& valueToOriginShardingMap) {
  Builder builder(context);
  for (auto& [value, axisToOriginSharding] : valueToOriginShardingMap) {
    Operation* owningOp = getOwningOp(value);

    func::FuncOp funcOp = getEnclosingOfType<func::FuncOp>(owningOp);

    // TODO(bartchr): Swop the map to sture `ValueOrFuncResult` to avoid having
    // to do this terminator finding logic just to set the func result attr.
    OpOperand* terminatorOperand = getTerminatorOperand(value, funcOp);

    SmallVector<NamedAttribute> entries =
        createOriginShardingEntries(axisToOriginSharding, context);

    if (terminatorOperand) {
      int64_t operandNumber = terminatorOperand->getOperandNumber();
      funcOp.setResultAttr(operandNumber, kShardingOriginsAttr,
                           builder.getDictionaryAttr(entries));
    }
    TypeSwitch<Operation*, void>(owningOp)
        .Case<func::FuncOp>([&](func::FuncOp funcOp) {
          if (value) {
            funcOp.setArgAttr(cast<BlockArgument>(value).getArgNumber(),
                              kShardingOriginsAttr,
                              builder.getDictionaryAttr(entries));
          }
        })
        .Case<ShardingConstraintOp, DataFlowEdgeOp>([&](Operation* op) {
          op->setAttr(kShardingOriginsAttr, builder.getDictionaryAttr(entries));
        })
        .Default([&, value = value](Operation* op) {
          auto result = cast<OpResult>(value);
          // Need to handle the case where the generic `op` has multiple
          // results, so multiple dictionaries.
          SmallVector<Attribute> newResultDicts =
              getOriginShardingDicts(op, builder);
          newResultDicts[result.getResultNumber()] =
              builder.getDictionaryAttr(entries);
          op->setAttr(kShardingOriginsAttr,
                      builder.getArrayAttr(newResultDicts));
        });
  }
}

// Converts the `node` to a `NamedAttribute`.
StringAttr edgeNodeToString(EdgeNode node, Builder& builder) {
  std::string typeString;
  switch (node.type) {
    case EdgeNodeType::OPERAND: {
      typeString = "operand";
      break;
    }
    case EdgeNodeType::RESULT: {
      typeString = "result";
      break;
    }
  }
  return builder.getStringAttr(
      llvm::formatv("{0}: {1}", typeString, node.index));
}

// In the case where we have a Value used multiple times as an operand, we
// should only add the edge once. For example:
// ```mlir
// %0 = stablehlo.add %arg0, %arg0 <[<@mesh, [{"a", ?}]>]> : tensor<8xf32>
// return %0 : tensor<8xf32>
// ```
// The sharding projection said that both operand 0 and 1 are updated. However,
// they are the same value, so we only need to add the edge once. This is only
// the case for the target of the edge, because if the source appears multiple
// times, then it's because it effects multiple other operands/results in the
// op.
bool insertSeenValue(Operation* op, const PropagationEdge& edge,
                     llvm::SmallDenseSet<Value>& seenValues) {
  EdgeNode target = edge.target;
  switch (target.type) {
    case EdgeNodeType::OPERAND: {
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        return seenValues.insert(
            getBodyTerminator(funcOp)->getOperand(target.index)).second;
      }
      return seenValues.insert(op->getOperand(target.index)).second;
    }
    case EdgeNodeType::RESULT: {
      return true;
    }
  }
}

// Create a list of entries from the `axisToEdge` map to save as a
// `DictionaryAttr`.
DictionaryAttr createEdgeEntries(Operation* op,
                                 const AxisToEdgesMap& axisToEdges,
                                 MLIRContext* context) {
  Builder builder(context);
  SmallVector<NamedAttribute> entries;
  for (const auto& [axisRef, edges] : axisToEdges) {
    std::string axisRefString = axisRef.toString();
    eraseDoubleQuotesInAxisRefString(axisRefString);
    SmallVector<Attribute> axisEntries;
    llvm::SmallDenseSet<Value> seenTargetValues;
    for (const PropagationEdge& edge : edges) {
      assert(edge.source.type == EdgeNodeType::OPERAND ||
             edge.source.type == EdgeNodeType::RESULT);
      if (!insertSeenValue(op, edge, seenTargetValues)) {
        continue;
      }
      StringAttr sourceEntry = edgeNodeToString(edge.source, builder);
      StringAttr targetEntry = edgeNodeToString(edge.target, builder);
      DictionaryAttr edgeEntry = builder.getDictionaryAttr({
          builder.getNamedAttr("source", sourceEntry),
          builder.getNamedAttr("target", targetEntry),
          builder.getNamedAttr("propagation_step",
                               builder.getI64IntegerAttr(edge.propagationStep)),
      });
      axisEntries.push_back(edgeEntry);
    }
    entries.emplace_back(builder.getStringAttr(axisRefString),
                         builder.getArrayAttr(axisEntries));
  }
  return builder.getDictionaryAttr(entries);
}

// Saves the originating sharding debug information on each `op` in
// `mappings->operationToEdgesMap`.
//
// This works by having each `Operation*` save a source/target edge, which is
// always composed of at least one operand. This is because results
// can be used several times, but an operand only ever has one defining op. So
// these edges always look "backwards" - never forwards towards a use of a
// result.
//
// As such, the `FuncOp` args don't contain edge source information: only the
// ops that use them.
void saveEdgesOnModule(MLIRContext* context,
                       const OperationToEdgesMap& operationToEdgesMap) {
  Builder builder(context);
  for (auto [op, axisToEdges] : operationToEdgesMap) {
    if (isa<func::FuncOp>(op)) {
      continue;
    }
    op->setAttr(kPropagationEdgesAttr,
                createEdgeEntries(op, axisToEdges, context));
  }
}

// Saves the edge source sharding debug information on the result attrs of
// `funcOp`.
//
// Since only the uses of a sharding save the edge, for `FuncOp` results which
// have been updated, we save the edges in each result's `resultAttr`. If the
// function has multiple results, then each `edge_source` on the func result
// attributes will have index 0. This is because of how propagation works with
// running propagation on each returned result. Having them have the right index
// would make sense if the `sdy.edge_sources` were saved as a top level
// attribute on the func, but since one is saved per result, then an index of 0
// makes most sense.
void saveEdgesOnFuncResults(func::FuncOp funcOp,
                            const FuncResultToEdgesMap& funcResultToEdgesMap) {
  for (auto [funcOp, resultToEdgesMap] : funcResultToEdgesMap) {
    for (auto [resultIndex, axisToEdgesMap] :
         llvm::enumerate(resultToEdgesMap)) {
      funcOp.setResultAttr(
          resultIndex, kPropagationEdgesAttr,
          createEdgeEntries(funcOp, axisToEdgesMap, funcOp->getContext()));
    }
  }
}

// Saves the sharding origin information on the `value` to the `handler`.
void saveShardingOrigins(ValueToOriginShardingMap& valueToOriginShardingMap,
                         TensorShardingAttr sharding, OriginShardingType type,
                         Value value, int64_t valueIndex,
                         int64_t sourceId = 0) {
  if (!sharding) {
    return;
  }
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    for (AxisRefAttr axisRef : dimSharding.getAxes()) {
      valueToOriginShardingMap[value].try_emplace(
          axisRef, OriginSharding{type, valueIndex, sourceId});
    }
  }
}

// Creates a new dictionary for `kShardingOriginsAttr` on the function inputs
// and outputs, setting the origin to `self` if the previous origin was the
// same as the `valueIndex` for the given `type`.
DictionaryAttr convertFuncOriginsToSelf(int64_t valueIndex,
                                        OriginShardingType type,
                                        DictionaryAttr dictAttr) {
  if (!dictAttr) {
    return {};
  }
  MLIRContext* context = dictAttr.getContext();
  StringAttr originName =
      shardingOriginToString(OriginSharding{type, valueIndex}, context);
  SmallVector<NamedAttribute> entries(dictAttr.getValue());
  for (auto& entry : entries) {
    if (cast<StringAttr>(entry.getValue()) == originName) {
      entry = NamedAttribute(entry.getName(), StringAttr::get(context, "self"));
    }
  }
  return DictionaryAttr::get(context, entries);
}

void setOpOriginsToSelf(Operation* op, StringRef originName) {
  MLIRContext* context = op->getContext();
  if (auto dictAttr = op->getAttrOfType<DictionaryAttr>(kShardingOriginsAttr)) {
    SmallVector<NamedAttribute> entries(dictAttr.getValue());
    for (auto& entry : entries) {
      if (cast<StringAttr>(entry.getValue()) == originName) {
        entry =
            NamedAttribute(entry.getName(), StringAttr::get(context, "self"));
      }
    }
    op->setAttr(kShardingOriginsAttr, DictionaryAttr::get(context, entries));
  }
}

// For `ShardingConstraintOp`s and `FuncOp` inputs/outputs, instead of saying
// that the existing axes at the start came from `constraint_0` or `input: 0`,
// we say that they came from `self`. Then it's not circular/clearer to
// understand.
void overrideOriginsToSelf(ModuleOp moduleOp) {
  moduleOp.walk([&](ShardingConstraintOp shardingConstraintOp) {
    setOpOriginsToSelf(shardingConstraintOp,
                       shardingConstraintOp->getAttrOfType<StringAttr>(
                           kShardingOriginNameAttr));
  });
  moduleOp.walk([&](func::FuncOp funcOp) {
    for (int64_t operandIndex = 0; operandIndex < funcOp.getNumArguments();
         ++operandIndex) {
      if (DictionaryAttr newDict = convertFuncOriginsToSelf(
              operandIndex, OriginShardingType::INPUT,
              funcOp.getArgAttrOfType<DictionaryAttr>(operandIndex,
                                                      kShardingOriginsAttr))) {
        funcOp.setArgAttr(operandIndex, kShardingOriginsAttr, newDict);
      }
    }
    for (int64_t resultIndex = 0; resultIndex < funcOp.getNumResults();
         ++resultIndex) {
      if (DictionaryAttr newDict = convertFuncOriginsToSelf(
              resultIndex, OriginShardingType::OUTPUT,
              funcOp.getResultAttrOfType<DictionaryAttr>(
                  resultIndex, kShardingOriginsAttr))) {
        funcOp.setResultAttr(resultIndex, kShardingOriginsAttr, newDict);
      }
    }
  });
  moduleOp.walk([&](ManualComputationOp manualComputationOp) {
    auto originName =
        manualComputationOp->getAttrOfType<StringAttr>(kShardingOriginNameAttr);
    for (BlockArgument blockArg :
         manualComputationOp.getBody().getArguments()) {
      setOpOriginsToSelf(DataFlowEdgeOp::lookup(blockArg),
                         manualComputationOriginName(
                             OriginShardingType::MC_INPUT,
                             originName.getValue(), blockArg.getArgNumber()));
    }
    for (OpResult result : manualComputationOp.getResults()) {
      setOpOriginsToSelf(DataFlowEdgeOp::lookup(result),
                         manualComputationOriginName(
                             OriginShardingType::MC_OUTPUT,
                             originName.getValue(), result.getResultNumber()));
    }
  });
}

// Sets up `valueToOriginShardingMap` with the initial sharding origin
// information on the `moduleOp`.
//
// The `SourceShardingHandler` will keep `valueToOriginShardingMap` up to date
// with the origin sharding information on the module during the propagation
// rewrite patterns.
void prepareShardingOriginsHandler(
    ModuleOp moduleOp, ValueToOriginShardingMap& valueToOriginShardingMap) {
  MLIRContext* context = moduleOp.getContext();
  // Build the initial sharding origin map.
  // NOTE(bartchr): This assumes that we do not propagate across different
  // functions. As all func inputs/outputs have the same source name. Update
  // this if we do propagate across `FuncOp`s.
  moduleOp.walk([&](func::FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      saveShardingOrigins(valueToOriginShardingMap, getSharding(arg),
                          OriginShardingType::INPUT, arg, arg.getArgNumber());
    }
    for (OpOperand& returnOperand : getBodyTerminatorOpOperands(funcOp)) {
      int64_t valueIndex = returnOperand.getOperandNumber();
      saveShardingOrigins(
          valueToOriginShardingMap, getFuncResultSharding(funcOp, valueIndex),
          OriginShardingType::OUTPUT, returnOperand.get(), valueIndex);
    }
  });
  // NOTE: all `ManualComputationOp`s and `ShardingConstraintOp`s will have a
  // unique source name, no matter if they aren't in the same `FuncOp`.
  int64_t sourceId = 0;
  moduleOp.walk([&](ShardingConstraintOp shardingConstraintOp) {
    saveShardingOrigins(valueToOriginShardingMap,
                        shardingConstraintOp.getSharding(),
                        OriginShardingType::CONSTRAINT,
                        shardingConstraintOp.getResult(), 0, sourceId);
    shardingConstraintOp->setAttr(
        kShardingOriginNameAttr,
        shardingOriginToString(OriginSharding{OriginShardingType::CONSTRAINT,
                                              /*index=*/0, sourceId},
                               context));
    ++sourceId;
  });
  sourceId = 0;
  moduleOp.walk([&](ManualComputationOp manualComputationOp) {
    for (auto [i, sharding] :
         llvm::enumerate(manualComputationOp.getInShardings().getShardings())) {
      // Assuming that the edges live as the only use of the block arguments.
      auto edge =
          DataFlowEdgeOp::lookup(manualComputationOp.getBody().getArgument(i));
      assert(edge);
      saveShardingOrigins(valueToOriginShardingMap, sharding,
                          OriginShardingType::MC_INPUT, edge.getResult(), i,
                          sourceId);
    }
    for (auto [i, sharding] : llvm::enumerate(
             manualComputationOp.getOutShardings().getShardings())) {
      // Assuming that the edges live as the only use of the op results.
      auto edge = DataFlowEdgeOp::lookup(manualComputationOp.getResult(i));
      assert(edge);
      saveShardingOrigins(valueToOriginShardingMap, sharding,
                          OriginShardingType::MC_OUTPUT, edge.getResult(), i,
                          sourceId);
    }
    manualComputationOp->setAttr(
        kShardingOriginNameAttr,
        StringAttr::get(context, llvm::formatv("mc_{0}", sourceId)));
    ++sourceId;
  });
}

// Sets up `funcResultToEdgesMap` for saving the edge source information
// on the `moduleOp`.
//
// The `SourceShardingHandler` will keep `funcResultToEdgesMap` up to date
// with the source sharding information on the module during the propagation
// rewrite patterns.
void prepareFuncResultToEdgesHandler(
    ModuleOp moduleOp, FuncResultToEdgesMap& funcResultToEdgesMap) {
  moduleOp.walk([&](func::FuncOp funcOp) {
    funcResultToEdgesMap[funcOp] =
        SmallVector<AxisToEdgesMap>(funcOp.getNumResults());
  });
}

OriginSharding lookUpValueOriginSharding(
    Value value, AxisRefAttr axisRef,
    const ValueToOriginShardingMap& valueToOriginShardingMap) {
  // NOTE: need to call `getShardableValue` in case the operand/result is
  // part of a `ShardableDataFlowOpInterface` and the `Value` the sharding
  // lives on is a `DataFlowEdgeOp` instead of the `edge` itself.
  const AxisToOriginShardingMap& axisToOriginSharding =
      valueToOriginShardingMap.at(getShardableValue(value));
  if (auto it = axisToOriginSharding.find(axisRef);
      it != axisToOriginSharding.end()) {
    return it->second;
  }
  // If we can't find the axis, it may mean it's been split due to a
  // (AC,)->(A, C) reshape or merged due to a (A, C)->(AC) reshape.
  // In that case, we just return the first sharding origin we find with
  // the same full axis name.
  for (auto [otherAxis, originSharding] : axisToOriginSharding) {
    if (otherAxis.contains(axisRef) || axisRef.contains(otherAxis)) {
      return originSharding;
    }
  }
  llvm_unreachable("Couldn't find sharding origin");
  return {};
}

}  // namespace

ShardingDebugMappings::ShardingDebugMappings(bool debugShardingOrigins,
                                             bool debugPropagationEdgeSharding)
    : debugShardingOrigins(debugShardingOrigins),
      debugPropagationEdgeSharding(debugPropagationEdgeSharding) {}

SourceShardingHandler::SourceShardingHandler(ShardingDebugMappings* mappings)
    : mappings(mappings) {}

void SourceShardingHandler::operator()(function_ref<void()> transform,
                                       const tracing::Action& action) {
  // NOTE: need to call `transform` first so that the `newShardingProjection`
  // is up to date.
  transform();
  if (action.getTag() != SourceShardingAction::tag) {
    return;
  }

  auto sourceShardingAction = cast<SourceShardingAction>(action);
  if (!sourceShardingAction.anyUpdated) {
    return;
  }
  FactorsToEdgeMap factorsToEdges =
      createSourceMap(sourceShardingAction.oldShardingProjection,
                      sourceShardingAction.newShardingProjection,
                      sourceShardingAction.shardingRule,
                      sourceShardingAction.mesh, propagationStep);
  propagationStep++;
  // If the new and old shardings are different, something was propagated to it.
  // Find and save it.
  auto lookUpOriginSharding = [&](EdgeNode edgeNode,
                                  AxisRefAttr axisRef) -> OriginSharding {
    switch (edgeNode.type) {
      case OPERAND:
        return lookUpValueOriginSharding(
            sourceShardingAction.operands[edgeNode.index], axisRef,
            mappings->valueToOriginShardingMap);
      case RESULT:
        return lookUpValueOriginSharding(
            sourceShardingAction.results[edgeNode.index], axisRef,
            mappings->valueToOriginShardingMap);
    }
    llvm_unreachable("unknown EdgeNode");
  };

  auto updateMappings = [&](int64_t i, AxisRefAttr axisRef,
                            PropagationEdge edge, Value value) {
    if (mappings->debugPropagationEdgeSharding) {
      if (auto funcOp = dyn_cast<func::FuncOp>(sourceShardingAction.op)) {
        mappings->funcResultToEdgesMap[funcOp][i][axisRef].push_back(edge);
      } else {
        mappings->operationToEdgesMap[sourceShardingAction.op][axisRef]
            .push_back(edge);
      }
    }
    if (mappings->debugShardingOrigins) {
      mappings->valueToOriginShardingMap[value].try_emplace(
          axisRef, lookUpOriginSharding(edge.source, axisRef));
    }
  };

  for (auto [i, operand] : llvm::enumerate(sourceShardingAction.operands)) {
    for (auto [axisRef, edge] : factorsToEdges.operands[i]) {
      updateMappings(i, axisRef, edge, operand);
    }
  }

  for (auto [i, result] : llvm::enumerate(sourceShardingAction.results)) {
    for (auto [axisRef, edge] : factorsToEdges.results[i]) {
      updateMappings(i, axisRef, edge, result);
    }
  }
}

void SourceShardingHandler::prepareHandler(ModuleOp moduleOp) {
  if (mappings->debugShardingOrigins) {
    prepareShardingOriginsHandler(moduleOp, mappings->valueToOriginShardingMap);
  }
  if (mappings->debugPropagationEdgeSharding) {
    prepareFuncResultToEdgesHandler(moduleOp, mappings->funcResultToEdgesMap);
  }
  if (mappings->debugShardingOrigins ||
      mappings->debugPropagationEdgeSharding) {
    moduleOp->getContext()->registerActionHandler(*this);
  }
}

void SourceShardingHandler::saveOnModule(ModuleOp moduleOp) {
  MLIRContext* context = moduleOp.getContext();
  if (mappings->debugShardingOrigins) {
    saveShardingOriginsOnModule(context, mappings->valueToOriginShardingMap);
    overrideOriginsToSelf(moduleOp);
  }
  if (mappings->debugPropagationEdgeSharding) {
    saveEdgesOnModule(context, mappings->operationToEdgesMap);
    moduleOp.walk([&](func::FuncOp funcOp) {
      saveEdgesOnFuncResults(funcOp, mappings->funcResultToEdgesMap);
    });
  }
}

namespace {

// Looks for the debug info dictionary on the `dataFlowEdgeOp` called
// `debugAttrName` and pushes it back to the `debugInfoDict`. If the dictionary
// doesn't exist, pushes an empty dictionary.
void pushBackToDebugInfoDict(DataFlowEdgeOp dataFlowEdgeOp,
                             StringRef debugAttrName,
                             SmallVector<Attribute>& debugInfoDict,
                             IRRewriter& rewriter) {
  assert(dataFlowEdgeOp);
  if (auto edgeDebugInfo =
          dataFlowEdgeOp->getAttrOfType<DictionaryAttr>(debugAttrName)) {
    debugInfoDict.push_back(edgeDebugInfo);
  } else {
    debugInfoDict.push_back(rewriter.getDictionaryAttr({}));
  }
}

}  // namespace

void saveDebugInfoDictsFromDataFlowEdges(ValueRange edgeOwners, Operation* op,
                                         bool sinkDebugShardingOrigins,
                                         bool sinkDebugPropagationEdgeSharding,
                                         EdgeNodeType edgeNodeType,
                                         IRRewriter& rewriter) {
  if (!sinkDebugShardingOrigins && !sinkDebugPropagationEdgeSharding) {
    return;
  }
  SmallVector<Attribute> originShardingDicts;
  if (sinkDebugShardingOrigins) {
    originShardingDicts.reserve(edgeOwners.size());
  }
  SmallVector<Attribute> propagationEdgeDicts;
  if (sinkDebugPropagationEdgeSharding) {
    propagationEdgeDicts.reserve(edgeOwners.size());
  }
  for (Value edgeOwner : edgeOwners) {
    if (auto dataFlowEdgeOp = DataFlowEdgeOp::lookup(edgeOwner)) {
      if (sinkDebugShardingOrigins) {
        pushBackToDebugInfoDict(dataFlowEdgeOp, kShardingOriginsAttr,
                                originShardingDicts, rewriter);
      }
      if (sinkDebugPropagationEdgeSharding) {
        pushBackToDebugInfoDict(dataFlowEdgeOp, kPropagationEdgesAttr,
                                propagationEdgeDicts, rewriter);
      }
    }
  }

  if (sinkDebugShardingOrigins) {
    op->setAttr(edgeNodeType == EdgeNodeType::OPERAND
                    ? kBlockArgShardingOriginsAttr
                    : kResultShardingOriginsAttr,
                rewriter.getArrayAttr(originShardingDicts));
  }
  if (sinkDebugPropagationEdgeSharding) {
    op->setAttr(edgeNodeType == EdgeNodeType::OPERAND
                    ? kBlockArgPropagationEdgesAttr
                    : kResultPropagationEdgesAttr,
                rewriter.getArrayAttr(propagationEdgeDicts));
  }
}

}  // namespace sdy
}  // namespace mlir
