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
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/data_flow_utils.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/sharding_projection.h"

namespace mlir {
namespace sdy {

namespace {

// The map from factor to edge source for operands and results.
struct FactorsToEdgeSourceMap {
  llvm::SmallVector<AxisToEdgeSourceMap> operands, results;
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
FactorsToEdgeSourceMap createSourceMap(
    const ShardingProjection& oldShardingProjection,
    const ShardingProjection& newShardingProjection,
    OpShardingRuleAttr shardingRule, MeshAttr mesh) {
  FactorsToEdgeSourceMap axisToEdgeSourceMap{
      llvm::SmallVector<AxisToEdgeSourceMap>(
          oldShardingProjection.getNumOperands(), AxisToEdgeSourceMap()),
      llvm::SmallVector<AxisToEdgeSourceMap>(
          oldShardingProjection.getNumResults(), AxisToEdgeSourceMap())};

  // Saves the `axisRefs` to the specified `valueSourceMap` of
  // `axisToEdgeSourceMap`.
  auto saveEdgeSources = [&](ArrayRef<AxisRefAttr> newAxisRefs,
                             ArrayRef<AxisRefAttr> oldAxisRefs,
                             EdgeSourceType type, int64_t sourceIndex,
                             AxisToEdgeSourceMap& valueSourceMap) {
    // To avoid iterating over all the new axes, only compare the very last old
    // axis (since there could have been a sub-axis update) and then the
    // trailing new axes.
    int64_t oldAxisIndex = oldAxisRefs.size() - 1;
    if (!oldAxisRefs.empty() &&
        oldAxisRefs[oldAxisIndex] != newAxisRefs[oldAxisIndex]) {
      valueSourceMap.try_emplace(newAxisRefs[oldAxisIndex],
                                 EdgeSource{type, sourceIndex});
    }
    for (AxisRefAttr axisRef : newAxisRefs.drop_front(oldAxisRefs.size())) {
      valueSourceMap.try_emplace(axisRef, EdgeSource{type, sourceIndex});
    }
  };

  MLIRContext* context = mesh.getContext();
  ArrayRef<int64_t> factorSizes = shardingRule.getFactorSizes();
  auto visitValue =
      [&](const TensorFactorShardings& oldValue,
          const TensorFactorShardings& newValue, int64_t valueIndex,
          TensorMappingAttr tensorMapping,
          llvm::SmallVector<AxisToEdgeSourceMap>& valueSourceMap) {
        DenseSet<AxisRefAttr> oldAxes;
        for (const auto& [_, oldFactorSharding] :
             oldValue.factorIndexToSharding) {
          oldAxes.insert(oldFactorSharding.axisRefs.begin(),
                         oldFactorSharding.axisRefs.end());
        }
        for (const auto& [factorIndex, oldFactorSharding] :
             oldValue.factorIndexToSharding) {
          const FactorSharding& newFactorSharding =
              newValue.factorIndexToSharding.at(factorIndex);
          if (oldFactorSharding.axisRefs == newFactorSharding.axisRefs) {
            continue;
          }
          SmallVector<AxisRefAttr> newlyIntroducedAxes;
          // If multiple sub axes can be merged due to a dimension sharding
          // having multiple factors, each sharded on a sub axis, make sure we
          // only save the merged one. This can happen during an
          // `(A, B) -> (AB,)` reshape.
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
                  // We need to avoid any axes that already existed
                  // in the old sharding, but aren't in the new
                  // projection as the conflicted. E.g. for a
                  // contracting dim matmul, if both the LHS/RHS are
                  // sharded on the same axis on their respective
                  // non-contracting dims, the dimension sharding
                  // will contain the conflicting axes, but the
                  // factor sharding will not. And we don't want this
                  // axis as it isn't a newly introduced axis.
                  for (AxisRefAttr newAxisRef : newFactorSharding.axisRefs) {
                    if (newAxisRef.prefixOf(axisRef)) {
                      return true;
                    }
                  }
                  return false;
                });
          }
          // This factor sharding has changed, let's find who changed it.
          if (std::optional<int64_t> operandSource =
                  findNewAxisRefMatch(newFactorSharding.axisRefs, factorIndex,
                                      oldShardingProjection.getOperands())) {
            saveEdgeSources(newlyIntroducedAxes, oldFactorSharding.axisRefs,
                            EdgeSourceType::OPERAND, *operandSource,
                            valueSourceMap[valueIndex]);
          } else if (std::optional<int64_t> resultSource = findNewAxisRefMatch(
                         newFactorSharding.axisRefs, factorIndex,
                         oldShardingProjection.getResults())) {
            saveEdgeSources(newlyIntroducedAxes, oldFactorSharding.axisRefs,
                            EdgeSourceType::RESULT, *resultSource,
                            valueSourceMap[valueIndex]);
          }
        }
      };

  for (auto [i, packedOperands] :
       llvm::enumerate(llvm::zip_equal(oldShardingProjection.getOperands(),
                                       newShardingProjection.getOperands()))) {
    auto [oldOperand, newOperand] = packedOperands;
    visitValue(oldOperand, newOperand, i, shardingRule.getOperandMapping(i),
               axisToEdgeSourceMap.operands);
  }
  for (auto [i, packedResults] :
       llvm::enumerate(llvm::zip_equal(oldShardingProjection.getResults(),
                                       newShardingProjection.getResults()))) {
    auto [oldResult, newResult] = packedResults;
    visitValue(oldResult, newResult, i, shardingRule.getResultMapping(i),
               axisToEdgeSourceMap.results);
  }

  return axisToEdgeSourceMap;
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

// Create a list of entries from the `axisToOriginSharding` map to save as a
// `DictionaryAttr`.
SmallVector<NamedAttribute> createOriginShardingEntries(
    const AxisToOriginShardingMap& axisToOriginSharding, MLIRContext* context) {
  SmallVector<NamedAttribute> entries;
  entries.reserve(axisToOriginSharding.size());
  for (const auto& [axisRef, shardingOrigin] : axisToOriginSharding) {
    std::string axisRefString = axisRef.toString();
    // Avoid printing the string with escaping quotes, aka "\22".
    axisRefString.erase(remove(axisRefString.begin(), axisRefString.end(), '"'),
                        axisRefString.end());
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

// Saves the originating sharding debug information on the `moduleOp`.
void saveShardingOriginsOnModule(ModuleOp moduleOp,
                                 ShardingDebugMappings* mappings) {
  MLIRContext* context = moduleOp.getContext();
  Builder builder(context);
  for (auto [value, axisToOriginSharding] :
       mappings->valueToOriginShardingMap) {
    Operation* owningOp = getOwningOp(value);

    func::FuncOp funcOp = getEnclosingOfType<func::FuncOp>(owningOp);

    // TODO(bartchr): Swap the map to store `ValueOrFuncResult` to avoid having
    // to do this terminator finding logic just to set the func result attr.
    OpOperand* terminatorOperand = nullptr;
    ArrayRef<OpOperand> terminatorOperands =
        getBodyTerminator(funcOp)->getOpOperands();
    if (auto it = llvm::find_if(value.getUses(),
                                [&](const OpOperand& use) {
                                  return llvm::is_contained(terminatorOperands,
                                                            use);
                                });
        it != value.getUses().end()) {
      terminatorOperand = it.getOperand();
    }

    SmallVector<NamedAttribute> entries =
        createOriginShardingEntries(axisToOriginSharding, context);
    if (terminatorOperand) {
      int64_t operandNumber = terminatorOperand->getOperandNumber();
      funcOp.setResultAttr(operandNumber, kShardingOriginsAttr,
                           builder.getDictionaryAttr(entries));
    }
    TypeSwitch<Operation*, void>(owningOp)
        .Case<func::FuncOp>([&, value = value](func::FuncOp funcOp) {
          funcOp.setArgAttr(cast<BlockArgument>(value).getArgNumber(),
                            kShardingOriginsAttr,
                            builder.getDictionaryAttr(entries));
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
      setOpOriginsToSelf(getDataFlowEdge(blockArg),
                         manualComputationOriginName(
                             OriginShardingType::MC_INPUT,
                             originName.getValue(), blockArg.getArgNumber()));
    }
    for (OpResult result : manualComputationOp.getResults()) {
      setOpOriginsToSelf(getDataFlowEdge(result),
                         manualComputationOriginName(
                             OriginShardingType::MC_OUTPUT,
                             originName.getValue(), result.getResultNumber()));
    }
  });
}

// Sets up the `handler` with the initial sharding origin information on
// the `moduleOp`.
// The `SourceShardingHandler` will keep `valueToEdgeSourceMap` and
// `valueToOriginShardingMap` up to date with the source sharding information
// on the module during the propagation rewrite patterns.
void prepareShardingOriginsHandler(ModuleOp moduleOp,
                                   ShardingDebugMappings* mappings) {
  MLIRContext* context = moduleOp.getContext();
  // Build the initial sharding origin map.
  // NOTE(bartchr): This assumes that we do not propagate across different
  // functions. As all func inputs/outputs have the same source name. Update
  // this if we do propagate across `FuncOp`s.
  moduleOp.walk([&](func::FuncOp funcOp) {
    for (BlockArgument arg : funcOp.getArguments()) {
      saveShardingOrigins(mappings->valueToOriginShardingMap, getSharding(arg),
                          OriginShardingType::INPUT, arg, arg.getArgNumber());
    }
    for (OpOperand& returnOperand : getBodyTerminatorOpOperands(funcOp)) {
      int64_t valueIndex = returnOperand.getOperandNumber();
      saveShardingOrigins(mappings->valueToOriginShardingMap,
                          getFuncResultSharding(funcOp, valueIndex),
                          OriginShardingType::OUTPUT, returnOperand.get(),
                          valueIndex);
    }
  });
  // NOTE: all `ManualComputationOp`s and `ShardingConstraintOp`s will have a
  // unique source name, no matter if they aren't in the same `FuncOp`.
  int64_t sourceId = 0;
  moduleOp.walk([&](ShardingConstraintOp shardingConstraintOp) {
    saveShardingOrigins(mappings->valueToOriginShardingMap,
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
      DataFlowEdgeOp edge = DataFlowEdgeOp::getDataFlowEdgeUser(
          manualComputationOp.getBody().getArgument(i));
      assert(edge);
      saveShardingOrigins(mappings->valueToOriginShardingMap, sharding,
                          OriginShardingType::MC_INPUT, edge.getResult(), i,
                          sourceId);
    }
    for (auto [i, sharding] : llvm::enumerate(
             manualComputationOp.getOutShardings().getShardings())) {
      // Assuming that the edges live as the only use of the op results.
      DataFlowEdgeOp edge =
          DataFlowEdgeOp::getDataFlowEdgeUser(manualComputationOp.getResult(i));
      assert(edge);
      saveShardingOrigins(mappings->valueToOriginShardingMap, sharding,
                          OriginShardingType::MC_OUTPUT, edge.getResult(), i,
                          sourceId);
    }
    manualComputationOp->setAttr(
        kShardingOriginNameAttr,
        StringAttr::get(context, llvm::formatv("mc_{0}", sourceId)));
    ++sourceId;
  });
}

OriginSharding lookUpValueOriginSharding(
    Value value, AxisRefAttr axisRef,
    const ValueToOriginShardingMap& valueToOriginShardingMap) {
  // NOTE: need to call `getShardableValue` in case the operand/result is
  // part of a `ShardableDataFlowOpInterface` and the `Value` the sharding
  // lives on is a `DataFlowEdgeOp` instead of the `edgeSource` itself.
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
                                             bool debugEdgeSourceSharding)
    : debugShardingOrigins(debugShardingOrigins),
      debugEdgeSourceSharding(debugEdgeSourceSharding) {}

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
  FactorsToEdgeSourceMap factorsToEdgeSources = createSourceMap(
      sourceShardingAction.oldShardingProjection,
      sourceShardingAction.newShardingProjection,
      sourceShardingAction.shardingRule, sourceShardingAction.mesh);
  // If the new and old shardings are different, something was propagated to it.
  // Find and save it.
  auto lookUpOriginSharding = [&](EdgeSource edgeSource,
                                  AxisRefAttr axisRef) -> OriginSharding {
    switch (edgeSource.type) {
      case OPERAND:
        return lookUpValueOriginSharding(
            sourceShardingAction.operands[edgeSource.index], axisRef,
            mappings->valueToOriginShardingMap);
      case RESULT:
        return lookUpValueOriginSharding(
            sourceShardingAction.results[edgeSource.index], axisRef,
            mappings->valueToOriginShardingMap);
    }
    llvm_unreachable("unknown EdgeSource");
  };

  auto updateMappings = [&](ShardingDebugMappings* mappings,
                            AxisToEdgeSourceMap axisToEdgeSource, Value value) {
    for (auto [axisRef, edgeSource] : axisToEdgeSource) {
      if (mappings->debugEdgeSourceSharding) {
        mappings->valueToEdgeSourceMap[value].try_emplace(axisRef, edgeSource);
      }
      if (mappings->debugShardingOrigins) {
        mappings->valueToOriginShardingMap[value].try_emplace(
            axisRef, lookUpOriginSharding(edgeSource, axisRef));
      }
    }
  };
  for (auto [operand, axisToEdgeSource] : llvm::zip_equal(
           sourceShardingAction.operands, factorsToEdgeSources.operands)) {
    updateMappings(mappings, axisToEdgeSource, operand);
  }
  for (auto [result, axisToEdgeSource] : llvm::zip_equal(
           sourceShardingAction.results, factorsToEdgeSources.results)) {
    updateMappings(mappings, axisToEdgeSource, result);
  }
}

void SourceShardingHandler::prepareHandler(ModuleOp moduleOp) {
  if (mappings->debugShardingOrigins) {
    prepareShardingOriginsHandler(moduleOp, mappings);
  }
  if (mappings->debugEdgeSourceSharding) {
    llvm_unreachable("edge sharding not implemented yet");
  }
  if (mappings->debugShardingOrigins || mappings->debugEdgeSourceSharding) {
    moduleOp->getContext()->registerActionHandler(*this);
  }
}

void SourceShardingHandler::saveOnModule(ModuleOp moduleOp) {
  if (mappings->debugShardingOrigins) {
    saveShardingOriginsOnModule(moduleOp, mappings);
    overrideOriginsToSelf(moduleOp);
  }
  if (mappings->debugEdgeSourceSharding) {
    llvm_unreachable("edge sharding not implemented yet");
  }
}

}  // namespace sdy
}  // namespace mlir
