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
#include <cstdint>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Action.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace mlir {
namespace sdy {

namespace {

// Saves what operand/result each axis came from to `axisToEdgeSourceMap`.
// When multiple operands/results have the same axis, the first one inserted
// into `axisToEdgeSourceMap` will be used.
// TODO(b/379280210): This algorithm is very naive and not correct when handling
// conflicts.
void saveEdgeSources(EdgeSourceType type,
                     ArrayRef<TensorShardingAttr> oldShardings,
                     AxisToEdgeSourceMap& axisToEdgeSourceMap) {
  for (auto [i, oldSharding] : llvm::enumerate(oldShardings)) {
    if (!oldSharding) {
      continue;
    }
    for (DimensionShardingAttr dimSharding : oldSharding.getDimShardings()) {
      for (AxisRefAttr axisRef : dimSharding.getAxes()) {
        axisToEdgeSourceMap.try_emplace(axisRef, EdgeSource{
            type, static_cast<int64_t>(i)});
      }
    }
  }
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
    case OriginShardingType::MC_INPUT: {
      return StringAttr::get(
          context,
          llvm::formatv("mc_{0}_input: {1}", source.sourceId, source.index));
    }
    case OriginShardingType::MC_OUTPUT: {
      return StringAttr::get(
          context,
          llvm::formatv("mc_{0}_output: {1}", source.sourceId, source.index));
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

// Saves the originating sharding debug information on the `moduleOp`.
void saveShardingOriginsOnModule(ModuleOp moduleOp,
                                 ShardingDebugMappings* mappings) {
  MLIRContext* context = moduleOp.getContext();
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
      funcOp.setResultAttr(operandNumber, kOriginShardingAttr,
                           DictionaryAttr::get(context, entries));
    }
    TypeSwitch<Operation*, void>(owningOp)
        .Case<func::FuncOp>([&](func::FuncOp funcOp) {
          funcOp.setArgAttr(cast<BlockArgument>(value).getArgNumber(),
                            kOriginShardingAttr,
                            DictionaryAttr::get(context, entries));
        })
        .Default([&](Operation* op) {
          op->setAttr(kOriginShardingAttr,
                      DictionaryAttr::get(context, entries));
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

// Creates a new dictionary for `kOriginShardingAttr` on the function inputs
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

// For `ShardingConstraintOp`s and `FuncOp` inputs/outputs, instead of saying
// that the existing axes at the start came from `constraint_0` or `input: 0`,
// we say that they came from `self`. Then it's not circular/clearer to
// understand.
void overrideOriginsToSelf(ModuleOp moduleOp) {
  MLIRContext* context = moduleOp.getContext();
  moduleOp.walk([&](ShardingConstraintOp shardingConstraintOp) {
    if (auto dictAttr = shardingConstraintOp->getAttrOfType<DictionaryAttr>(
            kOriginShardingAttr)) {
      auto originName = shardingConstraintOp->getAttrOfType<StringAttr>(
          kOriginShardingNameAttr);
      SmallVector<NamedAttribute> entries(dictAttr.getValue());
      for (auto& entry : entries) {
        if (cast<StringAttr>(entry.getValue()) == originName) {
          entry =
              NamedAttribute(entry.getName(), StringAttr::get(context, "self"));
        }
      }
      shardingConstraintOp->setAttr(kOriginShardingAttr,
                                    DictionaryAttr::get(context, entries));
    }
  });
  moduleOp.walk([&](func::FuncOp funcOp) {
    for (int64_t valueIndex = 0; valueIndex < funcOp.getNumArguments();
         ++valueIndex) {
      funcOp.setArgAttr(
          valueIndex, kOriginShardingAttr,
          convertFuncOriginsToSelf(valueIndex, OriginShardingType::INPUT,
                                   funcOp.getArgAttrOfType<DictionaryAttr>(
                                       valueIndex, kOriginShardingAttr)));
    }
    for (int64_t valueIndex = 0; valueIndex < funcOp.getNumResults();
         ++valueIndex) {
      funcOp.setResultAttr(
          valueIndex, kOriginShardingAttr,
          convertFuncOriginsToSelf(valueIndex, OriginShardingType::OUTPUT,
                                   funcOp.getResultAttrOfType<DictionaryAttr>(
                                       valueIndex, kOriginShardingAttr)));
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
        kOriginShardingNameAttr,
        shardingOriginToString(OriginSharding{OriginShardingType::CONSTRAINT,
                                              /*index=*/0, sourceId},
                               context));
    ++sourceId;
  });
  sourceId = 0;
  moduleOp.walk([&](ManualComputationOp manualComputationOp) {
    for (auto [i, sharding] :
         llvm::enumerate(manualComputationOp.getInShardings().getShardings())) {
      saveShardingOrigins(mappings->valueToOriginShardingMap, sharding,
                          OriginShardingType::MC_INPUT,
                          manualComputationOp.getBody().getArgument(i), i,
                          sourceId);
    }
    for (auto [i, sharding] : llvm::enumerate(
             manualComputationOp.getOutShardings().getShardings())) {
      saveShardingOrigins(mappings->valueToOriginShardingMap, sharding,
                          OriginShardingType::MC_OUTPUT,
                          manualComputationOp.getResult(i), i, sourceId);
    }
    manualComputationOp->setAttr(
        kOriginShardingNameAttr,
        StringAttr::get(context, llvm::formatv("mc_{0}", sourceId)));
    ++sourceId;
  });
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
  transform();
  if (action.getTag() != SourceShardingAction::tag) {
    return;
  }

  auto sourceShardingAction = cast<SourceShardingAction>(action);
  AxisToEdgeSourceMap axisToEdgeSourceMap;
  saveEdgeSources(OPERAND, sourceShardingAction.oldOperandShardings,
                  axisToEdgeSourceMap);
  saveEdgeSources(RESULT, sourceShardingAction.oldResultShardings,
                  axisToEdgeSourceMap);

  // If the new and old shardings are different, something was propagated to it.
  // Find and save it.
  auto lookUpOriginSharding = [&](EdgeSource edgeSource,
                                  AxisRefAttr axisRef) -> OriginSharding {
    switch (edgeSource.type) {
      case OPERAND:
        return mappings->valueToOriginShardingMap
            .at(sourceShardingAction.operands[edgeSource.index])
            .at(axisRef);
      case RESULT:
        return mappings->valueToOriginShardingMap
            .at(sourceShardingAction.results[edgeSource.index])
            .at(axisRef);
    }
    llvm_unreachable("unknown EdgeSource");
  };
  auto updateEdgeMap = [&](AxisRefAttr newAxisRef, Value value) {
    EdgeSource source = axisToEdgeSourceMap.at(newAxisRef);
    if (mappings->debugEdgeSourceSharding) {
      mappings->valueToEdgeSourceMap[value][newAxisRef] = source;
    }
    if (mappings->debugShardingOrigins) {
      mappings->valueToOriginShardingMap[value][newAxisRef] =
          lookUpOriginSharding(source, newAxisRef);
    }
  };
  auto setValueSources = [&](ValueRange values,
                             ArrayRef<TensorShardingAttr> oldShardings) {
    for (auto [value, oldShardingRef, newSharding] :
         llvm::zip_equal(values, oldShardings, getShardings(values))) {
      // Skip if there is no sharding or the sharding didn't change.
      if (!newSharding || oldShardingRef == newSharding) {
        continue;
      }
      TensorShardingAttr oldSharding = oldShardingRef;
      if (!oldSharding) {
        oldSharding = TensorShardingAttr::getFullyOpenLike(newSharding);
      }
      for (auto [oldDimSharding, newDimSharding] : llvm::zip_equal(
               oldSharding.getDimShardings(), newSharding.getDimShardings())) {
        if (oldDimSharding == newDimSharding) {
          continue;
        }
        const int oldDimShardingSize = oldDimSharding.getAxes().size();
        ArrayRef<AxisRefAttr> newDimAxes = newDimSharding.getAxes();
        // Check in case the new axis introduced was an update to a sub axis.
        // E.g. going from "x":(1)2 to "x":(1)4.
        // To grab the possibly updated sub axis, look at the very last axis in
        // the old sharding (if it exists).
        if (!oldDimSharding.emptyAxes()) {
          AxisRefAttr possibleNewSubAxis = newDimAxes[oldDimShardingSize - 1];
          // Only save it when:
          // 1. There is an axis in the old sharding.
          // 2. The new axis in the same place is different, aka both are not
          //    just "x" and "x".
          // Then it must be a new sub axis.
          if (oldDimSharding.getAxes().back() != possibleNewSubAxis) {
            AxisRefAttr oldLastAxis = oldDimSharding.getAxes().back();
            if (mappings->debugEdgeSourceSharding) {
              mappings->valueToEdgeSourceMap[value].erase(oldLastAxis);
            }
            if (mappings->debugShardingOrigins) {
              mappings->valueToOriginShardingMap[value].erase(oldLastAxis);
            }
            updateEdgeMap(possibleNewSubAxis, value);
          }
        }
        // Skip any pre-existing axes in the old sharding.
        for (AxisRefAttr newAxisRef :
             newDimAxes.drop_front(oldDimShardingSize)) {
          updateEdgeMap(newAxisRef, value);
        }
      }
    }
  };
  setValueSources(sourceShardingAction.operands,
                  sourceShardingAction.oldOperandShardings);
  setValueSources(sourceShardingAction.results,
                  sourceShardingAction.oldResultShardings);
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
