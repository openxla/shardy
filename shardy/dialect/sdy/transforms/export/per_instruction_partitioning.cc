/* Copyright 2026 The Shardy Authors.

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

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_PERINSTRUCTIONPARTITIONINGPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

// =============================================================================
// Filter & Selection Helpers
// =============================================================================

struct FilterConfig {
  int64_t selectLow = -1;
  int64_t selectHigh = -1;
  std::string targetFunc;
  SmallVector<std::string> opNames;
  bool selectAll = false;
};

// Returns true if op has at least one operand or result with a
// non-fully-replicated sharding.
bool hasAnyShardedValue(Operation* op) {
  auto isSharded = [](Value v) {
    TensorShardingAttr s = getSharding(v);
    return s && !s.isFullyReplicated();
  };
  return llvm::any_of(op->getResults(), isSharded) ||
         llvm::any_of(op->getOperands(), isSharded);
}

// Returns the unique user that is a sdy.all_reduce for a stablehlo.convolution
// that produces a result with unreduced_axes.
sdy::AllReduceOp getTrailingConvAllReduce(Operation* op) {
  auto convOp = dyn_cast<stablehlo::ConvolutionOp>(op);
  if (!convOp || !convOp.getResult().hasOneUse()) {
    return nullptr;
  }
  TensorShardingAttr sharding = getSharding(convOp.getResult());
  if (!sharding || sharding.getUnreducedAxes().empty()) {
    return nullptr;
  }
  return dyn_cast<sdy::AllReduceOp>(*convOp.getResult().getUsers().begin());
}

// Parses a filter string that controls which operations should be partitioned.
//
// The filter string format is a comma-separated list of tokens in any order:
//    - selectLow=<N>: Only partition operations with sequence ID >= N.
//    - selectHigh=<N>: Only partition operations with sequence ID <= N.
//    - func=<name>: Subroutine name to select from.
//    - <op_name>: An operation matches if its MLIR name contains any of
//      these substrings (e.g. "dot", "add", "convolution").
//
// If the filter string is empty or omitted, all eligible candidate operations
// are partitioned.
FilterConfig parseFilter(StringRef filterStr) {
  FilterConfig config;
  filterStr = filterStr.trim();
  if (filterStr.empty()) {
    config.selectAll = true;
    return config;
  }

  if ((filterStr.front() == '\'' && filterStr.back() == '\'') ||
      (filterStr.front() == '"' && filterStr.back() == '"')) {
    filterStr = filterStr.drop_front().drop_back().trim();
  }

  SmallVector<StringRef> rawTokens;
  filterStr.split(rawTokens, ',');
  for (StringRef raw : rawTokens) {
    StringRef token = raw.trim();
    if (token.empty()) {
      continue;
    }

    if (token.consume_front("selectLow=")) {
      int64_t val;
      if (!token.trim().getAsInteger(10, val)) {
        config.selectLow = val;
      }
    } else if (token.consume_front("selectHigh=")) {
      int64_t val;
      if (!token.trim().getAsInteger(10, val)) {
        config.selectHigh = val;
      }
    } else if (token.consume_front("function=") ||
               token.consume_front("func=")) {
      config.targetFunc = token.trim().str();
    } else {
      config.opNames.push_back(token.str());
    }
  }

  if (config.opNames.empty() && config.selectLow < 0 && config.selectHigh < 0) {
    config.selectAll = true;
  }

  return config;
}

bool matchesFilter(Operation* op, int64_t seqId, const FilterConfig& config) {
  if (config.selectAll) {
    return true;
  }
  if (config.selectLow >= 0 && seqId < config.selectLow) {
    return false;
  }
  if (config.selectHigh >= 0 && seqId > config.selectHigh) {
    return false;
  }
  if (config.opNames.empty()) {
    return true;
  }
  StringRef opName = op->getName().getStringRef();
  return llvm::any_of(config.opNames, [&](const std::string& name) {
    return opName.contains(name);
  });
}

// =============================================================================
// Live-In/Live-Out & Callee Analysis Helpers
// =============================================================================

// Collects all external live-in SSA values (direct operands followed by unique
// implicit captures inside regions) for an operation.
void collectLiveInValues(Operation* op, SmallVector<Value>& liveInValues,
                         llvm::SetVector<Value>& implicitCaptures) {
  for (Value operand : op->getOperands()) {
    liveInValues.push_back(operand);
  }

  if (op->getNumRegions() > 0) {
    llvm::SetVector<Value> usedAbove;
    mlir::getUsedValuesDefinedAbove(op->getRegions(), usedAbove);
    for (Value val : usedAbove) {
      if (implicitCaptures.insert(val)) {
        liveInValues.push_back(val);
      }
    }
  }
}

void collectSymbolsFromAttr(Attribute attr,
                            SmallVectorImpl<StringRef>& symbols) {
  if (auto symRef = dyn_cast<SymbolRefAttr>(attr)) {
    symbols.push_back(symRef.getRootReference().getValue());
  } else if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute elem : arrayAttr) {
      collectSymbolsFromAttr(elem, symbols);
    }
  } else if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute named : dictAttr) {
      collectSymbolsFromAttr(named.getValue(), symbols);
    }
  }
}

// Collects all symbol-referenced func::FuncOp callees transitively invoked by
// op or its nested regions.
void collectTransitiveCallees(Operation* op, const SymbolTable& symbolTable,
                              SmallVector<func::FuncOp>& callees) {
  llvm::DenseSet<Operation*> visited;
  SmallVector<Operation*> worklist{op};
  while (!worklist.empty()) {
    Operation* curr = worklist.pop_back_val();
    curr->walk([&](Operation* nested) {
      SmallVector<StringRef> symbols;
      for (NamedAttribute attr : nested->getAttrs()) {
        collectSymbolsFromAttr(attr.getValue(), symbols);
      }
      for (StringRef sym : symbols) {
        if (auto funcOp = symbolTable.lookup<func::FuncOp>(sym)) {
          if (visited.insert(funcOp).second) {
            callees.push_back(funcOp);
            worklist.push_back(funcOp);
          }
        }
      }
    });
  }
}

// =============================================================================
// Mesh & Sharding Analysis Helpers
// =============================================================================

// Represents the sharding and manual axes metadata for an instruction.
struct InstructionShardingInfo {
  SmallVector<TensorShardingAttr> inShardings;
  SmallVector<TensorShardingAttr> outShardings;
  SmallVector<StringAttr> manualAxesList;
  llvm::DenseSet<StringRef> manualAxesSet;
};

// Returns the common mesh attribute or symbol reference attached to any
// live-in value or result of the operation.
Attribute getTargetMeshOrRef(ArrayRef<Value> liveInValues, Operation* op,
                             const SymbolTable& symbolTable) {
  SmallVector<TensorShardingAttr> inShardings;
  for (Value val : liveInValues) {
    inShardings.push_back(getSharding(val));
  }
  SmallVector<TensorShardingAttr> outShardings;
  for (Value val : op->getResults()) {
    outShardings.push_back(getSharding(val));
  }
  return getCommonMeshOrRef(inShardings, outShardings, symbolTable);
}

// Collects live-in and result shardings (defaulting to fully replicated if
// missing) and all unique manual axes used across the instruction.
InstructionShardingInfo getInstructionShardingInfo(Operation* op,
                                                   ArrayRef<Value> liveInValues,
                                                   FlatSymbolRefAttr meshSym,
                                                   Attribute targetMesh) {
  MLIRContext* ctx = op->getContext();
  InstructionShardingInfo info;

  auto getShardings = [&](ValueRange values, bool clearUnreducedAxes) {
    SmallVector<TensorShardingAttr> shardings;
    shardings.reserve(values.size());
    for (Value val : values) {
      TensorShardingAttr sharding =
          getOrCreateSharding(val, targetMesh, /*closedIfMissing=*/true);
      if (clearUnreducedAxes ||
          (meshSym && isa<MeshAttr>(sharding.getMeshOrRef()))) {
        sharding = TensorShardingAttr::get(
            ctx, targetMesh, sharding.getDimShardings(),
            sharding.getReplicatedAxes(),
            clearUnreducedAxes ? ArrayRef<AxisRefAttr>{}
                               : sharding.getUnreducedAxes(),
            sharding.getReductionOp());
      }
      shardings.push_back(sharding);
    }
    return shardings;
  };

  info.inShardings = getShardings(liveInValues, /*clearUnreducedAxes=*/false);
  info.outShardings =
      getShardings(op->getResults(), /*clearUnreducedAxes=*/false);

  MeshAttr meshAttr = getMeshOrLookup(op, targetMesh);

  if (meshAttr) {
    for (MeshAxisAttr axis : meshAttr.getAxes()) {
      info.manualAxesList.push_back(StringAttr::get(ctx, axis.getName()));
      info.manualAxesSet.insert(axis.getName());
    }
  } else {
    llvm::SmallDenseSet<StringAttr> seenAxes;
    auto collectAxes = [&](TensorShardingAttr sharding) {
      if (!sharding) {
        return;
      }
      sharding.forEachAxisRef([&](AxisRefAttr axisRef) {
        StringAttr axisName = StringAttr::get(ctx, axisRef.getName());
        if (seenAxes.insert(axisName).second) {
          info.manualAxesList.push_back(axisName);
          info.manualAxesSet.insert(axisRef.getName());
        }
      });
    };

    for (TensorShardingAttr sharding : info.inShardings) {
      collectAxes(sharding);
    }
    for (TensorShardingAttr sharding : info.outShardings) {
      collectAxes(sharding);
    }
  }

  return info;
}

// =============================================================================
// Divisible Padding & Outlined Partitioning Helpers
// =============================================================================

// Computes the per-dimension divisor for a specific sharding.
SmallVector<int64_t> getDivisorsForSharding(
    TensorShardingAttr sharding, int64_t rank,
    const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable) {
  SmallVector<int64_t> divisors(rank, 1);
  if (!sharding || sharding.isFullyReplicated()) {
    return divisors;
  }
  MeshAttr mesh = sharding.getMesh(symbolTable);
  if (!mesh) {
    return divisors;
  }
  for (auto [dim, dimSharding] : llvm::enumerate(sharding.getDimShardings())) {
    if (dim >= rank) {
      break;
    }
    for (AxisRefAttr axisRef : dimSharding.getAxes()) {
      if (shardingInfo.manualAxesSet.contains(axisRef.getName())) {
        int64_t axisSize = axisRef.getSize(mesh);
        divisors[dim] *= axisSize;
      }
    }
  }
  return divisors;
}

// Computes the divisible padded type for a ranked tensor given per-dimension
// divisors.
Type getPaddedTypeWithDivisors(Type type, ArrayRef<int64_t> divisors) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType) {
    return type;
  }
  SmallVector<int64_t> newShape;
  bool changed = false;
  for (size_t d = 0; d < rankedType.getShape().size(); ++d) {
    int64_t dimSize = rankedType.getDimSize(d);
    int64_t divisor = d < divisors.size() ? divisors[d] : 1;
    if (dimSize == ShapedType::kDynamic || divisor <= 1) {
      newShape.push_back(dimSize);
      continue;
    }
    int64_t paddedDim = llvm::alignTo(dimSize, divisor);
    newShape.push_back(paddedDim);
    if (paddedDim != dimSize) {
      changed = true;
    }
  }
  return changed ? RankedTensorType::get(newShape, rankedType.getElementType())
                 : type;
}

// Computes a unified padded type exclusively for sdy.reshard operations.
//
// Unlike other operations where operand and result types are computed
// independently per-tensor using getDivisiblePaddedType, sdy.reshard has the
// SameOperandsAndResultType MLIR trait requiring operand and result types to
// match identically. To satisfy this constraint while ensuring all assigned
// mesh axes remain divisible, each dimension is padded to the least common
// multiple (LCM) of its input and output sharding divisors.
//
// For all other operations, this returns nullptr so that operand and result
// types are padded independently according to their own sharding attributes.
RankedTensorType getUnifiedPaddedTypeForReshard(
    Operation* op, const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable) {
  auto reshardOp = dyn_cast<ReshardOp>(op);
  if (!reshardOp) {
    return nullptr;
  }
  auto rankedType = dyn_cast<RankedTensorType>(reshardOp.getType());
  if (!rankedType) {
    return nullptr;
  }
  SmallVector<int64_t> inDivisors =
      getDivisorsForSharding(shardingInfo.inShardings[0], rankedType.getRank(),
                             shardingInfo, symbolTable);
  SmallVector<int64_t> outDivisors =
      getDivisorsForSharding(shardingInfo.outShardings[0], rankedType.getRank(),
                             shardingInfo, symbolTable);
  SmallVector<int64_t> unifiedDivisors(rankedType.getRank(), 1);
  for (int64_t d = 0; d < rankedType.getRank(); ++d) {
    unifiedDivisors[d] = std::lcm(inDivisors[d], outDivisors[d]);
  }
  return cast<RankedTensorType>(
      getPaddedTypeWithDivisors(rankedType, unifiedDivisors));
}

// Computes the padded argument types for manual computation.
// For sdy.reshard, uses the unified LCM padded type. For all other ops, pads
// each operand independently using getDivisiblePaddedType.
SmallVector<Type> computePaddedOperandTypes(
    Operation* op, ArrayRef<Value> liveInValues,
    const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable) {
  if (RankedTensorType unifiedType =
          getUnifiedPaddedTypeForReshard(op, shardingInfo, symbolTable)) {
    return {unifiedType};
  }
  SmallVector<Type> paddedArgTypes;
  for (auto [operand, sharding] :
       llvm::zip_equal(liveInValues, shardingInfo.inShardings)) {
    paddedArgTypes.push_back(getDivisiblePaddedType(
        operand.getType(), sharding, symbolTable, &shardingInfo.manualAxesSet));
  }
  return paddedArgTypes;
}

// Computes the padded result types for manual computation.
// For sdy.reshard, uses the unified LCM padded type. For all other ops, pads
// each result independently using getDivisiblePaddedType.
SmallVector<Type> computePaddedResultTypes(
    Operation* op, const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable) {
  if (RankedTensorType unifiedType =
          getUnifiedPaddedTypeForReshard(op, shardingInfo, symbolTable)) {
    return {unifiedType};
  }
  SmallVector<Type> paddedResultTypes;
  for (auto [result, sharding] :
       llvm::zip_equal(op->getResults(), shardingInfo.outShardings)) {
    paddedResultTypes.push_back(getDivisiblePaddedType(
        result.getType(), sharding, symbolTable, &shardingInfo.manualAxesSet));
  }
  return paddedResultTypes;
}

// Pads indivisible live-in values in the parent module before manual
// computation.
SmallVector<Value> padIndivisibleLiveInValues(
    Operation* op, ArrayRef<Value> liveInValues,
    const InstructionShardingInfo& shardingInfo, ArrayRef<Type> paddedArgTypes,
    IRRewriter& rewriter) {
  SmallVector<Value> manualOperands;
  for (auto [operand, paddedType, sharding] : llvm::zip_equal(
           liveInValues, paddedArgTypes, shardingInfo.inShardings)) {
    manualOperands.push_back(padHighSideToType(
        rewriter, op->getLoc(), operand, paddedType, sharding,
        /*paddingValue=*/nullptr, /*allowSlicePeephole=*/true));
  }
  return manualOperands;
}

// Outlines the target instruction and its mesh/callee dependencies into an
// ephemeral module containing a single private func::FuncOp.
LogicalResult outlineInstruction(Operation* op,
                                 const llvm::SetVector<Value>& implicitCaptures,
                                 ArrayRef<func::FuncOp> callees,
                                 const InstructionShardingInfo& shardingInfo,
                                 ArrayRef<Type> paddedArgTypes,
                                 ArrayRef<Type> paddedResultTypes,
                                 OwningOpRef<ModuleOp>& tempModule,
                                 func::FuncOp& outlinedFunc,
                                 sdy::AllReduceOp trailingAllReduce = nullptr) {
  MLIRContext* ctx = op->getContext();
  tempModule = ModuleOp::create(op->getLoc());
  IRRewriter tempRewriter(ctx);
  tempRewriter.setInsertionPointToStart(tempModule->getBody());

  for (MeshOp meshOp : op->getParentOfType<ModuleOp>().getOps<MeshOp>()) {
    tempRewriter.clone(*meshOp);
  }

  for (func::FuncOp callee : callees) {
    tempRewriter.clone(*callee);
  }

  FunctionType funcType =
      FunctionType::get(ctx, paddedArgTypes, paddedResultTypes);

  outlinedFunc = func::FuncOp::create(
      tempRewriter, op->getLoc(), "outlined_op", funcType,
      tempRewriter.getStringAttr("private"),
      /*argAttrs=*/ArrayAttr(), /*resultAttrs=*/ArrayAttr());

  for (size_t i = 0; i < shardingInfo.inShardings.size(); ++i) {
    if (shardingInfo.inShardings[i]) {
      outlinedFunc.setArgAttr(i, kShardingAttr, shardingInfo.inShardings[i]);
    }
  }
  for (size_t i = 0; i < shardingInfo.outShardings.size(); ++i) {
    if (shardingInfo.outShardings[i]) {
      setFuncResultSharding(outlinedFunc, i, shardingInfo.outShardings[i]);
    }
  }

  // We slice the padded operands back to their original unpadded types, so that
  // the outlined op preserves its shapes. We then pad the outlined op results
  // to match the function results which are required to be divisible.
  //
  // We do the above to all ops except for sdy.reshard, because sdy.reshard's
  // padded input type is computed to be jointly divisible by both its input and
  // output shardings; slicing it back to the unpadded type along a sharded
  // dimension can introduce a non-communication-free slice that fails
  // pad-for-divisibility.

  // Replicates the given dimension in the given sharding.
  auto replicateDimension = [&](TensorShardingAttr sharding,
                                int64_t dim) -> TensorShardingAttr {
    if (!sharding || dim < 0 || dim >= sharding.getRank() ||
        sharding.getDimShardings()[dim].getAxes().empty()) {
      return sharding;
    }
    MLIRContext* ctx = sharding.getContext();
    SmallVector<DimensionShardingAttr> dimShardings(
        sharding.getDimShardings().begin(), sharding.getDimShardings().end());
    SmallVector<AxisRefAttr> replicatedAxes(
        sharding.getReplicatedAxes().begin(),
        sharding.getReplicatedAxes().end());
    replicatedAxes.append(dimShardings[dim].getAxes().begin(),
                          dimShardings[dim].getAxes().end());
    if (MeshAttr mesh = sharding.getMesh(op)) {
      sdy::sortAndMergeAxes(replicatedAxes, mesh);
    }
    dimShardings[dim] =
        DimensionShardingAttr::get(ctx, /*axes=*/{}, /*isClosed=*/true);
    return TensorShardingAttr::get(ctx, sharding.getMeshOrRef(), dimShardings,
                                   replicatedAxes, sharding.getUnreducedAxes());
  };

  Block* block = outlinedFunc.addEntryBlock();
  tempRewriter.setInsertionPointToStart(block);
  bool isReshard = isa<ReshardOp>(op);

  auto getBlockArgOrSliced = [&](size_t argIdx, Type origType) -> Value {
    Value arg = block->getArgument(argIdx);
    TensorShardingAttr operandSharding = shardingInfo.inShardings[argIdx];
    if (!isReshard && arg.getType() != origType) {
      arg = sliceHighSideToType(tempRewriter, op->getLoc(), arg, origType,
                                operandSharding);
    }
    if (auto concatOp = dyn_cast<stablehlo::ConcatenateOp>(op)) {
      int64_t concatDim = concatOp.getDimension();
      TensorShardingAttr newSharding =
          replicateDimension(operandSharding, concatDim);
      if (newSharding != operandSharding) {
        // A concatenating dim may have a kPermutation factor before the op is
        // outlined if the operands are slice ops. In the outlined function,
        // since the operands are no longer slices, it now has a
        // kNeedReplication factor. We need to reshard the dim to replicated
        // to ensure sharding consistency.
        operandSharding = newSharding;
        arg =
            ReshardOp::create(tempRewriter, op->getLoc(), arg, operandSharding);
      }
    }
    return arg;
  };

  IRMapping mapping;
  size_t implicitArgIdx = op->getNumOperands();
  for (Value capturedVal : implicitCaptures) {
    mapping.map(capturedVal,
                getBlockArgOrSliced(implicitArgIdx++, capturedVal.getType()));
  }

  SmallVector<TensorShardingAttr> opOutShardings(
      shardingInfo.outShardings.begin(), shardingInfo.outShardings.end());
  if (auto concatOp = dyn_cast<stablehlo::ConcatenateOp>(op)) {
    int64_t concatDim = concatOp.getDimension();
    if (!opOutShardings.empty() && opOutShardings[0] &&
        concatDim < opOutShardings[0].getRank() &&
        !opOutShardings[0].getDimShardings()[concatDim].getAxes().empty()) {
      opOutShardings[0] = replicateDimension(opOutShardings[0], concatDim);
    }
  }

  SmallVector<Value> directOperands;
  directOperands.reserve(op->getNumOperands());
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    directOperands.push_back(
        getBlockArgOrSliced(i, op->getOperand(i).getType()));
  }

  Operation* clonedOp = tempRewriter.clone(*op, mapping);
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    clonedOp->setOperand(i, directOperands[i]);
  }

  if (isReshard) {
    for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
      clonedOp->getResult(i).setType(paddedResultTypes[i]);
    }
  }

  // TODO(b/553579414): Remove the need of handling stablehlo.constant when
  // shardy internal pass generate sdy.constant instead of stablehlo.constant
  // in global view of the program.
  ElementsAttr valueAttr = nullptr;
  if (auto constantOp = dyn_cast<sdy::ConstantOp>(clonedOp)) {
    valueAttr = constantOp.getValue();
  } else if (auto constantOp = dyn_cast<stablehlo::ConstantOp>(clonedOp)) {
    valueAttr = constantOp.getValue();
  }
  if (valueAttr) {
    auto origRanked = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto paddedRanked = dyn_cast<RankedTensorType>(paddedResultTypes[0]);
    if (origRanked && paddedRanked && origRanked != paddedRanked) {
      ElementsAttr paddedAttr =
          padElementsAttr(valueAttr, origRanked, paddedRanked);
      if (paddedAttr && paddedAttr.getType() == paddedRanked) {
        if (auto sdyConst = dyn_cast<sdy::ConstantOp>(clonedOp)) {
          sdyConst.setValueAttr(paddedAttr);
        } else if (auto shloConst = dyn_cast<stablehlo::ConstantOp>(clonedOp)) {
          shloConst.setValueAttr(paddedAttr);
        }
        clonedOp->getResult(0).setType(paddedRanked);
      }
    }
  }
  if (trailingAllReduce) {
    setShardings(clonedOp, getShardings(op->getResults()));
    auto clonedAllReduce = AllReduceOp::create(
        tempRewriter, trailingAllReduce.getLoc(), clonedOp->getResult(0),
        trailingAllReduce.getReductionAxes(),
        trailingAllReduce.getReductionOp(), trailingAllReduce.getOutSharding());
    Value res = clonedAllReduce.getResult();
    Type targetType = paddedResultTypes[0];
    if (res.getType() != targetType) {
      res = padHighSideToType(tempRewriter, op->getLoc(), res, targetType,
                              shardingInfo.outShardings[0]);
    }
    func::ReturnOp::create(tempRewriter, op->getLoc(), ValueRange{res});
    return success();
  }

  setShardings(clonedOp, opOutShardings);
  SmallVector<Value> returnValues;
  returnValues.reserve(op->getNumResults());
  for (size_t i = 0; i < op->getNumResults(); ++i) {
    Value res = clonedOp->getResult(i);
    if (opOutShardings[i] != shardingInfo.outShardings[i]) {
      res = ReshardOp::create(tempRewriter, op->getLoc(), res,
                              shardingInfo.outShardings[i]);
    }
    Type targetType = paddedResultTypes[i];
    if (res.getType() != targetType) {
      returnValues.push_back(padHighSideToType(tempRewriter, op->getLoc(), res,
                                               targetType,
                                               shardingInfo.outShardings[i]));
    } else {
      returnValues.push_back(res);
    }
  }

  func::ReturnOp::create(tempRewriter, op->getLoc(), returnValues);
  return success();
}

// TODO(b/545097355): share this code with the export pipeline for whole-module
// partitioning.
//
// Runs the standard Shardy export partitioner pipeline on a module.
LogicalResult runPartitionerPipeline(ModuleOp module, bool enableHaloExchange,
                                     int64_t replicaCount = 1,
                                     int64_t partitionCount = 1,
                                     bool rngBitGeneratorUnsafe = true) {
  MLIRContext* ctx = module.getContext();
  PassManager pm(ctx);
  ShardyResolvePermutationFactorsPassOptions resolveFactorsOptions;
  resolveFactorsOptions.enableHaloExchange = enableHaloExchange;
  resolveFactorsOptions.replicaCount = replicaCount;
  resolveFactorsOptions.partitionCount = partitionCount;
  resolveFactorsOptions.rngBitGeneratorUnsafe = rngBitGeneratorUnsafe;
  pm.addPass(createShardyResolvePermutationFactorsPass(resolveFactorsOptions));
  pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
  pm.addNestedPass<func::FuncOp>(createOptimizeCollectivesPass());
  pm.addPass(createPadForDivisibilityPass());
  ConvertGlobalToLocalPassOptions convertOptions;
  convertOptions.replicaCount = replicaCount;
  convertOptions.partitionCount = partitionCount;
  pm.addPass(createConvertGlobalToLocalPass(convertOptions));
  pm.addPass(createDropShardingAndMeshPass());
  return pm.run(module);
}

// =============================================================================
// Wrapping & Post-Processing Helpers
// =============================================================================

// Wraps the partitioned device-local function body with an
// sdy.manual_computation.
ManualComputationOp createManualComputationFromFunc(
    Operation* op, func::FuncOp outlinedFunc,
    const InstructionShardingInfo& shardingInfo, ArrayRef<Value> manualOperands,
    ArrayRef<Type> paddedResultTypes, IRRewriter& rewriter) {
  auto manualCompOp = ManualComputationOp::create(
      rewriter, op->getLoc(), paddedResultTypes, manualOperands,
      shardingInfo.inShardings, shardingInfo.outShardings,
      shardingInfo.manualAxesList);

  manualCompOp.getRegion().takeBody(outlinedFunc.getBody());
  for (Block& block : manualCompOp.getRegion().getBlocks()) {
    Operation* term = block.getTerminator();
    if (term && !isa<ReturnOp>(term)) {
      OpBuilder termBuilder(&block, block.end());
      ReturnOp::create(termBuilder, term->getLoc(), term->getOperands());
      term->erase();
    }
  }
  return manualCompOp;
}

// Slices any padded results back to their original tensor shapes in the parent
// module and replaces uses of the original operation results.
void sliceIndivisibleResults(Operation* op, ManualComputationOp manualCompOp,
                             ArrayRef<TensorShardingAttr> outShardings,
                             IRRewriter& rewriter) {
  for (size_t j = 0; j < op->getNumResults(); ++j) {
    Value origRes = op->getResult(j);
    Value manualRes = manualCompOp.getResult(j);
    Type origResType = origRes.getType();
    Value slicedRes = sliceHighSideToType(rewriter, op->getLoc(), manualRes,
                                          origResType, outShardings[j]);
    rewriter.replaceAllUsesWith(origRes, slicedRes);
  }
}

// =============================================================================
// Find target region to collect operations for per-instruction partitioning:
// - If targetFuncName is empty:
//     - If the module contains a single FuncOp, use that FuncOp's body.
//     - Otherwise, use @main's body (or first public FuncOp).
// - If targetFuncName is non-empty:
//     - Use the body of the FuncOp with that symbol name in module.
Region* getTargetRegion(ModuleOp module, StringRef targetFuncName) {
  auto getDefaultEntryFunc = [&]() -> func::FuncOp {
    auto funcOps = module.getOps<func::FuncOp>();
    if (llvm::hasSingleElement(funcOps)) {
      return *funcOps.begin();
    }
    if (auto mainFunc = module.lookupSymbol<func::FuncOp>("main")) {
      return mainFunc;
    }
    for (func::FuncOp funcOp : funcOps) {
      if (!funcOp.isDeclaration() && funcOp.isPublic()) {
        return funcOp;
      }
    }
    return nullptr;
  };

  if (!targetFuncName.empty()) {
    auto funcOp = module.lookupSymbol<func::FuncOp>(targetFuncName);
    SDY_CHECK(funcOp) << "Target function not found: " << targetFuncName.str();
    return &funcOp.getBody();
  }

  func::FuncOp entryFunc = getDefaultEntryFunc();
  SDY_CHECK(entryFunc) << "No entry function found in module.";
  return &entryFunc.getBody();
}

// =============================================================================
// Pass Definition
// =============================================================================

struct PerInstructionPartitioningPass
    : public impl::PerInstructionPartitioningPassBase<
          PerInstructionPartitioningPass> {
  using PerInstructionPartitioningPassBase::PerInstructionPartitioningPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module = getOperation();
    FilterConfig config = parseFilter(filter);

    Region* targetRegion = getTargetRegion(module, config.targetFunc);

    // Collect ops for partitioning from target region.
    SmallVector<Operation*> opsToPartition;
    int64_t instructionSeqId = 0;
    for (Block& block : *targetRegion) {
      for (Operation& op : block) {
        // Filter non-candidate / metadata from counting.
        if (op.hasTrait<OpTrait::IsTerminator>()) {
          continue;
        }
        // TODO(b/545097355): fine tune the list of ops to skip.
        //
        // We skip metadata/control ops and unreduced-related collective
        // operations (sdy.replicated_to_unreduced, sdy.sharded_to_unreduced)
        // because unreduced collectives are handled later in export.
        if (isa<DataFlowEdgeOp, ManualComputationOp, MeshOp,
                PropagationBarrierOp, ReplicatedToUnreducedOp, ReturnOp,
                ShardedToUnreducedOp, ShardingConstraintOp, ShardingGroupOp>(
                &op)) {
          continue;
        }
        // Avoid taking a dependency on the MHLO dialect by checking operation
        // names directly.
        if (op.getName().getStringRef() == "mhlo.copy") {
          continue;
        }
        if (auto customCall = dyn_cast<mlir::stablehlo::CustomCallOp>(&op)) {
          if (customCall.getCallTargetName() == "Sharding" ||
              customCall.getCallTargetName() == "mhlo.sharding") {
            continue;
          }
        }
        if (auto allReduceOp = dyn_cast<sdy::AllReduceOp>(&op)) {
          if (Operation* defOp = allReduceOp.getTensor().getDefiningOp();
              defOp && getTrailingConvAllReduce(defOp) == allReduceOp) {
            continue;
          }
        }
        if (!hasAnyShardedValue(&op)) {
          continue;
        }

        // Count the number of instructions that need to be partitioned.
        int64_t currentId = instructionSeqId++;

        if (!matchesFilter(&op, currentId, config)) {
          continue;
        }

        opsToPartition.push_back(&op);
      }
    }

    // Partition the ops and wrap the device code with manual_computation.
    for (Operation* op : opsToPartition) {
      if (!partitionAndWrapInstruction(op)) {
        signalPassFailure();
        return;
      }
    }
  }

 private:
  bool partitionAndWrapInstruction(Operation* op) {
    MLIRContext* ctx = op->getContext();
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());

    SmallVector<Value> liveInValues;
    llvm::SetVector<Value> implicitCaptures;
    collectLiveInValues(op, liveInValues, implicitCaptures);

    SmallVector<func::FuncOp> callees;
    collectTransitiveCallees(op, symbolTable, callees);

    Attribute meshOrRef = getTargetMeshOrRef(liveInValues, op, symbolTable);
    if (!meshOrRef) {
      return false;
    }

    FlatSymbolRefAttr meshSym =
        getOrCreateMeshSymbol(op, meshOrRef, symbolTable);
    Attribute targetMesh = meshSym ? Attribute(meshSym) : meshOrRef;

    sdy::AllReduceOp trailingAllReduce = getTrailingConvAllReduce(op);
    InstructionShardingInfo shardingInfo =
        getInstructionShardingInfo(op, liveInValues, meshSym, targetMesh);
    if (trailingAllReduce) {
      shardingInfo.outShardings = {trailingAllReduce.getOutSharding()};
    }

    SmallVector<Type> paddedArgTypes =
        computePaddedOperandTypes(op, liveInValues, shardingInfo, symbolTable);
    SmallVector<Type> paddedResultTypes =
        computePaddedResultTypes(op, shardingInfo, symbolTable);

    SmallVector<Value> manualOperands = padIndivisibleLiveInValues(
        op, liveInValues, shardingInfo, paddedArgTypes, rewriter);

    OwningOpRef<ModuleOp> tempModule;
    func::FuncOp outlinedFunc;
    if (failed(outlineInstruction(op, implicitCaptures, callees, shardingInfo,
                                  paddedArgTypes, paddedResultTypes, tempModule,
                                  outlinedFunc, trailingAllReduce)) ||
        failed(runPartitionerPipeline(*tempModule, enableHaloExchange,
                                      replicaCount, partitionCount,
                                      rngBitGeneratorUnsafe))) {
      return false;
    }

    // Move partitioned helper functions from tempModule into parent module
    // with unique symbol names and update symbol references.
    SmallVector<func::FuncOp> insertedCallees;
    SmallVector<std::pair<StringAttr, StringAttr>> renamedCallees;
    for (func::FuncOp localCallee : tempModule->getOps<func::FuncOp>()) {
      if (localCallee == outlinedFunc) {
        continue;
      }
      StringAttr oldSymName = localCallee.getSymNameAttr();
      func::FuncOp clonedCallee = localCallee.clone();
      StringAttr newSymName = symbolTable.insert(clonedCallee);
      insertedCallees.push_back(clonedCallee);
      if (newSymName != oldSymName) {
        renamedCallees.emplace_back(oldSymName, newSymName);
      }
    }
    for (auto [oldSymName, newSymName] : renamedCallees) {
      if (failed(SymbolTable::replaceAllSymbolUses(oldSymName, newSymName,
                                                   outlinedFunc))) {
        return false;
      }
      for (func::FuncOp insertedFunc : insertedCallees) {
        if (failed(SymbolTable::replaceAllSymbolUses(oldSymName, newSymName,
                                                     insertedFunc))) {
          return false;
        }
      }
    }

    auto manualCompOp = createManualComputationFromFunc(
        op, outlinedFunc, shardingInfo, manualOperands, paddedResultTypes,
        rewriter);

    Operation* resultOp =
        trailingAllReduce ? trailingAllReduce.getOperation() : op;
    sliceIndivisibleResults(resultOp, manualCompOp, shardingInfo.outShardings,
                            rewriter);

    if (trailingAllReduce) {
      rewriter.eraseOp(trailingAllReduce);
    }
    rewriter.eraseOp(op);
    return true;
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
