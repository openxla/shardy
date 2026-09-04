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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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

// Parses a filter string that controls which operations should be partitioned.
//
// The filter string format is a comma-separated list of tokens in any order:
//    - selectLow=<N>: Only partition operations with sequence ID >= N.
//    - selectHigh=<N>: Only partition operations with sequence ID <= N.
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
    } else {
      config.opNames.push_back(token.str());
    }
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
// operand or result of op.
Attribute getTargetMeshOrRef(Operation* op, const SymbolTable& symbolTable) {
  SmallVector<TensorShardingAttr> inShardings;
  for (Value operand : op->getOperands()) {
    inShardings.push_back(getSharding(operand));
  }
  SmallVector<TensorShardingAttr> outShardings;
  for (Value result : op->getResults()) {
    outShardings.push_back(getSharding(result));
  }
  return getCommonMeshOrRef(inShardings, outShardings, symbolTable);
}

// Collects operand and result shardings (defaulting to fully replicated if
// missing) and all unique manual axes used across the instruction.
InstructionShardingInfo getInstructionShardingInfo(Operation* op,
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

  info.inShardings =
      getShardings(op->getOperands(), /*clearUnreducedAxes=*/false);
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
    Operation* op, const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable) {
  if (RankedTensorType unifiedType =
          getUnifiedPaddedTypeForReshard(op, shardingInfo, symbolTable)) {
    return {unifiedType};
  }
  SmallVector<Type> paddedArgTypes;
  for (auto [operand, sharding] :
       llvm::zip_equal(op->getOperands(), shardingInfo.inShardings)) {
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

// Pads indivisible operands in the parent module before manual computation.
SmallVector<Value> padIndivisibleOperands(
    Operation* op, const InstructionShardingInfo& shardingInfo,
    ArrayRef<Type> paddedArgTypes, IRRewriter& rewriter) {
  SmallVector<Value> manualOperands;
  for (auto [operand, paddedType, sharding] : llvm::zip_equal(
           op->getOperands(), paddedArgTypes, shardingInfo.inShardings)) {
    manualOperands.push_back(padHighSideToType(
        rewriter, op->getLoc(), operand, paddedType, sharding,
        /*paddingValue=*/nullptr, /*allowSlicePeephole=*/true));
  }
  return manualOperands;
}



// Outlines the target instruction and its mesh dependencies into an ephemeral
// module containing a single private func::FuncOp.
LogicalResult outlineInstruction(Operation* op,
                                 const InstructionShardingInfo& shardingInfo,
                                 ArrayRef<Type> paddedArgTypes,
                                 ArrayRef<Type> paddedResultTypes,
                                 OwningOpRef<ModuleOp>& tempModule,
                                 func::FuncOp& outlinedFunc) {
  MLIRContext* ctx = op->getContext();
  tempModule = ModuleOp::create(op->getLoc());
  IRRewriter tempRewriter(ctx);
  tempRewriter.setInsertionPointToStart(tempModule->getBody());

  for (MeshOp meshOp : op->getParentOfType<ModuleOp>().getOps<MeshOp>()) {
    tempRewriter.clone(*meshOp);
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

  Block* block = outlinedFunc.addEntryBlock();
  tempRewriter.setInsertionPointToStart(block);
  SmallVector<Value> operands(block->getArguments().begin(),
                              block->getArguments().end());
  SmallVector<Type> resultTypes;
  auto inferTypeOp = dyn_cast<InferTypeOpInterface>(op);
  if (!inferTypeOp ||
      failed(inferTypeOp.inferReturnTypes(
          ctx, op->getLoc(), operands, op->getAttrDictionary(),
          op->getPropertiesStorage(), op->getRegions(), resultTypes))) {
    resultTypes.assign(paddedResultTypes.begin(), paddedResultTypes.end());
  }

  Operation* clonedOp = tempRewriter.clone(*op);
  for (size_t i = 0; i < clonedOp->getNumOperands(); ++i) {
    clonedOp->setOperand(i, block->getArgument(i));
  }
  for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
    if (i < resultTypes.size()) {
      clonedOp->getResult(i).setType(resultTypes[i]);
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
  setShardings(clonedOp, shardingInfo.outShardings);

  SmallVector<Value> returnValues;
  returnValues.reserve(clonedOp->getNumResults());
  for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
    Value res = clonedOp->getResult(i);
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
                                     int64_t partitionCount = 1) {
  MLIRContext* ctx = module.getContext();
  PassManager pm(ctx);
  ShardyResolvePermutationFactorsPassOptions resolveFactorsOptions;
  resolveFactorsOptions.enableHaloExchange = enableHaloExchange;
  resolveFactorsOptions.replicaCount = replicaCount;
  resolveFactorsOptions.partitionCount = partitionCount;
  pm.addPass(createShardyResolvePermutationFactorsPass(resolveFactorsOptions));
  pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
  pm.addNestedPass<func::FuncOp>(createOptimizeCollectivesPass());
  pm.addNestedPass<func::FuncOp>(createPadForDivisibilityPass());
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
// module and replaces uses of the original operation.
void sliceIndivisibleResults(Operation* op, ManualComputationOp manualCompOp,
                             ArrayRef<TensorShardingAttr> outShardings,
                             IRRewriter& rewriter) {
  for (size_t j = 0; j < op->getNumResults(); ++j) {
    Value manualRes = manualCompOp.getResult(j);
    Type origResType = op->getResult(j).getType();
    Value slicedRes = sliceHighSideToType(rewriter, op->getLoc(), manualRes,
                                          origResType, outShardings[j]);
    rewriter.replaceAllUsesWith(op->getResult(j), slicedRes);
  }
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

    // Collect ops for partitioning.
    SmallVector<Operation*> opsToPartition;
    int64_t instructionSeqId = 0;
    module.walk([&](func::FuncOp funcOp) {
      if (funcOp.isDeclaration()) {
        return;
      }
      // TODO(b/545097355): support partitioning ops nested inside control flow,
      // such as stablehlo.while, stablehlo.if, and stablehlo.case etc.
      for (Block& block : funcOp.getBody()) {
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
    });

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
    Attribute meshOrRef = getTargetMeshOrRef(op, symbolTable);
    if (!meshOrRef) {
      return false;
    }

    FlatSymbolRefAttr meshSym =
        getOrCreateMeshSymbol(op, meshOrRef, symbolTable);
    Attribute targetMesh = meshSym ? Attribute(meshSym) : meshOrRef;

    InstructionShardingInfo shardingInfo =
        getInstructionShardingInfo(op, meshSym, targetMesh);

    SmallVector<Type> paddedArgTypes =
        computePaddedOperandTypes(op, shardingInfo, symbolTable);
    SmallVector<Type> paddedResultTypes =
        computePaddedResultTypes(op, shardingInfo, symbolTable);

    SmallVector<Value> manualOperands =
        padIndivisibleOperands(op, shardingInfo, paddedArgTypes, rewriter);

    OwningOpRef<ModuleOp> tempModule;
    func::FuncOp outlinedFunc;
    if (failed(outlineInstruction(op, shardingInfo, paddedArgTypes,
                                  paddedResultTypes, tempModule,
                                  outlinedFunc)) ||
        failed(runPartitionerPipeline(*tempModule, enableHaloExchange,
                                      replicaCount, partitionCount))) {
      return false;
    }

    auto manualCompOp = createManualComputationFromFunc(
        op, outlinedFunc, shardingInfo, manualOperands, paddedResultTypes,
        rewriter);

    sliceIndivisibleResults(op, manualCompOp, shardingInfo.outShardings,
                            rewriter);

    rewriter.eraseOp(op);
    return true;
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
