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
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
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
      getShardings(op->getResults(), /*clearUnreducedAxes=*/true);

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

  return info;
}

// =============================================================================
// Divisible Padding & Outlined Partitioning Helpers
// =============================================================================

// Returns the unified padded type for ops whose operand and result types must
// match (such as sdy.reshard).
RankedTensorType getUnifiedPaddedTypeForReshard(
    Operation* op, const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable) {
  if (!isa<ReshardOp>(op)) {
    return nullptr;
  }
  Type origType = op->getOperand(0).getType();
  Type inPaddedType =
      getDivisiblePaddedType(origType, shardingInfo.inShardings[0], symbolTable,
                             &shardingInfo.manualAxesSet);
  Type outPaddedType =
      getDivisiblePaddedType(origType, shardingInfo.outShardings[0],
                             symbolTable, &shardingInfo.manualAxesSet);
  if (inPaddedType == origType && outPaddedType == origType) {
    return nullptr;
  }
  auto ranked = cast<RankedTensorType>(origType);
  SmallVector<int64_t> maxShape(ranked.getShape());
  if (auto inPadded = dyn_cast<RankedTensorType>(inPaddedType)) {
    for (int d = 0; d < ranked.getRank(); ++d) {
      maxShape[d] = std::max(maxShape[d], inPadded.getDimSize(d));
    }
  }
  if (auto outPadded = dyn_cast<RankedTensorType>(outPaddedType)) {
    for (int d = 0; d < ranked.getRank(); ++d) {
      maxShape[d] = std::max(maxShape[d], outPadded.getDimSize(d));
    }
  }
  return RankedTensorType::get(maxShape, ranked.getElementType());
}

// TODO(b/545097355): Consider code-sharing with pad-for-divisibility.
//
// Pads indivisible operands in the parent module before manual computation.
SmallVector<Value> padIndivisibleOperands(
    Operation* op, const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable, SmallVector<Type>& paddedArgTypes,
    IRRewriter& rewriter) {
  RankedTensorType unifiedReshardType =
      getUnifiedPaddedTypeForReshard(op, shardingInfo, symbolTable);
  SmallVector<Value> manualOperands;
  for (auto [operand, sharding] :
       llvm::zip_equal(op->getOperands(), shardingInfo.inShardings)) {
    Type origType = operand.getType();
    Type paddedType = unifiedReshardType ? unifiedReshardType
                                         : getDivisiblePaddedType(
                                               origType, sharding, symbolTable,
                                               &shardingInfo.manualAxesSet);
    if (paddedType != origType) {
      auto origRanked = cast<RankedTensorType>(origType);
      auto paddedRanked = cast<RankedTensorType>(paddedType);
      auto zeroType = RankedTensorType::get({}, paddedRanked.getElementType());
      Value zero = stablehlo::ConstantOp::create(
          rewriter, op->getLoc(), rewriter.getZeroAttr(zeroType));
      SmallVector<int64_t> edgePaddingLow(paddedRanked.getRank(), 0);
      SmallVector<int64_t> edgePaddingHigh(paddedRanked.getRank(), 0);
      SmallVector<int64_t> interiorPadding(paddedRanked.getRank(), 0);
      for (int d = 0; d < paddedRanked.getRank(); ++d) {
        edgePaddingHigh[d] =
            paddedRanked.getDimSize(d) - origRanked.getDimSize(d);
      }
      Value paddedOperand = stablehlo::PadOp::create(
          rewriter, op->getLoc(), paddedType, operand, zero,
          rewriter.getDenseI64ArrayAttr(edgePaddingLow),
          rewriter.getDenseI64ArrayAttr(edgePaddingHigh),
          rewriter.getDenseI64ArrayAttr(interiorPadding));
      setSharding(paddedOperand, sharding);
      manualOperands.push_back(paddedOperand);
      paddedArgTypes.push_back(paddedType);
    } else {
      manualOperands.push_back(operand);
      paddedArgTypes.push_back(origType);
    }
  }
  return manualOperands;
}

// Computes the padded result types for manual computation.
SmallVector<Type> computePaddedResultTypes(
    Operation* op, const InstructionShardingInfo& shardingInfo,
    const SymbolTable& symbolTable) {
  RankedTensorType unifiedReshardType =
      getUnifiedPaddedTypeForReshard(op, shardingInfo, symbolTable);
  SmallVector<Type> paddedResultTypes;
  for (auto [result, sharding] :
       llvm::zip_equal(op->getResults(), shardingInfo.outShardings)) {
    Type origResType = result.getType();
    Type paddedResType =
        unifiedReshardType
            ? unifiedReshardType
            : getDivisiblePaddedType(origResType, sharding, symbolTable,
                                     &shardingInfo.manualAxesSet);
    paddedResultTypes.push_back(paddedResType);
  }
  return paddedResultTypes;
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
  Operation* clonedOp = tempRewriter.clone(*op);
  for (size_t i = 0; i < clonedOp->getNumOperands(); ++i) {
    clonedOp->setOperand(i, block->getArgument(i));
  }
  if (clonedOp->getResultTypes() != TypeRange(paddedResultTypes)) {
    for (size_t i = 0; i < clonedOp->getNumResults(); ++i) {
      Type origResType = op->getResult(i).getType();
      Type paddedResType = paddedResultTypes[i];
      if (origResType != paddedResType) {
        auto origRanked = cast<RankedTensorType>(origResType);
        auto paddedRanked = cast<RankedTensorType>(paddedResType);
        // TODO(b/545097355): Update shape-dependent attributes for other ops
        // requiring padding (e.g. slice_sizes for stablehlo.dynamic_slice,
        // gather, scatter) if they are partitioned individually with
        // indivisible shardings.
        if (auto sliceOp = dyn_cast<stablehlo::SliceOp>(clonedOp)) {
          SmallVector<int64_t> newLimit(sliceOp.getLimitIndices().begin(),
                                        sliceOp.getLimitIndices().end());
          for (int d = 0; d < paddedRanked.getRank(); ++d) {
            newLimit[d] +=
                (paddedRanked.getDimSize(d) - origRanked.getDimSize(d));
          }
          sliceOp.setLimitIndicesAttr(
              tempRewriter.getDenseI64ArrayAttr(newLimit));
        } else if (auto padOp = dyn_cast<stablehlo::PadOp>(clonedOp)) {
          SmallVector<int64_t> newHigh(padOp.getEdgePaddingHigh().begin(),
                                       padOp.getEdgePaddingHigh().end());
          for (int d = 0; d < paddedRanked.getRank(); ++d) {
            newHigh[d] +=
                (paddedRanked.getDimSize(d) - origRanked.getDimSize(d));
          }
          padOp.setEdgePaddingHighAttr(
              tempRewriter.getDenseI64ArrayAttr(newHigh));
        }
        clonedOp->getResult(i).setType(paddedResType);
      }
    }
  }
  func::ReturnOp::create(tempRewriter, op->getLoc(), clonedOp->getResults());
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
  InsertExplicitReshardsPassOptions insertReshardOptions;
  insertReshardOptions.enableFullVersion = true;
  pm.addNestedPass<func::FuncOp>(
      createInsertExplicitReshardsPass(insertReshardOptions));
  pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
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
                             ArrayRef<Type> paddedResultTypes,
                             IRRewriter& rewriter) {
  for (size_t j = 0; j < op->getNumResults(); ++j) {
    Value manualRes = manualCompOp.getResult(j);
    Type origResType = op->getResult(j).getType();
    Type paddedResType = paddedResultTypes[j];
    if (origResType == paddedResType) {
      rewriter.replaceAllUsesWith(op->getResult(j), manualRes);
      continue;
    }
    auto origRanked = cast<RankedTensorType>(origResType);
    SmallVector<int64_t> startIndices(origRanked.getRank(), 0);
    SmallVector<int64_t> limitIndices(origRanked.getShape().begin(),
                                      origRanked.getShape().end());
    SmallVector<int64_t> strides(origRanked.getRank(), 1);
    Value slicedRes = stablehlo::SliceOp::create(
        rewriter, op->getLoc(), origResType, manualRes,
        rewriter.getDenseI64ArrayAttr(startIndices),
        rewriter.getDenseI64ArrayAttr(limitIndices),
        rewriter.getDenseI64ArrayAttr(strides));
    setSharding(slicedRes, outShardings[j]);
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
          // We currently skip Collective operations because we are not ready
          // to handle them in this pass yet, for example,
          // sdy.replicated_to_unreduced is currently handled later in the
          // export pipeline.
          if (isa<CollectiveOpInterface, DataFlowEdgeOp, ManualComputationOp,
                  MeshOp, PropagationBarrierOp, ReturnOp, ShardingConstraintOp,
                  ShardingGroupOp>(&op)) {
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

    SmallVector<Type> paddedArgTypes;
    SmallVector<Value> manualOperands = padIndivisibleOperands(
        op, shardingInfo, symbolTable, paddedArgTypes, rewriter);

    SmallVector<Type> paddedResultTypes =
        computePaddedResultTypes(op, shardingInfo, symbolTable);

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
                            paddedResultTypes, rewriter);

    rewriter.eraseOp(op);
    return true;
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
