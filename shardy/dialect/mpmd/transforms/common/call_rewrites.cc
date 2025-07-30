/* Copyright 2025 The MPMD Authors.

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

#include <string_view>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/common/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_CALLINLINEPASS
#define GEN_PASS_DEF_ERASEUNUSEDCALLEEBLOCKARGUMENTSPASS
#define GEN_PASS_DEF_FROMUNROLLTOCALLCOUNTERPASS
#define GEN_PASS_DEF_SINKNEGLIGIBLEOPSINTOCALLOPPASS
#include "shardy/dialect/mpmd/transforms/common/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

// TODO(jupvfranco): Consider whether inlining should be recursive to handle
// nested calls.
class CallOpInliner : public InlinerInterface {
 public:
  explicit CallOpInliner(MLIRContext* context, Operation* inlined_call)
      : InlinerInterface(context), inlined_call_(inlined_call) {}

 private:
  // Iterates over all inlined blocks and sets attributes passed as input to
  // this inliner.
  void processInlinedBlocks(
      iterator_range<Region::iterator> inlined_blocks) final {
    for (Block& block : inlined_blocks) {
      block.walk([&](Operation* op) {
        if (isa<FragmentCallOp, FragmentOp>(op)) {
          CopyAttributes(inlined_call_, op, /*elided_attrs_set=*/{"callee"});
        }
      });
    }
  }

  // mpmd.call_ops are inlinable when using this inliner. Because we
  // define this function, the one in the mpmd::MpmdDialectInlinerInterface
  // won't be applied.
  // NOTE: this is enough because we only used this InlinerInterface for
  // mpmd.call ops. In fact, returning `true` would suffice here. We
  // check if the `call` op is a CallOp for readability and to be more future
  // proof.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return isa<CallOp>(call);
  }

  Operation* inlined_call_;
};

class CallInlinePass : public impl::CallInlinePassBase<CallInlinePass> {
  using CallInlinePassBase::CallInlinePassBase;

 private:
  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    module_op.walk([](CallOp call_op) {
      CallOpInliner inliner(call_op->getContext(), call_op);
      InlinerConfig config;
      auto call_op_interface = cast<CallOpInterface>(*call_op);
      FuncOp callable = cast<FuncOp>(call_op_interface.resolveCallable());
      auto res = inlineCall(inliner, config.getCloneCallback(),
                            call_op_interface, callable, &callable.getRegion(),
                            /*shouldCloneInlinedRegion =*/true);
      SDY_CHECK(res.succeeded())
          << "Failed to inline " << std::string_view(callable.getSymName());
      call_op->erase();
    });
  }
};

}  // namespace

void AddCallInliningRelatedPasses(OpPassManager& pm) {
  pm.addPass(createCallInlinePass());
  // TODO(jupvfranco): Investigate whether a CSE here to remove duplication of
  // sinked ops is a good idea.
  // Remove any private function that is no longer needed.
  pm.addPass(createSymbolDCEPass());
}

namespace {

class SinkNegligibleOpsIntoCallOpPass
    : public impl::SinkNegligibleOpsIntoCallOpPassBase<
          SinkNegligibleOpsIntoCallOpPass> {
  using SinkNegligibleOpsIntoCallOpPassBase::
      SinkNegligibleOpsIntoCallOpPassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    for (FuncOp func_op : GetMpmdFunctions(module_op)) {
      runOnFunc(func_op);
    }
  }

  void runOnFunc(FuncOp func_op) {
    BitVector erase_arguments(func_op.getNumArguments());
    OpBuilder block_builder = OpBuilder::atBlockBegin(&func_op.front());
    for (const MpmdDataflowEdge& edge :
         GetMpmdDataflowEdgesForFuncArgs(func_op)) {
      OpOperand* first_operand = edge.sources.front();
      if (!llvm::all_of(edge.sources, [first_operand](OpOperand* operand) {
            return operand->get() == first_operand->get();
          })) {
        continue;
      }
      // We are guaranteed a single edge target (`arg`) in this case.
      BlockArgument arg = cast<BlockArgument>(edge.targets.front());
      // We only allow sinking of ops without operands, assuming that these will
      // be negligible, as this may duplicate such ops.
      if (Operation* operand_producer = first_operand->get().getDefiningOp();
          operand_producer && operand_producer->getNumOperands() == 0 &&
          operand_producer->getNumResults() == 1) {
        Operation* cloned = Clone(block_builder, *operand_producer, {});
        arg.replaceAllUsesWith(cloned->getResult(0));
        erase_arguments.set(arg.getArgNumber());
      }
    }
    if (erase_arguments.none()) {
      return;
    }
    DenseSet<FuncOp> caller_functions;
    (void)func_op.eraseArguments(erase_arguments);
    for (CallOp call : GetCallOps(func_op)) {
      call->eraseOperands(erase_arguments);
      caller_functions.insert(call->getParentOfType<FuncOp>());
    }
    IRRewriter rewriter(func_op.getContext());
    for (FuncOp caller_func : caller_functions) {
      (void)simplifyRegions(rewriter, caller_func.getRegion());
    }
  }
};

class FromUnrollToCallCounterPass
    : public impl::FromUnrollToCallCounterPassBase<
          FromUnrollToCallCounterPass> {
  using FromUnrollToCallCounterPassBase::FromUnrollToCallCounterPassBase;

  void runOnFunc(func::FuncOp func_op) override {
    func_op->walk([](CallOp call_op) {
      if (auto unroll_counter =
              call_op->getAttrOfType<IntegerAttr>(kUnrollCounterAttrName)) {
        call_op->setAttr(kCallCounterAttrName, unroll_counter);
        call_op->removeAttr(kUnrollCounterAttrName);
      }
    });
  }
};

class EraseUnusedCalleeBlockArgumentsPass
    : public impl::EraseUnusedCalleeBlockArgumentsPassBase<
          EraseUnusedCalleeBlockArgumentsPass> {
  using EraseUnusedCalleeBlockArgumentsPassBase::
      EraseUnusedCalleeBlockArgumentsPassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    for (FuncOp func_op : GetMpmdFunctions(module_op)) {
      if (!IsEntryPointFunction(func_op)) {
        runOnFunc(func_op);
      }
    }
  }

  void runOnFunc(FuncOp func_op) {
    SmallVector<CallOp> call_ops = GetCallOps(func_op);
    if (call_ops.empty()) {
      return;
    }
    IRRewriter rewriter(func_op.getContext());
    BitVector erase_arguments(func_op.getNumArguments());
    BitVector erase_results(func_op.getNumResults());
    // Maps the index of the call_op result to the index of the call_op operand,
    // with which we will replace the result.
    DenseMap<int, int> replacement_map;
    Operation* terminator = func_op.getBody().front().getTerminator();
    for (BlockArgument arg : func_op.getArguments()) {
      if (HasOtherUsersExcept(arg, terminator)) {
        // If the argument is used by any op other than the terminator, then it
        // is needed by a computation and we cannot erase it from the function.
        continue;
      }
      // At this point, we know that the arg is used by the terminator and only
      // by the terminator.
      erase_arguments.set(arg.getArgNumber());
      for (OpOperand& use : arg.getUses()) {
        erase_results.set(use.getOperandNumber());
        replacement_map[use.getOperandNumber()] = arg.getArgNumber();
      }
    }

    // If no argument is erased, then there's no work to do.
    if (erase_arguments.none()) {
      return;
    }

    // Replace any call op result that needs replacement.
    for (CallOp call_op : call_ops) {
      for (auto [result_index, operand_index] : replacement_map) {
        rewriter.replaceAllUsesWith(call_op.getResult(result_index),
                                    call_op.getOperand(operand_index));
      }
    }

    if (erase_results.all()) {
      // Now that we replaced the call_op results, if they have to be removed,
      // then we know they are completely unused.
      for (CallOp call_op : call_ops) {
        rewriter.eraseOp(call_op);
      }
      rewriter.eraseOp(func_op);
      return;
    }

    // Erase the arguments and results from the function itself.
    if (erase_results.any()) {
      (void)func_op.eraseResults(erase_results);
      terminator->eraseOperands(erase_results);
    }
    // The arguments must be erased after the return's operands, so that they
    // have no uses.
    (void)func_op.eraseArguments(erase_arguments);
    UpdateFunctionType(func_op);

    // Flip the bits, so it tells us which results to keep.
    BitVector& kept_results = erase_results.flip();
    // Erase the operands and results from the call-ops.
    for (CallOp call_op : call_ops) {
      call_op->eraseOperands(erase_arguments);
      std::vector<Type> result_types;
      for (unsigned int index : kept_results.set_bits()) {
        result_types.push_back(call_op->getResult(index).getType());
      }
      // Alas, we cannot directly erase results of an op, so we need to create
      // a new call op, and use it to replace the old one.
      rewriter.setInsertionPoint(call_op);
      auto new_call_op =
          CallOp::create(rewriter, call_op.getLoc(), result_types,
                         call_op->getOperands(), call_op.getCalleeAttr());
      new_call_op->setDiscardableAttrs(call_op->getDiscardableAttrDictionary());
      for (auto [new_result, old_result_index] :
           llvm::zip_equal(new_call_op.getResults(), kept_results.set_bits())) {
        rewriter.replaceAllUsesWith(call_op->getResult(old_result_index),
                                    new_result);
      }
      rewriter.eraseOp(call_op);
    }
  }
};
}  // namespace

}  // namespace mlir::mpmd
