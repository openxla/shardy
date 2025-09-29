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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_MARKINPUTOUTPUTWITHLAYOUTSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

bool isAutoLayout(StringAttr layout) {
  return layout && layout == kLayoutModeAuto;
}

// A layout attribute denotes a default layout if it is empty or if it is a
// `default` string.
bool isDefaultLayout(StringAttr layout) {
  return !layout || layout == kLayoutModeDefault;
}

// Two layouts are compatible if: 1) they are both default layouts, 2) they are
// identical custom layouts, or 3) one or both are auto layouts.
bool areLayoutsCompatible(StringAttr layout1, StringAttr layout2) {
  if (isDefaultLayout(layout1) && isDefaultLayout(layout2)) {
    return true;
  }
  if (isAutoLayout(layout1) || isAutoLayout(layout2)) {
    return true;
  }
  return layout1 == layout2;
}

void setFragmentLayout(SmallVector<Attribute>& attrs, int idx,
                       StringAttr layout) {
  if (isDefaultLayout(layout)) {
    RemoveAttr(attrs[idx], kLayoutModeAttr);
  } else {
    InsertAttr(attrs[idx], kLayoutModeAttr, layout,
               /*insert_if_not_present=*/true);
  }
}

class MarkInputOutputWithLayoutsPass
    : public impl::MarkInputOutputWithLayoutsPassBase<
          MarkInputOutputWithLayoutsPass> {
  using MarkInputOutputWithLayoutsPassBase::MarkInputOutputWithLayoutsPassBase;

 private:
  void runOnFunc(FuncOp func) final {
    if (!IsEntryPointFunction(func)) {
      return;
    }
    for (BlockArgument programArg : func.getArguments()) {
      if (!propagateInputsToEverything(programArg, func)) {
        signalPassFailure();
        return;
      }
    }
    for (FragmentOp frag : func.getOps<FragmentOp>()) {
      SmallVector<Attribute> fragResAttrs = GetResAttrsOrCreateDefault(frag);
      for (OpResult fragRes : frag->getOpResults()) {
        if (!propagateFragmentResultsToEverything(fragRes, func,
                                                  fragResAttrs)) {
          signalPassFailure();
          return;
        }
      }
      SetResAttrs(frag, fragResAttrs);
    }
  }

  // Choose layout for FragmentOp result based on its uses {TransferOp,
  // ReturnOp, another FragmentOp} and then propagate to all these uses.
  //
  // *Important*: If the chosen layout is AUTO, but there are uses in any
  // fragment, enforce DEFAULT layout, since we don't support cross-fragment
  // layout propagation.
  bool propagateFragmentResultsToEverything(
      OpResult fragRes, FuncOp& programFunc,
      SmallVector<Attribute>& fragResAttrs) {
    // the fragment result layouts are not set by default - meaning they are
    // auto
    StringAttr fragResLayoutAttr =
        StringAttr::get(programFunc.getContext(), kLayoutModeAuto);
    if (fragRes.use_empty()) {
      setFragmentLayout(fragResAttrs, fragRes.getResultNumber(),
                        fragResLayoutAttr);
      return true;
    }
    int numUsesInFragments = 0;
    int propagatedFromProgramResIdx = -1;
    for (OpOperand& use : fragRes.getUses()) {
      if (isa<TransferOp>(use.getOwner())) {
        // if used in a transfer op, we *MUST* stick to default layout, since
        // ifrt transfers only support default layouts
        fragResLayoutAttr =
            StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
        break;
      }
      if (use.getOwner() != programFunc.front().getTerminator()) {
        numUsesInFragments++;
        continue;
      }
      StringAttr programResLayoutAttr =
          programFunc.getResultAttrOfType<StringAttr>(use.getOperandNumber(),
                                                      kLayoutModeAttr);
      if (!areLayoutsCompatible(fragResLayoutAttr, programResLayoutAttr)) {
        emitError(use.getOwner()->getLoc())
            << "Result #" << propagatedFromProgramResIdx
            << " is also returned as result #" << use.getOperandNumber()
            << ", but with incompatible layouts: " << fragResLayoutAttr
            << " vs. " << programResLayoutAttr;
        return false;
      }
      if (!isAutoLayout(programResLayoutAttr) &&
          isAutoLayout(fragResLayoutAttr)) {
        fragResLayoutAttr = programResLayoutAttr;
        propagatedFromProgramResIdx = use.getOperandNumber();
      }
    }
    if (isAutoLayout(fragResLayoutAttr) && (numUsesInFragments > 0)) {
      fragResLayoutAttr =
          StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
    }
    setFragmentLayout(fragResAttrs, fragRes.getResultNumber(),
                      fragResLayoutAttr);
    for (OpOperand& use : fragRes.getUses()) {
      if (FragmentOp fragOp = dyn_cast<FragmentOp>(use.getOwner())) {
        SmallVector<Attribute> fragAttrs = GetArgAttrsOrCreateDefault(fragOp);
        setFragmentLayout(fragAttrs, use.getOperandNumber(), fragResLayoutAttr);
        SetArgAttrs(fragOp, fragAttrs);
      } else if (use.getOwner() == programFunc.front().getTerminator()) {
        if (isDefaultLayout(fragResLayoutAttr)) {
          programFunc.removeResultAttr(use.getOperandNumber(), kLayoutModeAttr);
        } else {
          programFunc.setResultAttr(use.getOperandNumber(), kLayoutModeAttr,
                                    fragResLayoutAttr);
        }
      }
    }
    return true;
  }

  // Choose a common layout for program arg based on its initial value and uses
  // {TransferOp(always DEFAULT), ReturnOp, another FragmentOp}, and then
  // propagate to all these uses.
  //
  // *Important*: If the chosen layout is AUTO, but there are *MORE THAN ONE*
  // use in a fragment, enforce DEFAULT layout, since we don't support
  // cross-fragment layout propagation yet.
  bool propagateInputsToEverything(BlockArgument arg, FuncOp& programFunc) {
    StringAttr argLayout = programFunc.getArgAttrOfType<StringAttr>(
        arg.getArgNumber(), kLayoutModeAttr);
    if (arg.use_empty()) {
      if (isDefaultLayout(argLayout)) {
        programFunc.removeArgAttr(arg.getArgNumber(), kLayoutModeAttr);
      }
      return true;
    }

    int numUsesInFragments = 0;
    int propagatedFromResultIdx = -1;
    for (OpOperand& use : arg.getUses()) {
      if (isa<TransferOp>(use.getOwner())) {
        // if used in a transfer op, we *MUST* stick to default layout, since
        // ifrt transfers only support default layouts
        argLayout =
            StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
        break;
      }
      if (use.getOwner() != programFunc.front().getTerminator()) {
        numUsesInFragments++;
        continue;
      }
      StringAttr resultLayoutAttr = programFunc.getResultAttrOfType<StringAttr>(
          use.getOperandNumber(), kLayoutModeAttr);
      if (!areLayoutsCompatible(argLayout, resultLayoutAttr)) {
        if (propagatedFromResultIdx == -1) {
          emitError(programFunc->getLoc())
              << "Arg #" << arg.getArgNumber() << " is returned as result #"
              << use.getOperandNumber()
              << ", but with incompatible layouts: " << argLayout << " vs. "
              << resultLayoutAttr;
          return false;
        }
        emitError(programFunc->getLoc())
            << "Arg #" << arg.getArgNumber() << " is returned as result #"
            << propagatedFromResultIdx << " and result #"
            << use.getOperandNumber()
            << ", but with incompatible layouts: " << argLayout << " vs. "
            << resultLayoutAttr;
        return false;
      }
      if (!isAutoLayout(resultLayoutAttr) && isAutoLayout(argLayout)) {
        propagatedFromResultIdx = use.getOperandNumber();
        argLayout = resultLayoutAttr;
      }
    }
    if (isAutoLayout(argLayout) && (numUsesInFragments > 1)) {
      argLayout = StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
    }
    if (isDefaultLayout(argLayout)) {
      programFunc.removeArgAttr(arg.getArgNumber(), kLayoutModeAttr);
    } else {
      programFunc.setArgAttr(arg.getArgNumber(), kLayoutModeAttr, argLayout);
    }
    for (OpOperand& use : arg.getUses()) {
      if (FragmentOp fragOp = dyn_cast<FragmentOp>(use.getOwner())) {
        SmallVector<Attribute> fragAttrs = GetArgAttrsOrCreateDefault(fragOp);
        setFragmentLayout(fragAttrs, use.getOperandNumber(), argLayout);
        SetArgAttrs(fragOp, fragAttrs);
      } else if (use.getOwner() == programFunc.front().getTerminator()) {
        if (isDefaultLayout(argLayout)) {
          programFunc.removeResultAttr(use.getOperandNumber(), kLayoutModeAttr);
        } else {
          programFunc.setResultAttr(use.getOperandNumber(), kLayoutModeAttr,
                                    argLayout);
        }
      }
    }
    return true;
  };
};

}  // namespace
}  // namespace mlir::mpmd
