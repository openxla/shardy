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

#include <optional>

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

void setFunctionResultLayout(int resultIndex, StringAttr layout,
                             FuncOp& programFunc) {
  if (isDefaultLayout(layout)) {
    programFunc.removeResultAttr(resultIndex, kLayoutModeAttr);
  } else {
    programFunc.setResultAttr(resultIndex, kLayoutModeAttr, layout);
  }
}

void setFunctionArgumentLayout(int argumentIndex, StringAttr layout,
                               FuncOp& programFunc) {
  if (isDefaultLayout(layout)) {
    programFunc.removeArgAttr(argumentIndex, kLayoutModeAttr);
  } else {
    programFunc.setArgAttr(argumentIndex, kLayoutModeAttr, layout);
  }
}

void setLayoutForUsers(const Value& value, const StringAttr layout,
                       FuncOp& programFunc) {
  for (OpOperand& use : value.getUses()) {
    if (FragmentOp fragOp = dyn_cast<FragmentOp>(use.getOwner())) {
      SmallVector<Attribute> fragAttrs = GetArgAttrsOrCreateDefault(fragOp);
      setFragmentLayout(fragAttrs, use.getOperandNumber(), layout);
      SetArgAttrs(fragOp, fragAttrs);
    } else if (use.getOwner() == programFunc.front().getTerminator()) {
      setFunctionResultLayout(use.getOperandNumber(), layout, programFunc);
    }
  }
};

struct CommonLayoutResult {
  int numUsesInFragments;
  int propagatedFromProgramResIdx;
  bool hasTransferOpUse;
  StringAttr commonLayout;
};

std::optional<CommonLayoutResult> extractCommonLayoutFromUsers(
    const mlir::Value& value, FuncOp& programFunc) {
  StringAttr commonLayout =
      StringAttr::get(programFunc.getContext(), kLayoutModeAuto);
  int numUsesInFragments = 0;
  int propagatedFromProgramResIdx = -1;
  bool hasTransferOpUse = false;
  for (OpOperand& use : value.getUses()) {
    if (isa<TransferOp>(use.getOwner())) {
      hasTransferOpUse = true;
      continue;
    }
    if (use.getOwner() != programFunc.front().getTerminator()) {
      numUsesInFragments++;
      continue;
    }
    StringAttr programResLayoutAttr =
        programFunc.getResultAttrOfType<StringAttr>(use.getOperandNumber(),
                                                    kLayoutModeAttr);
    if (!areLayoutsCompatible(commonLayout, programResLayoutAttr)) {
      emitError(use.getOwner()->getLoc())
          << "Result #" << propagatedFromProgramResIdx
          << " is also returned as result #" << use.getOperandNumber()
          << ", but with incompatible layouts: " << commonLayout << " vs. "
          << programResLayoutAttr;
      return std::nullopt;
    }
    if (!isAutoLayout(programResLayoutAttr)) {
      commonLayout = programResLayoutAttr;
      propagatedFromProgramResIdx = use.getOperandNumber();
    }
  }
  return CommonLayoutResult{numUsesInFragments, propagatedFromProgramResIdx,
                            hasTransferOpUse, commonLayout};
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
  if (fragRes.use_empty()) {
    // the fragment result layouts are not set by default - meaning they are
    // auto
    setFragmentLayout(
        fragResAttrs, fragRes.getResultNumber(),
        StringAttr::get(programFunc.getContext(), kLayoutModeAuto));
    return true;
  }
  std::optional<CommonLayoutResult> layoutExtractionResult =
      extractCommonLayoutFromUsers(
          /*value=*/fragRes, programFunc);
  if (!layoutExtractionResult.has_value()) {
    return false;
  }
  StringAttr chosenLayout = layoutExtractionResult->commonLayout;

  if (layoutExtractionResult->hasTransferOpUse) {
    // if used in a transfer op, we *MUST* stick to default layout, since
    // ifrt transfers only support default layouts
    // TODO(icgog): b/450351477 ifrt support non-default layouts
    StringAttr defaultLayout =
        StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
    if (!areLayoutsCompatible(chosenLayout, defaultLayout)) {
      emitWarning(fragRes.getLoc())
          << "Result #" << fragRes.getResultNumber()
          << " is used in a transfer op, but with a non-default layout: "
          << chosenLayout << ". Forcing the layout to default.";
    }
    chosenLayout = defaultLayout;
  } else if (isAutoLayout(chosenLayout) &&
             (layoutExtractionResult->numUsesInFragments > 0)) {
    chosenLayout =
        StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
  }
  setFragmentLayout(fragResAttrs, fragRes.getResultNumber(), chosenLayout);
  setLayoutForUsers(fragRes, chosenLayout, programFunc);
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
  if (arg.use_empty()) {
    return true;
  }
  // init with given layout
  StringAttr chosenLayout = programFunc.getArgAttrOfType<StringAttr>(
      arg.getArgNumber(), kLayoutModeAttr);

  std::optional<CommonLayoutResult> layoutExtractionResult =
      extractCommonLayoutFromUsers(
          /*value=*/arg, programFunc);
  if (!layoutExtractionResult.has_value()) {
    return false;
  }
  if (!areLayoutsCompatible(chosenLayout,
                            layoutExtractionResult->commonLayout)) {
    emitError(programFunc->getLoc())
        << "Arg #" << arg.getArgNumber() << " is returned as result #"
        << layoutExtractionResult->propagatedFromProgramResIdx
        << ", but with incompatible layouts: " << chosenLayout << " vs. "
        << layoutExtractionResult->commonLayout;
    return false;
  }
  if (!isAutoLayout(layoutExtractionResult->commonLayout)) {
    chosenLayout = layoutExtractionResult->commonLayout;
  }

  if (layoutExtractionResult->hasTransferOpUse) {
    // if used in a transfer op, we *MUST* stick to default layout, since
    // ifrt transfers only support default layouts
    // TODO(icgog): b/450351477 ifrt support non-default layouts
    StringAttr defaultLayout =
        StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
    if (!areLayoutsCompatible(chosenLayout, defaultLayout)) {
      emitWarning(arg.getLoc())
          << "Arg #" << arg.getArgNumber()
          << " is used in a transfer op, but with a non-default layout: "
          << chosenLayout << ". Forcing the layout to default.";
    }
    chosenLayout = defaultLayout;
  } else if (isAutoLayout(chosenLayout) &&
             (layoutExtractionResult->numUsesInFragments > 1)) {
    chosenLayout =
        StringAttr::get(programFunc.getContext(), kLayoutModeDefault);
  }
  setFunctionArgumentLayout(arg.getArgNumber(), chosenLayout, programFunc);
  setLayoutForUsers(arg, chosenLayout, programFunc);
  return true;
};

class MarkInputOutputWithLayoutsPass
    : public impl::MarkInputOutputWithLayoutsPassBase<
          MarkInputOutputWithLayoutsPass> {
  using MarkInputOutputWithLayoutsPassBase::MarkInputOutputWithLayoutsPassBase;

 protected:
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
};

}  // namespace
}  // namespace mlir::mpmd
