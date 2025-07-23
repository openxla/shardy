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
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_MARKINPUTOUTPUTWITHLAYOUTSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

bool IsAutoLayout(StringAttr layout) {
  return layout && layout == kLayoutModeAuto;
}

// A layout attribute denotes a default layout if it is empty or if it is a
// `default` string.
bool IsDefaultLayout(StringAttr layout) {
  return !layout || layout == kLayoutModeDefault;
}

// Two layouts are compatible if: 1) they are both default layouts, 2) they are
// identical custom layouts, or 3) one or both are auto layouts.
bool AreLayoutsCompatible(StringAttr layout1, StringAttr layout2) {
  if (IsDefaultLayout(layout1) && IsDefaultLayout(layout2)) {
    return true;
  }
  if (IsAutoLayout(layout1) || IsAutoLayout(layout2)) {
    return true;
  }
  return layout1 == layout2;
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

    // Propagate layouts for func args that are returned from the func.
    // It is necessary to do this before propagating layouts from func args or
    // results to fragment arg or results because this function might refine
    // func arg/result layouts (e.g., an auto arg layout is converted to a
    // custom layout if the arg's corresponding result has a custom layout).
    if (!PropagateLayoutsForReturnedFuncArgs(func)) {
      signalPassFailure();
      return;
    }

    // Propagate layouts from func args/results to fragment arg/results.
    for (FragmentOp frag : func.getOps<FragmentOp>()) {
      if (!PropagateFuncLayoutsToFragments(frag, func)) {
        signalPassFailure();
        return;
      }
    }
  }

  // Propagates layouts from func args/results to fragment arg/results.
  // Returns false if a fragment result is returned multiple times with
  // incompatible layouts. See `AreLayoutsCompatible` for the definition of when
  // layouts are considered compatible.
  bool PropagateFuncLayoutsToFragments(FragmentOp frag, FuncOp parent) {
    SmallVector<Attribute> arg_attrs = GetArgAttrsOrCreateDefault(frag);
    for (OpOperand& operand : frag->getOpOperands()) {
      if (auto block_arg = dyn_cast<BlockArgument>(operand.get())) {
        SDY_CHECK_EQ(block_arg.getOwner()->getParentOp(), parent);
        if (auto layout_attr = parent.getArgAttrOfType<StringAttr>(
                block_arg.getArgNumber(), kLayoutModeAttr)) {
          if (IsDefaultLayout(layout_attr)) {
            // Remove the layout attribute because the default layout is
            // implicit.
            parent.removeArgAttr(
                block_arg.getArgNumber(),
                StringAttr::get(parent.getContext(), kLayoutModeAttr));
          } else {
            InsertAttr(arg_attrs[operand.getOperandNumber()], kLayoutModeAttr,
                       layout_attr);
          }
        }
      }
    }
    SetArgAttrs(frag, arg_attrs);

    SmallVector<Attribute> res_attrs = GetResAttrsOrCreateDefault(frag);
    for (const OpResult& res : frag->getOpResults()) {
      StringAttr frag_result_layout_attr;
      int propagated_from_result_idx = -1;
      for (OpOperand& operand : res.getUses()) {
        // Only propagate result layouts from the func op to the fragment op.
        if (operand.getOwner() != parent.front().getTerminator()) {
          continue;
        }
        auto func_result_layout_attr = parent.getResultAttrOfType<StringAttr>(
            operand.getOperandNumber(), kLayoutModeAttr);
        if (propagated_from_result_idx == -1 ||
            IsAutoLayout(frag_result_layout_attr)) {
          // Propagate from the func result if no layout has been propagated
          // yet or the propagated layout is auto. It is always safe to
          // propagate to auto layouts because in the worst case we'll get
          // another auto layout.
          propagated_from_result_idx = operand.getOperandNumber();
          frag_result_layout_attr = func_result_layout_attr;
        } else if (!AreLayoutsCompatible(func_result_layout_attr,
                                         frag_result_layout_attr)) {
          emitError(operand.getOwner()->getLoc())
              << "Result #" << operand.getOperandNumber()
              << " is also returned as result #" << propagated_from_result_idx
              << ", but with incompatible layouts: " << func_result_layout_attr
              << " vs. " << frag_result_layout_attr;
          return false;
        }
      }

      if (propagated_from_result_idx > -1 && frag_result_layout_attr) {
        if (!IsDefaultLayout(frag_result_layout_attr)) {
          InsertAttr(res_attrs[res.getResultNumber()], kLayoutModeAttr,
                     frag_result_layout_attr);
        }
        // Update the parent result layout in case the fragment result is
        // returned multiple times, and this has helped refined the result
        // layouts further.
        for (OpOperand& operand : res.getUses()) {
          if (operand.getOwner() != parent.front().getTerminator()) {
            continue;
          }
          if (IsDefaultLayout(frag_result_layout_attr)) {
            parent.removeResultAttr(
                operand.getOperandNumber(),
                StringAttr::get(parent.getContext(), kLayoutModeAttr));
          } else {
            parent.setResultAttr(operand.getOperandNumber(), kLayoutModeAttr,
                                 frag_result_layout_attr);
          }
        }
      }
    }
    SetResAttrs(frag, res_attrs);
    return true;
  }

  // Propagates layouts for func args that are returned from the func. Emits
  // an error and returns false if a func arg and its corresponding results have
  // incompatible layouts. Otherwise, it finds the layout that is compatible and
  // possibly specified (i.e., device and custom layouts) from the func arg and
  // its corresponding results layouts, and updates their attributes to this
  // layout.
  bool PropagateLayoutsForReturnedFuncArgs(FuncOp func) {
    for (BlockArgument arg : func.getArguments()) {
      if (arg.use_empty()) {
        continue;
      }

      // Find the possibly specified and compatible layout from the results
      // corresponding to the arg.
      StringAttr res_layout_attr;
      int propagated_from_result_idx = -1;
      for (OpOperand& use : arg.getUses()) {
        if (!isa<func::ReturnOp>(use.getOwner())) {
          continue;
        }
        auto func_res_layout_attr = func.getResultAttrOfType<StringAttr>(
            use.getOperandNumber(), kLayoutModeAttr);
        if (propagated_from_result_idx == -1 || IsAutoLayout(res_layout_attr)) {
          res_layout_attr = func_res_layout_attr;
          propagated_from_result_idx = use.getOperandNumber();
        } else if (!AreLayoutsCompatible(res_layout_attr,
                                         func_res_layout_attr)) {
          emitError(func->getLoc())
              << "Arg #" << arg.getArgNumber() << " is returned as result #"
              << propagated_from_result_idx << " and result #"
              << use.getOperandNumber()
              << ", but with incompatible layouts: " << res_layout_attr
              << " vs. " << func_res_layout_attr;
          return false;
        }
      }

      // No propagation is required because the arg is not returned.
      if (propagated_from_result_idx == -1) {
        continue;
      }
      if (auto arg_layout_attr = func.getArgAttrOfType<StringAttr>(
              arg.getArgNumber(), kLayoutModeAttr);
          AreLayoutsCompatible(res_layout_attr, arg_layout_attr)) {
        if (IsAutoLayout(arg_layout_attr)) {
          arg_layout_attr = res_layout_attr;
        }
        // Update the arg layout and the corresponding result layouts to the
        // compatible layout. If the compatible layout is device default then
        // we remove the layout attribute because the default layout is
        // implicit.
        if (IsDefaultLayout(arg_layout_attr)) {
          func.removeArgAttr(arg.getArgNumber(), kLayoutModeAttr);
        } else {
          func.setArgAttr(arg.getArgNumber(), kLayoutModeAttr, arg_layout_attr);
        }
        for (OpOperand& use : arg.getUses()) {
          if (isa<func::ReturnOp>(use.getOwner())) {
            if (IsDefaultLayout(arg_layout_attr)) {
              func.removeResultAttr(
                  use.getOperandNumber(),
                  StringAttr::get(func.getContext(), kLayoutModeAttr));
            } else {
              func.setResultAttr(use.getOperandNumber(), kLayoutModeAttr,
                                 arg_layout_attr);
            }
          }
        }
      } else {
        emitError(func->getLoc())
            << "Arg #" << arg.getArgNumber() << " is returned as result #"
            << propagated_from_result_idx
            << ", but with incompatible layouts: " << arg_layout_attr << " vs. "
            << res_layout_attr;
      }
    }
    return true;
  }
};

}  // namespace
}  // namespace mlir::mpmd
