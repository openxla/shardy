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

#include <cstdint>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/export/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_MARKOFFLOADEDINPUTOUTPUTPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;
using ::mlir::stablehlo::CustomCallOp;

// Name of the custom_call which indicates moving data from host to device
// or device to host. E.g. %host_v = mhlo.custom_call
// @annotate_device_placement(%v) {
//   mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
// }
inline constexpr StringRef kOffloadCustomCallName = "annotate_device_placement";
// Name of Attr which holds further information in a dict.
inline constexpr StringRef kMhloFrontendAttr = "mhlo.frontend_attributes";
// Attr in the `kMhloFrontendAttr` dict attr which indicates where the buffer is
// on: host or device.
inline constexpr StringRef kXlaBufferPlacementAttr = "_xla_buffer_placement";
// Attr in the `kMhloFrontendAttr` dict attr which indicates where the compute
// will take place: host or device.
inline constexpr StringRef kXlaComputeTypeAttr = "_xla_compute_type";
inline constexpr StringRef kXlaComputeTypeHost = "host";

// These are compatible custom calls that we have identified are compatible with
// offload. The list may not be exhaustive. The sharding custom_calls are
// compatible because they are just annotations for sharding, and are noops
// otherwise.
bool IsOffloadCompatibleCustomCall(Operation* op) {
  if (auto custom_call = dyn_cast<CustomCallOp>(op)) {
    return custom_call.getCallTargetName() == "Sharding" ||
           custom_call.getCallTargetName() == "SPMDShardToFullShape" ||
           custom_call.getCallTargetName() == "SPMDFullToShardShape";
  }
  return false;
}

// Returns true if the reshape just modifies trivial dims, i.e. dims of size 1.
bool IsTrivialReshape(Operation* op) {
  if (auto reshape = dyn_cast<stablehlo::ReshapeOp>(op)) {
    // Check equality excluding dims of size 1.
    return llvm::equal(
        llvm::make_filter_range(reshape.getResult().getType().getShape(),
                                [](int64_t dim) { return dim != 1; }),
        llvm::make_filter_range(reshape.getOperand().getType().getShape(),
                                [](int64_t dim) { return dim != 1; }));
  }
  if (auto broadcast = dyn_cast<stablehlo::BroadcastInDimOp>(op)) {
    // Check equality excluding dims of size 1.
    return llvm::equal(
        llvm::make_filter_range(broadcast.getResult().getType().getShape(),
                                [](int64_t dim) { return dim != 1; }),
        llvm::make_filter_range(broadcast.getOperand().getType().getShape(),
                                [](int64_t dim) { return dim != 1; }));
  }
  return false;
}

// Retrieves the offload destination of `op` if it is an offload custom call.
std::optional<StringRef> GetOffloadValueIfExists(Operation* op) {
  auto custom_call = dyn_cast_if_present<CustomCallOp>(op);
  if (!custom_call ||
      custom_call.getCallTargetName() != kOffloadCustomCallName) {
    return std::nullopt;
  }

  auto frontend_attr =
      custom_call->getAttrOfType<DictionaryAttr>(kMhloFrontendAttr);
  SDY_CHECK(frontend_attr);
  auto buffer_attr = frontend_attr.getAs<StringAttr>(kXlaBufferPlacementAttr);
  SDY_CHECK(buffer_attr);
  return buffer_attr.getValue();
}

// Returns the operand corresponding to the result of the op, for ops which are
// compatible with host offload. If the op is not compatible, return nullptr;
Value WalkBackwardThroughOffloadCompatibleResult(OpResult res) {
  Operation* op = res.getOwner();
  if (auto barrier = dyn_cast<stablehlo::OptimizationBarrierOp>(op)) {
    return barrier->getOperand(res.getResultNumber());
  }
  if (auto while_op = dyn_cast<stablehlo::WhileOp>(op)) {
    return while_op.getBody().front().getTerminator()->getOperand(
        res.getResultNumber());
  }
  if (auto dynamic_update_slice = dyn_cast<stablehlo::DynamicUpdateSliceOp>(op);
      dynamic_update_slice && dynamic_update_slice.getResult() == res) {
    return dynamic_update_slice.getUpdate();
  }
  if (IsTrivialReshape(op) || IsOffloadCompatibleCustomCall(op)) {
    return op->getOperand(0);
  }
  if (auto call = dyn_cast<func::CallOp>(op)) {
    auto callee = cast<FuncOp>(cast<CallOpInterface>(*call).resolveCallable());
    return callee.front().getTerminator()->getOperand(res.getResultNumber());
  }
  return nullptr;
}

// Returns whether a value is a result, and whether the result is stored on the
// host memory via an annotate custom call on the source of the result, or if
// the result was computed on host.
bool IsResultAndOnHostMemory(Value val, mlir::StringRef memory_kind) {
  auto res = mlir::dyn_cast_if_present<OpResult>(val);
  if (!res) {
    return false;
  }

  if (auto mhlo_frontend_attr =
          res.getOwner()->getAttrOfType<DictionaryAttr>(kMhloFrontendAttr)) {
    if (auto compute_type =
            mhlo_frontend_attr.getAs<StringAttr>(kXlaComputeTypeAttr)) {
      return compute_type.getValue() == kXlaComputeTypeHost;
    }
  }
  if (Value operand = WalkBackwardThroughOffloadCompatibleResult(res)) {
    return IsResultAndOnHostMemory(operand, memory_kind);
  }

  return GetOffloadValueIfExists(res.getOwner()) == memory_kind;
}

std::optional<StringRef> GetOnHostMemoryKindIfResult(Value val) {
  if (IsResultAndOnHostMemory(val, kMemoryKindPinnedHost)) {
    return kMemoryKindPinnedHost;
  }
  if (IsResultAndOnHostMemory(val, kMemoryKindUnpinnedHost)) {
    return kMemoryKindUnpinnedHost;
  }
  return std::nullopt;
}

// Gets memory kind from user.
//
// We don't currently consider transfers, because we don't currently
// propagate host memory kinds across transfers.
Attribute GetMemoryKindFromUser(OpOperand& use, FuncOp func) {
  if (isa<FragmentOp>(use.getOwner())) {
    return GetArgAttr(use.getOwner(), use.getOperandNumber(), kMemoryKindAttr);
  }
  if (isa<func::ReturnOp>(use.getOwner())) {
    return func.getResultAttr(use.getOperandNumber(), kMemoryKindAttr);
  }
  return nullptr;
}

// Retrieve memory kind from the users of `arg`, checking that they all match.
// Emits an error otherwise.
StringAttr GetMemoryKindFromUsers(BlockArgument arg, FuncOp func,
                                  bool& has_error) {
  if (arg.use_empty()) {
    return nullptr;
  }

  OpOperand& first_use = *arg.getUses().begin();
  Attribute memory_kind = GetMemoryKindFromUser(first_use, func);
  for (OpOperand& use : llvm::drop_begin(arg.getUses())) {
    Attribute user_memory_kind = GetMemoryKindFromUser(use, func);

    if (memory_kind != user_memory_kind) {
      emitError(arg.getLoc())
          << "Memory kind mismatch between users of arg " << arg.getArgNumber()
          << ": " << memory_kind << " vs " << user_memory_kind << "\n"
          << PrintOperationForLog(first_use.getOwner(),
                                  OpPrintingFlags().skipRegions())
          << " vs \n"
          << PrintOperationForLog(use.getOwner(),
                                  OpPrintingFlags().skipRegions());
      has_error = true;
    }
  }

  return dyn_cast_if_present<StringAttr>(memory_kind);
}

// Retrieve memory kind from the defining op of `val`.
StringAttr GetMemoryKindFromDefiningOp(Value val, FuncOp func) {
  if (auto block_arg = dyn_cast<BlockArgument>(val)) {
    return func.getArgAttrOfType<StringAttr>(block_arg.getArgNumber(),
                                             kMemoryKindAttr);
  }

  auto op_result = cast<OpResult>(val);
  if (auto frag = dyn_cast<FragmentOp>(op_result.getOwner())) {
    return dyn_cast_if_present<StringAttr>(
        GetResAttr(frag, op_result.getResultNumber(), kMemoryKindAttr));
  }

  return nullptr;
}

bool IsOnDevice(StringAttr memory_kind) {
  return !memory_kind || memory_kind.getValue() == kMemoryKindDevice;
}

bool HasMismatchedMemories(StringAttr memory_a, StringAttr memory_b) {
  // Memory is either on device or host.
  return IsOnDevice(memory_a) != IsOnDevice(memory_b);
}

class MarkOffloadedInputOutputPass
    : public impl::MarkOffloadedInputOutputPassBase<
          MarkOffloadedInputOutputPass> {
  using MarkOffloadedInputOutputPassBase::MarkOffloadedInputOutputPassBase;

 private:
  void runOnFunc(FuncOp func) final {
    bool has_error = false;

    if (!IsEntryPointFunction(func)) {
      return;
    }

    // Mark func body first.
    for (FragmentOp frag : func.getOps<FragmentOp>()) {
      PropagateHostMemoryKindOnFragments(frag, func);
    }

    // Mark func args.
    for (BlockArgument arg : func.getArguments()) {
      StringAttr arg_memory_kind = func.getArgAttrOfType<StringAttr>(
          arg.getArgNumber(), kMemoryKindAttr);
      if (arg.use_empty()) {
        continue;
      }
      StringAttr user_memory_kind =
          GetMemoryKindFromUsers(arg, func, has_error);
      if (HasMismatchedMemories(arg_memory_kind, user_memory_kind)) {
        emitError(arg.getLoc())
            << "Memory kind mismatch between arg " << arg.getArgNumber()
            << " and users: " << arg_memory_kind << " vs " << user_memory_kind;
        has_error = true;
      }
    }

    // Mark func results.
    for (OpOperand& return_operand :
         func.front().getTerminator()->getOpOperands()) {
      StringAttr res_memory_kind = func.getResultAttrOfType<StringAttr>(
          return_operand.getOperandNumber(), kMemoryKindAttr);
      StringAttr defining_op_memory_kind =
          GetMemoryKindFromDefiningOp(return_operand.get(), func);
      if (HasMismatchedMemories(res_memory_kind, defining_op_memory_kind)) {
        emitError(return_operand.getOwner()->getLoc())
            << "Memory kind mismatch between result "
            << return_operand.getOperandNumber()
            << " and defining op: " << res_memory_kind << " vs "
            << defining_op_memory_kind;
        has_error = true;
      }
    }

    if (has_error) {
      signalPassFailure();
    }
  }

  // This propagates "host" memory kinds for:
  // - Func args which are on host.
  // - Frag args where the operands are on host (e.g. result of an offloaded
  // fragment result).
  // - Frag results which come from "activation offloaded" values.
  // - Frag results where the return value was computed on host.
  //
  // It does not currently handle:
  // - Propagating host annotation across transfers.
  // - Propagating host annotation from fragment args to fragment result.
  // - Propagating host annotation through ops implicitly computed on host.
  //
  // We don't propagate device memory kinds, because that's the default.

  void PropagateHostMemoryKindOnFragments(FragmentOp frag, FuncOp parent) {
    SmallVector<Attribute> arg_attrs = GetArgAttrsOrCreateDefault(frag);
    for (OpOperand& operand : frag->getOpOperands()) {
      if (auto result = mlir::dyn_cast<OpResult>(operand.get());
          result && mlir::isa<FragmentOp>(result.getOwner())) {
        if (std::optional<mlir::StringRef> memory_kind =
          GetMemoryKindIfResultOnHost(result)) {
          mlir::mpmd::InsertAttr(
              arg_attrs[operand.getOperandNumber()], kMemoryKindAttr,
              mlir::StringAttr::get(frag.getContext(), memory_kind.value()));
          continue;
        }
      }
      if (auto block_arg = mlir::dyn_cast<BlockArgument>(operand.get())) {
        if (std::optional<mlir::StringRef> memory_kind =
          GetMemoryKindIfArgOnHost(parent, block_arg.getArgNumber())) {
          mlir::mpmd::InsertAttr(
              arg_attrs[operand.getOperandNumber()], kMemoryKindAttr,
              mlir::StringAttr::get(frag.getContext(), memory_kind.value()));
        }
        continue;
      }
    }
    SetArgAttrs(frag, arg_attrs);

    SmallVector<Attribute> res_attrs = GetResAttrsOrCreateDefault(frag);
    for (auto [idx, return_operand] : llvm::enumerate(
             frag.getRegion().front().getTerminator()->getOperands())) {
      if (std::optional<StringRef> memory_kind =
              GetOnHostMemoryKindIfResult(return_operand)) {
        mlir::mpmd::InsertAttr(
            res_attrs[idx], kMemoryKindAttr,
            mlir::StringAttr::get(frag.getContext(), memory_kind.value()));
      }
    }
    SetResAttrs(frag, res_attrs);
  }
};

}  // namespace
}  // namespace mlir::mpmd
