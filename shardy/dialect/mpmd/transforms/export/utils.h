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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_UTILS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_UTILS_H_

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"

namespace mlir::mpmd {

// Name of attribute used by XLA to alias inputs to outputs. The attribute
// is of type `I32IntegerAttr`, is attached to the input, and stores the
// index of the output the input is aliased with.
constexpr StringRef kAliasingAttrName = "tf.aliasing_output";
// Name of attribute used by XLA to donate inputs. The attribute is of type
// `BoolAttr`, is attached to the input, and is set to true if the input is
// donated. This attribute is set in situations in which no aliasing was found
// for the input, which can happen if we do not sufficient information to
// compute the exact per-device byte size of the input. When this attribute is
// set to true, XLA will try to alias the input with an output using correct
// per-device byte size information.
constexpr StringRef kBufferDonationAttrName = "jax.buffer_donor";

// Name of attribute that will tell XLA how much memory to reserve while
// compiling each fragment.
constexpr StringRef kReservedHbmBytes = "xla_tpu_user_reserved_hbm_bytes";

// Returns the set of user marked block arguments to be aliased.
DenseSet<BlockArgument> GetAliasedBlockArguments(
    func::FuncOp main_func);

// Returns the set of user marked block arguments to donate.
DenseSet<BlockArgument> GetDonatedBlockArguments(
    func::FuncOp main_func);

// Returns a map from a operation to a vector of OpOperand, such that each
// OpOperand is used in the respective operation last. This is done by tracking
// the last OpOperand use of a value in a map.
//
// For each operation with an OpOperand that is used for the last time, we
// store a list of indices of which operand(s) are the last use. This will
// then be used to figure out which operands can be deleted from the live
// range.
DenseMap<Operation*, SmallVector<unsigned int>>
OperandsForDeletionMapping(func::FuncOp main_func);

inline bool IsMemoryKindOnHost(mlir::StringAttr memory_kind) {
  if (!memory_kind) {
    return false;
  }
  mlir::StringRef memory_kind_val = memory_kind.getValue();
  return memory_kind_val == mpmd::kMemoryKindPinnedHost ||
         memory_kind_val == mpmd::kMemoryKindUnpinnedHost;
}


// Checks the arg attrs of the op to see if the arg is on the host.
inline bool IsArgOnHost(mlir::Operation* op, int index) {
  mlir::StringAttr memory_kind_attr = mlir::dyn_cast_or_null<mlir::StringAttr>(
      mlir::mpmd::GetArgAttr(op, index, mlir::mpmd::kMemoryKindAttr));
  return IsMemoryKindOnHost(memory_kind_attr);
}

inline std::optional<mlir::StringRef> GetMemoryKindIfArgOnHost(
    mlir::func::FuncOp func, int index) {
  mlir::StringAttr memory_kind_attr = func.getArgAttrOfType<mlir::StringAttr>(
      index, mlir::mpmd::kMemoryKindAttr);
  if (!memory_kind_attr) {
    return std::nullopt;
  }
  mlir::StringRef memory_kind_val = memory_kind_attr.getValue();
  if (memory_kind_val == mlir::mpmd::kMemoryKindPinnedHost ||
      memory_kind_val == mlir::mpmd::kMemoryKindUnpinnedHost) {
    return memory_kind_val;
  }
  return std::nullopt;
}

inline std::optional<mlir::StringRef> GetMemoryKindIfResultOnHost(
    mlir::OpResult op_result) {
  auto memory_kind = dyn_cast_or_null<mlir::StringAttr>(
      mlir::mpmd::GetResAttr(op_result.getOwner(), op_result.getResultNumber(),
                             mlir::mpmd::kMemoryKindAttr));

  if (!memory_kind) {
    return std::nullopt;
  }
  mlir::StringRef memory_kind_val = memory_kind.getValue();
  if (memory_kind_val == mlir::mpmd::kMemoryKindPinnedHost ||
      memory_kind_val == mlir::mpmd::kMemoryKindUnpinnedHost) {
    return memory_kind_val;
  }
  return std::nullopt;
}

// Checks if the layout of the input and output match.
inline bool IsInputOutputLayoutMatch(Operation* op, int input_index,
                                     int output_index) {
  return GetArgAttr(op, input_index, kLayoutModeAttr) ==
         GetResAttr(op, output_index, kLayoutModeAttr);
}

// Checks if the arg of the function is donated or aliased.
inline bool IsArgDonated(func::FuncOp func, int index) {
  return func.getArgAttrOfType<StringAttr>(index, kAliasingAttrName) ||
         func.getArgAttrOfType<BoolAttr>(index,
                                               kBufferDonationAttrName) ==
             BoolAttr::get(func.getContext(), true);
}

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_EXPORT_UTILS_H_
