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

#include <cmath>
#include <cstdint>
#include <string_view>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/export/utils.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_MARKFRAGMENTRESERVEDMEMORYPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

using OperandLastUseMap = DenseMap<Operation*, SmallVector<unsigned int>>;

// Mapping from the mesh name to how much memory is being used on it (in bytes).
using LiveMemoryMap = DenseMap<StringRef, double>;

// Gets the size of the tensor in bytes.
double GetMeshTensorSize(MeshTensorType mesh_tensor_type, Operation* op) {
  RankedTensorType local_tensor_type = mesh_tensor_type.getLocalTensorType(op);
  return local_tensor_type.getNumElements() *
         (static_cast<double>(local_tensor_type.getElementTypeBitWidth()) / 8);
}

// Gets the current amount of live memory on the mesh the fragment lives
// on. But any operands need to be excluded from this calculation since
// XLA already knows about them.
template <typename FragOpT>
int64_t GetLiveMemoryBytes(FragOpT op, OpBuilder& builder,
                           LiveMemoryMap current_memory_usage_per_mesh) {
  auto it_memory = current_memory_usage_per_mesh.find(op.getMeshName());
  SDY_CHECK(it_memory != current_memory_usage_per_mesh.end())
      << "Required mesh_name missing: " << std::string_view(op.getMeshName());
  double current_size_bytes = it_memory->second;

  llvm::DenseSet<Value> already_seen;
  for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
    // If a fragment takes a value multiple times, then only subtract from
    // the size of live buffers once. We also don't subtract args on host, since
    // they're not in HBM.
    if (already_seen.contains(operand) || IsArgOnHost(op, index)) {
      continue;
    }
    already_seen.insert(operand);

    MeshTensorType mesh_tensor_type = cast<MeshTensorType>(operand.getType());
    current_size_bytes -=
        GetMeshTensorSize(mesh_tensor_type, op.getOperation());
  }
  // Don't expect a program to have values less than a byte, but better to
  // be safe.
  return ceil(current_size_bytes);
}

class MarkFragmentReservedMemoryPass
    : public impl::MarkFragmentReservedMemoryPassBase<
          MarkFragmentReservedMemoryPass> {
  using MarkFragmentReservedMemoryPassBase::MarkFragmentReservedMemoryPassBase;

 private:
  void runOnFunc(func::FuncOp main_func) override {
    if (!mpmd::IsMpmdFunction(main_func)) {
      return;
    }

    LiveMemoryMap current_memory_usage_per_mesh;
    Liveness liveness(main_func);

    // Set initial bytes on all meshes to 0.
    ArrayRef<mpmd::NamedMeshAttr> topology = mpmd::GetTopologyMeshes(main_func);
    for (auto mesh : topology) {
      auto [_, inserted] =
          current_memory_usage_per_mesh.try_emplace(mesh.getName(), 0);
      SDY_CHECK(inserted) << "Expected mapping to be empty, already saw mesh '"
                          << std::string_view(mesh.getName()) << "'.";
    }

    for (BlockArgument& arg : main_func.getArguments()) {
      // Do not add live values for args that are on the host or that are
      // donated and not used.
      if (!IsArgOnHost(main_func, arg.getArgNumber()) &&
          (!IsArgDonated(main_func, arg.getArgNumber()) || !arg.use_empty())) {
        MeshTensorType type = cast<MeshTensorType>(arg.getType());
        AddLiveValue(type, current_memory_usage_per_mesh, main_func);
      }
    }

    // Traversal of all ops is in order. Can add to tracked memory usage for
    // every new result, and remove any values which are the last usage.
    OpBuilder builder(main_func.getContext());
    for (Operation& op : main_func.getOps()) {
      if (auto fragment_op = dyn_cast<FragmentOp>(op)) {
        fragment_op->setAttr(
            kReservedHbmBytes,
            IntegerAttr::get(
                builder.getI64Type(),
                GetLiveMemoryBytes(fragment_op, builder,
                                   current_memory_usage_per_mesh)));
      }

      if (isa<TransferOp>(op) || isa<FragmentOp>(op)) {
        // Remove any of the operands which are the last use. If an operand is
        // an argument of the MPMD program then we can only subtract it here
        // if it has been donated to the program. Otherwise, we cannot safely
        // subtract it because we do not know if there are other references to
        // it outside of the program that would keep it alive. We take the safe
        // option here and possibly overestimate live buffers.
        llvm::DenseSet<Value> already_seen;
        for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
          if (!IsArgOnHost(&op, index) && already_seen.insert(operand).second &&
              liveness.isDeadAfter(operand, &op)) {
            if (auto block_arg = dyn_cast_or_null<BlockArgument>(operand);
                !block_arg ||
                IsArgDonated(main_func, block_arg.getArgNumber())) {
              MeshTensorType mesh_tensor_type =
                  cast<MeshTensorType>(operand.getType());
              SubtractLastUse(mesh_tensor_type, current_memory_usage_per_mesh,
                              &op);
            }
          }
        }
        // Add sizes of the results which are now live.
        for (OpResult result : op.getResults()) {
          if (!GetMemoryKindIfResultOnHost(result).has_value() && !result.use_empty()) {
            MeshTensorType type = cast<MeshTensorType>(result.getType());
            AddLiveValue(type, current_memory_usage_per_mesh, &op);
          }
        }
      } else if (isa<func::ReturnOp>(op)) {
        continue;
      } else {
        op.emitError(
            "Expected only TransferOp, FragmentOp, FragmentCallOp and ReturnOp "
            "in the function body.");
        signalPassFailure();
      }
    }
  }

  void AddLiveValue(MeshTensorType type,
                    LiveMemoryMap& current_memory_usage_per_mesh,
                    Operation* op) {
    StringRef mesh_name = type.getMeshName();
    auto it_memory = current_memory_usage_per_mesh.find(mesh_name);
    SDY_CHECK(it_memory != current_memory_usage_per_mesh.end())
        << "Required mesh_name missing: " << std::string_view(mesh_name);
    it_memory->second += GetMeshTensorSize(type, op);
  }

  void SubtractLastUse(MeshTensorType type,
                       LiveMemoryMap& current_memory_usage_per_mesh,
                       Operation* op) {
    StringRef mesh_name = type.getMeshName();
    auto it_memory = current_memory_usage_per_mesh.find(mesh_name);
    SDY_CHECK(it_memory != current_memory_usage_per_mesh.end())
        << "Required mesh_name missing: " << std::string_view(mesh_name);
    it_memory->second -= GetMeshTensorSize(type, op);
  }
};

}  // namespace
}  // namespace mlir::mpmd
