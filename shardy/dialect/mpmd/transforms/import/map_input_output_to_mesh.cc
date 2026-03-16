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
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_inference_origins.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_MAPINPUTOUTPUTTOMESHPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

void CleanUpMemoryKindAttributes(func::FuncOp func) {
  for (auto [index, arg_type] : llvm::enumerate(func.getArgumentTypes())) {
    StringAttr memory_kind_attr =
        func.getArgAttrOfType<StringAttr>(index, kMemoryKindAttr);
    if (!memory_kind_attr) {
      continue;
    }
    if (auto mesh_tensor_type = dyn_cast<MeshTensorType>(arg_type)) {
      if (!mesh_tensor_type.getMemoryKind()) {
        // TODO: b/374994155 - For now, we just ignore this. But going forward,
        // we should move the attribute to the type.
        continue;
      }
      if (mesh_tensor_type.getMemoryKind() != memory_kind_attr) {
        // TODO: b/374994155 - This should be an error once we unify input
        //  assignment with input shardings.
        SDY_LOG(WARNING) << "Memory kind attribute "
                         << memory_kind_attr.getValue().str()
                         << " on function argument " << index
                         << " does not match the memory kind "
                         << mesh_tensor_type.getMemoryKind().getValue().str()
                         << " in its type.";
      }
      func.removeArgAttr(index, kMemoryKindAttr);
    }
  }
  for (auto [index, result_type] : llvm::enumerate(func.getResultTypes())) {
    StringAttr memory_kind_attr =
        func.getResultAttrOfType<StringAttr>(index, kMemoryKindAttr);
    if (!memory_kind_attr) {
      continue;
    }
    if (auto mesh_tensor_type = dyn_cast<MeshTensorType>(result_type)) {
      if (!mesh_tensor_type.getMemoryKind()) {
        // TODO: b/374994155 - For now, we just ignore this. But going forward,
        // we should move the attribute to the type.
        continue;
      }
      if (mesh_tensor_type.getMemoryKind() != memory_kind_attr) {
        // TODO: b/374994155 - This should be an error once we unify output
        //  assignment with output shardings.
        SDY_LOG(WARNING) << "Memory kind attribute "
                         << memory_kind_attr.getValue().str()
                         << " on function result " << index
                         << " does not match the memory kind "
                         << mesh_tensor_type.getMemoryKind().getValue().str()
                         << " in its type.";
      }
      func.removeResultAttr(
          index, StringAttr::get(func.getContext(), kMemoryKindAttr));
    }
  }
}

class MapInputOutputToMeshPass
    : public impl::MapInputOutputToMeshPassBase<MapInputOutputToMeshPass> {
  using MapInputOutputToMeshPassBase::MapInputOutputToMeshPassBase;

 private:
  MeshTensorType GetMeshTensorType(Value value, StringRef mesh_name,
                                   SymbolTable& symbol_table) {
    auto ranked_tensor_type = cast<RankedTensorType>(value.getType());
    std::pair<StringRef, std::optional<StringRef>> mesh_name_and_memory_kind =
        TryToExtractMemoryKindFromMeshName(mesh_name);
    StringRef mesh_name_without_suffix = mesh_name_and_memory_kind.first;
    SDY_CHECK(symbol_table.lookup<sdy::MeshOp>(mesh_name_without_suffix))
        << "Mesh " << mesh_name_without_suffix.str() << " does not exist.";
    StringAttr memory_kind = {};
    if (mesh_name_and_memory_kind.second.has_value()) {
      memory_kind = StringAttr::get(value.getContext(),
                                    mesh_name_and_memory_kind.second.value());
    }
    return MeshTensorType::get(value.getContext(), mesh_name_without_suffix,
                               ranked_tensor_type, memory_kind);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbol_table(module);
    for (func::FuncOp func : GetMpmdFunctions(module)) {
      if (!IsEntryPointFunction(func)) {
        continue;
      }
      IRRewriter rewriter(func->getContext());

      // Go through the input index to mesh name map and assign the mesh to the
      // corresponding argument. Note that the order of this iteration is not
      // deterministic since llvm::DenseMap is unordered but the order does
      // not matter here.
      for (const auto& [input_index, mesh_name] : inputAssignment.value) {
        SDY_CHECK_GE(input_index, 0) << "Input index must be non-negative.";
        SDY_CHECK_LT(input_index, func.getNumArguments())
            << "Input index out of bounds.";

        // Assign the mesh to the argument.
        Value arg = func.getArgument(input_index);
        arg.setType(GetMeshTensorType(arg, mesh_name, symbol_table));

        // Unassign the argument mesh before use.
        rewriter.setInsertionPointAfterValue(arg);
        auto unassign_op = UnassignOp::create(rewriter, arg.getLoc(), arg,
                                              /*origin=*/kUserInputOrigin);
        rewriter.replaceAllUsesExcept(arg, unassign_op.getResult(),
                                      unassign_op);
      }

      // Go through the output index to mesh name map and assign the mesh to the
      // corresponding return value. Note that the order of this iteration is
      // not deterministic since llvm::DenseMap is unordered but the order
      // does not matter here because we only need the assigned ops to be
      // present and any order is valid.
      Operation* return_op = func.getBlocks().back().getTerminator();
      for (const auto& [output_index, mesh_name] : outputAssignment.value) {
        SDY_CHECK_GE(output_index, 0) << "Output index must be non-negative.";
        SDY_CHECK_LT(output_index, func.getNumResults())
            << "Output index must be less than the number of results.";

        rewriter.setInsertionPoint(return_op);
        Value output = return_op->getOperand(output_index);
        auto assign_op = AssignOp::create(
            rewriter,
            GetResultInfoLoc(func, output_index).value_or(output.getLoc()),
            GetMeshTensorType(output, mesh_name, symbol_table), output,
            /*origin=*/kUserOutputOrigin);
        return_op->setOperand(output_index, assign_op.getResult());
      }
      // Update the function signature.
      UpdateFunctionType(func);
      CleanUpMemoryKindAttributes(func);
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
