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
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/common/utils.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_inference_utils.h"
#include "shardy/dialect/mpmd/transforms/import/meshes_with_origins.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep

namespace mlir::mpmd {

#define GEN_PASS_DEF_INFERMESHVALIDATESRCSETNOTEMPTYPASS
#define GEN_PASS_DEF_INFERMESHVALIDATENOADDITIONALTRANSFERSNEEDEDPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

int GetValidatedMaxErrors(int error_limit) {
  SDY_CHECK_NE(error_limit, 0)
      << "mesh inference validation must not be disabled.";
  return error_limit == -1 ? std::numeric_limits<int>::max() : error_limit;
}

namespace {

using ::llvm::DenseMap;
using ::mlir::func::FuncOp;

inline constexpr StringRef kErrorMessageSeparator =
    "\n---------------------------------------------------\n";

enum class ValidationResult {
  kOk,
  kError,
  kErrorButDontEmit,
};

StringRef SanitizeCallName(StringRef call_name) {
  const StringRef mpmd_prefix = "shardy_mpmd";
  if (call_name.starts_with(mpmd_prefix)) {
    return call_name.drop_front(mpmd_prefix.size());
  }
  return call_name;
}

// Returns and caches the call ops for a given callee, since this is expensive
// to compute. This lazily computes the call ops, since it is quite expensive
// to do so, and we only need it for error reporting. I.e. we want to avoid
// doing this computation in the case where there's no error.
SmallVector<CallOp> GetMaybeCachedCallOps(
    FuncOp callee,
    DenseMap<StringRef, SmallVector<CallOp>>& lazy_call_ops_by_callee_) {
  if (auto it = lazy_call_ops_by_callee_.find(callee.getSymName());
      it != lazy_call_ops_by_callee_.end()) {
    return it->second;
  }

  lazy_call_ops_by_callee_[callee.getSymName()] = GetCallOps(callee);
  return lazy_call_ops_by_callee_[callee.getSymName()];
}

using LocToOrigins =
    SmallVector<std::pair<Location, SmallVector<MeshWithOriginsAttr>>>;

// Walk through the MLIR module and get the origins of the src_set of `value`,
// with the MLIR location. All values in the src_set must arise from either
// entrypoint func args or mpmd.unassign ops, so we just need to walk through.
LocToOrigins GetSrcOriginsWithLocs(
    ValueRange root,
    DenseMap<StringRef, SmallVector<CallOp>>& lazy_call_ops_by_callee) {
  LocToOrigins mesh_origins_with_locs;

  std::vector<Value> worklist(root.begin(), root.end());
  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current_value = worklist.back();
    worklist.pop_back();
    if (visited.contains(current_value)) {
      continue;
    }
    visited.insert(current_value);

    // Case 1: BlockArgument
    if (auto block_arg = dyn_cast<BlockArgument>(current_value)) {
      Operation* parent_op = block_arg.getOwner()->getParentOp();
      // Mesh inference only runs on values directly in func ops or for-ops.
      // TODO: b/421069658 - Add support for block arguments of for-ops.
      SDY_CHECK(!isa<ForOp>(parent_op)) << "Not supported yet.";
      auto func_op = cast<FuncOp>(parent_op);

      // Case 1a: BlockArgument of the entrypoint function.
      if (IsEntryPointFunction(func_op)) {
        mesh_origins_with_locs.emplace_back(
            block_arg.getLoc(), GetSrcSet(func_op, block_arg.getArgNumber())
                                    .ToArray(func_op.getContext()));
        continue;
      }

      // Case 1b: BlockArgument of a mpmd.call.
      SmallVector<CallOp> call_ops =
          GetMaybeCachedCallOps(func_op, lazy_call_ops_by_callee);
      for (CallOp call_op : call_ops) {
        worklist.push_back(call_op.getOperand(block_arg.getArgNumber()));
      }
      continue;
    }

    // Case 2: The value is defined by an Operation
    auto op_result = cast<OpResult>(current_value);
    Operation* defining_op = op_result.getOwner();

    // Case 2a: UnassignOp
    if (auto unassign_op = dyn_cast<mpmd::UnassignOp>(defining_op)) {
      MeshesWithOrigins src_set;
      // Ensure that the first mesh and origin of the src_set is from the
      // unassign op.
      src_set.insert(unassign_op.getMeshWithOrigin());
      src_set.Union(GetSrcSet(unassign_op));
      mesh_origins_with_locs.emplace_back(
          unassign_op.getLoc(), src_set.ToArray(defining_op->getContext()));
      continue;
    }

    // Case 2b: Meshless op, so add to worklist.
    if (IsMeshlessOp(defining_op)) {
      for (Value operand : defining_op->getOperands()) {
        worklist.push_back(operand);
      }
      continue;
    }

    // Case 2c: CallOp, so add to worklist.
    if (auto call_op = dyn_cast<CallOp>(defining_op)) {
      worklist.push_back(
          GetCalleeFunc(call_op).front().getTerminator()->getOperand(
              op_result.getResultNumber()));
    }

    // TODO: b/421069658 - Handle for ops.
}

  return mesh_origins_with_locs;
}

// Walk through the MLIR module and get the origins of the use_set of `value`,
// with the MLIR location. All values in the use_set must arise from either
// entrypoint func return values or mpmd.assign ops.
LocToOrigins GetUseOriginsWithLocs(
    ValueRange root,
    DenseMap<StringRef, SmallVector<CallOp>>& lazy_call_ops_by_callee) {
  LocToOrigins mesh_origins_with_locs;

  std::vector<Value> worklist(root.begin(), root.end());
  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current_value = worklist.back();
    worklist.pop_back();
    if (visited.contains(current_value)) {
      continue;
    }
    visited.insert(current_value);

    for (OpOperand& use : current_value.getUses()) {
      // Case 1: AssignOp
      if (auto assign_op = dyn_cast<mpmd::AssignOp>(use.getOwner())) {
        mesh_origins_with_locs.emplace_back(
            assign_op.getLoc(),
            SmallVector<MeshWithOriginsAttr>{assign_op.getMeshWithOrigin()});
        continue;
      }

      // Case 2: The value is used by a meshless op.
      if (IsMeshlessOp(use.getOwner())) {
        for (Value res : use.getOwner()->getResults()) {
          worklist.push_back(res);
        }
        continue;
      }

      // Case 3: Returned by a function.
      if (auto return_op = dyn_cast<func::ReturnOp>(use.getOwner())) {
        Operation* parent_op = return_op->getParentOp();
        FuncOp func_op = cast<FuncOp>(parent_op);

        // Case 3a: The value is returned by the entrypoint function.
        if (IsEntryPointFunction(func_op)) {
          mesh_origins_with_locs.emplace_back(
              GetResultInfoLoc(func_op, use.getOperandNumber())
                  .value_or(return_op->getLoc()),
              SmallVector<MeshWithOriginsAttr>{});
          continue;
        }

        // Case 3b: The value is returned by a mpmd.call.
        for (CallOp call_op :
             GetMaybeCachedCallOps(func_op, lazy_call_ops_by_callee)) {
          worklist.push_back(call_op.getResult(use.getOperandNumber()));
        }
        continue;
      }

      // Case 4: Call Op
      if (auto call_op = dyn_cast<CallOp>(use.getOwner())) {
        worklist.push_back(
            GetCalleeFunc(call_op).getArgument(use.getOperandNumber()));
        continue;
      }

      // Case 5: For Op
      // TODO: b/421069658 - Handle for ops.
    }
  }
  return mesh_origins_with_locs;
}

// Prints mesh with origins in the format of
// `mesh[origin1,origin2], mesh2[origin3]`.
std::string PrintMeshWithOrigins(MeshWithOriginsAttr mesh_with_origins) {
  std::string result;
  llvm::raw_string_ostream str_stream(result);
  str_stream << mesh_with_origins.getMeshName();
  str_stream << "["
             << llvm::join(llvm::map_range(mesh_with_origins.getOrigins(),
                                           [](const OriginAttr& origin) {
                                             return origin.getOriginLabel();
                                           }),
                           ",")
             << "]";
  return result;
}

std::string PrintMeshesWithOrigins(
    ArrayRef<MeshWithOriginsAttr> meshes_with_origins) {
  return llvm::join(llvm::map_range(meshes_with_origins, PrintMeshWithOrigins),
                    ", ");
}

// Prints the loc to origins in the format of one of:
// - `loc1 - Input <loc1>: mesh1[origin1], mesh2[origin3]`.
// - `loc1 - Output <loc1>: mesh1[origin1], mesh2[origin3]`.
// - `loc1 - named_computation "origin1": mesh1[origin1], mesh2[origin3]`.
// - `loc1 - named_tensor "origin1": mesh1[origin1], mesh2[origin3]`.
std::string PrintLocToOrigins(const LocToOrigins& loc_to_origins,
                              StringRef loc_str_prefix = "",
                              bool is_input = true) {
  std::string result;
  llvm::raw_string_ostream str_stream(result);

  for (auto& [loc, meshes_with_origins] : loc_to_origins) {
    std::string loc_str = PrintLocation(loc);
    bool append_loc = true;
    str_stream << "\n";
    // TODO: b/396601755 - Avoid having to string check to detect named
    // computation and named tensor.
    if (llvm::StringRef(loc_str).contains("named_computation")) {
      str_stream << loc_str_prefix << " - named_computation \""
                 << meshes_with_origins[0].getOrigins()[0].getOriginLabel()
                 << "\": ";

    } else if (llvm::StringRef(loc_str).contains("named_tensor")) {
      str_stream << loc_str_prefix << " - named_tensor \""
                 << meshes_with_origins[0].getOrigins()[0].getOriginLabel()
                 << "\": ";
    } else {
      if (is_input) {
        str_stream << loc_str_prefix << " - Input " << loc_str << ": ";
      } else {
        str_stream << loc_str_prefix << " - Output " << loc_str << ": ";
      }
      append_loc = false;
    }

    str_stream << PrintMeshesWithOrigins(meshes_with_origins);

    if (append_loc) {
      str_stream << loc_str;
    }
  }
  return result;
}

class InferMeshValidateSrcSetNotEmptyPass
    : public impl::InferMeshValidateSrcSetNotEmptyPassBase<
          InferMeshValidateSrcSetNotEmptyPass> {
  using InferMeshValidateSrcSetNotEmptyPassBase::
      InferMeshValidateSrcSetNotEmptyPassBase;

  void runOnOperation() override {
    error_count_ = 0;
    const int max_errors = GetValidatedMaxErrors(errorLimit);

    for (FuncOp func : GetMpmdFunctions(getOperation())) {
      runOnFunc(func, max_errors);
    }
  }

  void runOnFunc(FuncOp func, const int max_errors) {
    // The walk is interrupted if we hit the limit number of errors we can emit.
    func.walk([&](Operation* op) {
      if (error_count_ == max_errors) {
        return WalkResult::interrupt();
      }
      if (auto fragment = dyn_cast<FragmentOp>(op)) {
        // Skip fragment ops and their regions for efficiency, as they are
        // already assigned.
        return WalkResult::skip();
      }
      if (IsMeshlessOp(op) && !ValidateMeshlessOpHasNonEmptySrcSet(op)) {
        error_count_++;
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });

    for (int64_t arg_num = 0;
         arg_num < func.getNumArguments() && error_count_ < max_errors;
         ++arg_num) {
      if (!ValidateMpmdCalleeArgHasNonEmptySrcSet(func, arg_num)) {
        error_count_++;
      }
    }

    // Catch the errors emitted if any.
    if (error_count_ > 0) {
      return signalPassFailure();
    }
  }

  // Returns false for the first error spotted in an error chain. Otherwise
  // returns true.
  //
  // If the op has an empty src_set, then we emit an error and return false.
  // Also if the op should be a cross-mesh reduction but it is not, since in
  // that case it would also have an empty src_set if we didn't infer the
  // cross-mesh reduction.
  bool ValidateMeshlessOpHasNonEmptySrcSet(Operation* op) {
    std::optional<SetVector<StringRef>> src_set = GetSrcMeshes(op);
    if (!src_set.has_value()) {
      // src_set is not present means that the op can be assigned to any mesh,
      // so any assignment is possible.
      return true;
    }

    if (!src_set->empty() && !IsMpmdReduceAnnotated(op)) {
      return true;
    }

    SDY_CHECK(!(IsMpmdReduceAnnotated(op) && src_set->empty()))
        << "It should not be possible for an mpmd-reduce-annotated op to have "
           "an empty src_set.";

    if ((src_set->empty() && !IsFirstOpWithEmptySrcSet(op)) ||
        (IsMpmdReduceAnnotated(op) && !IsFirstOpInReductionChain(op))) {
      return true;
    }
    op->emitError(MeshlessOpEmptySrcSetError(op, IsMpmdReduceAnnotated(op)));
    return false;
  }

  bool IsFirstOpWithEmptySrcSet(Operation* op) {
    for (OpOperand& operand : op->getOpOperands()) {
      std::optional<SetVector<StringRef>> operand_src_set =
          GetSrcMeshes(operand);
      if (operand_src_set && operand_src_set->empty()) {
        return false;
      }
    }
    return true;
  }

  bool IsFirstOpInReductionChain(Operation* op) {
    for (Value operand : op->getOperands()) {
      if (Operation* defining_op = operand.getDefiningOp();
          defining_op && IsMpmdReduceAnnotated(defining_op)) {
        return false;
      }
    }
    return true;
  }

  bool ValidateMpmdCalleeArgHasNonEmptySrcSet(FuncOp callee, int arg_num) {
    std::optional<SetVector<StringRef>> src_set = GetSrcMeshes(callee, arg_num);
    if (!src_set || !src_set->empty()) {
      return true;
    }

    // Don't emit an error if the empty src_set is propagated from elsewhere.
    if (!IsCalleeFirstOpWithEmptySrcSet(callee, arg_num)) {
      return true;
    }

    emitError(callee->getLoc(), CalleeEmptySrcSetError(callee, arg_num));
    return false;
  }

  bool IsCalleeFirstOpWithEmptySrcSet(FuncOp callee, int arg_num) {
    for (CallOp call_op :
         GetMaybeCachedCallOps(callee, lazy_call_ops_by_callee_)) {
      std::optional<SetVector<StringRef>> operand_src_set =
          GetSrcMeshes(call_op->getOpOperand(arg_num));
      if (operand_src_set && operand_src_set->empty()) {
        return false;
      }
    }

    return true;
  }

  std::string MeshlessOpEmptySrcSetError(Operation* op,
                                         bool is_cross_mesh_reduction_error) {
    std::string error_str;
    llvm::raw_string_ostream error_stream(error_str);
    // Log errors for the op.
    error_stream
        << "Mesh assignment is not possible for op as its operands are on "
           "conflicting meshes and thus we need to transfer some of the "
           "operands. Add an explicit transfer to fix this.\n";

    if (is_cross_mesh_reduction_error) {
      error_stream << "To handle this automatically, set "
                      "`mpmd_infer_cross_mesh_reductions` in the "
                      "partitioning options.";
    } else {
      error_stream << "To handle this automatically, set "
                      "`mpmd_infer_transfers` in the partitioning options.";
    }
    error_stream << kErrorMessageSeparator;
    error_stream << "Op: \n\t" << PrintOperationForLog(op);

    // Log the meshes and details of the op operands.
    error_stream << "\n\nOperands with possibly conflicting meshes:\n";
    for (OpOperand& operand : op->getOpOperands()) {
      std::optional<SetVector<StringRef>> operand_src_set =
          GetSrcMeshes(operand);
      LocToOrigins loc_to_origins =
          GetSrcOriginsWithLocs(operand.get(), lazy_call_ops_by_callee_);

      error_stream << "\n - Operand " << operand.getOperandNumber();
      if (operand_src_set) {
        error_stream << " can be assigned to meshes {"
                     << llvm::join(operand_src_set->getArrayRef(), ",")
                     << "} which originates from:";
      } else {
        error_stream << " can be assigned to any mesh and originates from the "
                        "intersections of:";
      }
      error_stream << PrintLocToOrigins(loc_to_origins,
                                        /*loc_str_prefix=*/"  ");
    }

    auto use_meshes = GetUseMeshes(op);
    if (!use_meshes.empty()) {
      error_stream << "\n\nOp is used in meshes {"
                   << llvm::join(use_meshes, ",")
                   << "} which originates from the union of:";
    } else {
      error_stream << "\n\nOp is used in:";
    }
    error_stream << PrintLocToOrigins(
        GetUseOriginsWithLocs(op->getResults(), lazy_call_ops_by_callee_),
        /*loc_str_prefix=*/"", /*is_input=*/false);

    error_stream << "\n\nOp stack trace:\n"
                 << PrintStackTraceFromLoc(op->getLoc()) << "\n";
    error_stream << kErrorMessageSeparator;
    return error_str;
  }

  std::string CalleeEmptySrcSetError(FuncOp callee, int arg_num) {
    std::string error_str;
    llvm::raw_string_ostream error_stream(error_str);
    // Log errors for the callee. Don't print the callee itself, since it is
    // usually very verbose, so just print the name.
    error_stream
        << "Mesh assignment is not possible for arg" << arg_num
        << " of mpmd.call \"" << SanitizeCallName(callee.getSymName()).str()
        << "\" as its caller operands are on "
           "conflicting meshes and thus we need to transfer some of the "
           "operands. Add an explicit transfer to fix this.\n";

    error_stream << "To handle this automatically, set "
                    "`mpmd_infer_transfers` in the partitioning options.";

    error_stream << kErrorMessageSeparator;

    BlockArgument arg = callee.getArgument(arg_num);
    error_stream
        << "\nmpmd.call was called "
        << GetMaybeCachedCallOps(callee, lazy_call_ops_by_callee_).size()
        << " times and can be assigned to meshes: {"
        << llvm::join(GetSrcSet(callee, arg_num).MeshNamesOrEmpty(), ",")
        << "} which originates from the intersection of:";
    error_stream << PrintLocToOrigins(
        GetSrcOriginsWithLocs(arg, lazy_call_ops_by_callee_));

    auto use_meshes = GetArgUseSet(callee, arg_num).MeshNamesOrEmpty();
    if (!use_meshes.empty()) {
      error_stream << "\n\nArg is used in meshes {"
                   << llvm::join(use_meshes, ",")
                   << "} which originates from the union of:";
    } else {
      error_stream << "\n\nArg is used in:";
    }
    error_stream << PrintLocToOrigins(
        GetUseOriginsWithLocs(arg, lazy_call_ops_by_callee_));

    error_stream << "\n\nmpmd.call Stack trace:\n"
                 << PrintStackTraceFromLoc(callee.getLoc()) << "\n";

    error_stream << "\n\nSample arg users:";
    for (Operation* user : arg.getUsers()) {
      error_stream << "\n - " << PrintOperationForLog(user)
                   << PrintStackTraceFromLoc(user->getLoc());
    }

    error_stream << kErrorMessageSeparator;
    return error_str;
  }

  DenseMap<StringRef, SmallVector<CallOp>> lazy_call_ops_by_callee_;
  int error_count_ = 0;
};

// Attribute used to mark an op as visited by the validation pass, when
// validation fails. This is used to avoid emitting the same error multiple
// times.
inline constexpr StringRef kVisitedFailure = "mpmd.visited_failure";

void SetVisitedFailureAttr(Operation* op) {
  op->setAttr(kVisitedFailure, UnitAttr::get(op->getContext()));
}
void SetVisitedFailureAttr(FuncOp op, int arg_num) {
  op.setArgAttr(arg_num, kVisitedFailure, UnitAttr::get(op.getContext()));
}
bool IsFailureAndMeshlessOpVisited(Operation* op) {
  if (IsMeshlessOp(op)) {
    return op->hasAttr(kVisitedFailure);
  }
  return false;
}
bool IsFailureAndCalleeArgVisited(FuncOp op, int arg_num) {
  return op.getArgAttr(arg_num, kVisitedFailure) != nullptr;
}
bool IsFailureAndOpVisited(OpOperand& use) {
  if (auto call_op = dyn_cast<CallOp>(use.getOwner())) {
    FuncOp callee_func = GetCalleeFunc(call_op);
    return IsFailureAndCalleeArgVisited(callee_func, use.getOperandNumber());
  }
  return IsFailureAndMeshlessOpVisited(use.getOwner());
}

class InferMeshValidateNoAdditionalTransfersNeededPass
    : public impl::InferMeshValidateNoAdditionalTransfersNeededPassBase<
          InferMeshValidateNoAdditionalTransfersNeededPass> {
  using InferMeshValidateNoAdditionalTransfersNeededPassBase::
      InferMeshValidateNoAdditionalTransfersNeededPassBase;

  void runOnOperation() override {
    error_count_ = 0;
    emitted_error_count_ = 0;
    const int max_errors = GetValidatedMaxErrors(errorLimit);

    SmallVector<FuncOp> mpmd_funcs = GetMpmdFunctions(getOperation());

    // Run on non-entry point functions first, so that the call-ops are
    // validated before the entry point functions.
    for (FuncOp func : mpmd_funcs) {
      if (!IsEntryPointFunction(func)) {
        runOnFunc(func, max_errors);
      }
    }

    for (FuncOp func : mpmd_funcs) {
      if (IsEntryPointFunction(func)) {
        runOnFunc(func, max_errors);
      }
    }
  }

  void runOnFunc(FuncOp func, const int max_errors) {
    // The walk is interrupted if we hit the limit number of errors we can emit.
    func.walk<WalkOrder::PreOrder, ReverseIterator>([&](Operation* op) {
      if (error_count_ == max_errors) {
        return WalkResult::interrupt();
      }
      if (auto fragment = dyn_cast<FragmentOp>(op)) {
        // Skip fragment ops and their regions for efficiency, as they are
        // already assigned.
        return WalkResult::skip();
      }
      if (IsMeshlessOp(op) || isa<AssignOp>(op)) {
        ValidationResult result = ValidateMeshlessOpDoesNotNeedTransfer(op);
        if (ValidationResult::kOk != result) {
          error_count_++;
        }
        if (ValidationResult::kError == result) {
          emitted_error_count_++;
        }
      }
      if (auto call_op = dyn_cast<CallOp>(op);
          call_op && IsCallOpInCallChain(call_op)) {
        ValidateCallOpChainUsesMatch(call_op, error_count_,
                                     emitted_error_count_);
      }
      return WalkResult::advance();
    });
    for (int64_t arg_num = 0;
         arg_num < func.getNumArguments() && emitted_error_count_ < max_errors;
         ++arg_num) {
      ValidationResult result =
          ValidateCalleeArgDoesNotNeedTransfer(func, arg_num);
      if (ValidationResult::kOk != result) {
        error_count_++;
      }
      if (ValidationResult::kError == result) {
        emitted_error_count_++;
      }
    }

    // Catch the errors emitted if any.
    if (error_count_ > 0) {
      if (emitted_error_count_ == 0) {
        SDY_LOG(ERROR)
            << "No errors emitted, but error count is " << error_count_
            << ". This means we've failed to emit an error and is a bug "
               "in the error handling.";
      }
      return signalPassFailure();
    }
  }

  ValidationResult ValidateCallOpChainUsesMatch(CallOp call_op,
                                                int& error_count,
                                                int& emitted_error_count) {
    auto callee_func = GetCalleeFunc(call_op);
    for (auto result : call_op->getOpResults()) {
      for (OpOperand& use : result.getUses()) {
        if (auto user_call_op = dyn_cast<CallOp>(use.getOwner());
            !user_call_op || user_call_op.getCallee() != call_op.getCallee()) {
          continue;
        }

        MeshesWithOrigins res_use_set =
            GetResUseSet(callee_func, result.getResultNumber());
        MeshesWithOrigins arg_use_set =
            GetArgUseSet(callee_func, use.getOperandNumber());

        if (!res_use_set.HasSameMeshes(arg_use_set)) {
          emitError(callee_func->getLoc(),
                    CallOpChainMismatchError(
                        callee_func, use.getOperandNumber(),
                        result.getResultNumber(), arg_use_set, res_use_set));
          error_count++;
          emitted_error_count++;
        }
      }
    }
    return ValidationResult::kOk;
  }

  std::string CallOpChainMismatchError(FuncOp callee, int arg_num, int res_num,
                                       const MeshesWithOrigins& arg_use_set,
                                       const MeshesWithOrigins& res_use_set) {
    std::string error_str;
    llvm::raw_string_ostream error_stream(error_str);

    error_stream
        << "Mesh assignment is not possible for mpmd.call \""
        << SanitizeCallName(callee.getSymName()).str()
        << "\" as it passes result " << res_num << " to arg " << arg_num
        << " but they have mismatching mesh assignments: res: {"
        << PrintMeshesWithOrigins(res_use_set.ToArray(callee.getContext()))
        << "}, arg: {"
        << PrintMeshesWithOrigins(arg_use_set.ToArray(callee.getContext()))
        << "}. Please reach out if you see this.";

    error_stream << kErrorMessageSeparator;

    error_stream << "\n\nmpmd.call Stack trace:\n"
                 << PrintStackTraceFromLoc(callee.getLoc()) << "\n";

    return error_str;
  }

  // Validates that the necessary mesh assignments are possible
  // without introducing any transfer ops, i.e. that the use_set is contained in
  // the src_set.
  //
  // This must be used with a reverse iterator walk, as we want to visit the
  // users before the op.
  // To avoid emitting the same error multiple times, we mark the op as having
  // failed validation if we emit an error.
  // TODO(b/396601755): Return one error for all errors in an error chain.
  ValidationResult ValidateMeshlessOpDoesNotNeedTransfer(Operation* op) {
    std::optional<SetVector<StringRef>> src_set = GetSrcMeshes(op);
    SetVector<StringRef> use_set = GetUseMeshes(op);

    if (!src_set.has_value()) {
      // src_set is not present means that the op can be assigned to any mesh,
      // so any assignment is possible.
      return ValidationResult::kOk;
    }

    // This should be caught by ValidateSrcSetNotEmpty.
    SDY_CHECK(!src_set->empty())
        << "This should have been caught by an earlier validation check. Reach "
           "out if you see this.";

    if (llvm::set_is_subset(use_set, *src_set)) {
      return ValidationResult::kOk;
    }

    // Only emit error if none of the users have already been marked as having
    // failed validation.
    if (llvm::any_of(op->getUses(), IsFailureAndOpVisited)) {
      SetVisitedFailureAttr(op);
      return ValidationResult::kErrorButDontEmit;
    }

    op->emitError(
        MeshlessOpError(op, use_set.getArrayRef(), src_set->getArrayRef()));
    SetVisitedFailureAttr(op);
    return ValidationResult::kError;
  }

  std::string MeshlessOpError(Operation* op, ArrayRef<StringRef> use_set,
                              ArrayRef<StringRef> src_set) {
    std::string error_str;
    llvm::raw_string_ostream error_stream(error_str);

    // Log errors for the op.
    error_stream
        << "Mesh assignment is not possible for op as it is used in {"
        << llvm::join(use_set, ",") << "} but it can only be placed on {"
        << llvm::join(src_set, ",")
        << "} without introducing a transfer. Add an explicit transfer to fix "
           "this.\n";

    error_stream << "To handle this automatically, set "
                    "`mpmd_infer_transfers` in the partitioning options.";
    error_stream << kErrorMessageSeparator;
    error_stream << "Op: \n\t" << PrintOperationForLog(op);

    error_stream << "\n\nCan be assigned to meshes: {"
                 << llvm::join(src_set, ",")
                 << "} which originates from the intersection of:";
    error_stream << PrintLocToOrigins(
        GetSrcOriginsWithLocs(op->getOperands(), lazy_call_ops_by_callee_));

    error_stream << "\n\nBut was used in meshes: {" << llvm::join(use_set, ",")
                 << "} which originates from the union of:";
    error_stream << PrintLocToOrigins(
        GetUseOriginsWithLocs(op->getOperands(), lazy_call_ops_by_callee_),
        /*loc_str_prefix=*/"", /*is_input=*/false);

    error_stream << "\n\nOp stack trace:\n"
                 << PrintStackTraceFromLoc(op->getLoc()) << "\n";

    // Log the meshes and details of the op operands.
    error_stream << kErrorMessageSeparator;
    return error_str;
  }

  // Validates that the necessary mesh assignments are possible on CallOps
  // without introducing any transfer ops, i.e. that the use_set is contained in
  // the src_set.
  //
  // Returns `true` if validation succeeded (i.e., no error was emitted);
  // `false` otherwise, in which case an error was emitted.
  // TODO(b/396601755): Return one error for all errors in an error chain.
  ValidationResult ValidateCalleeArgDoesNotNeedTransfer(FuncOp func,
                                                        int arg_num) {
    std::optional<SetVector<StringRef>> src_set = GetSrcMeshes(func, arg_num);
    SetVector<StringRef> use_set =
        GetArgUseSet(func, arg_num).MeshNamesOrEmpty();

    if (!src_set.has_value()) {
      // src_set is not present means that the op can be assigned to any mesh,
      // so any assignment is possible.
      return ValidationResult::kOk;
    }

    // This should be caught by ValidateSrcSetNotEmpty.
    SDY_CHECK(!src_set->empty())
        << "This should have been caught by an earlier validation check. Reach "
           "out if you see this.";

    if (llvm::set_is_subset(use_set, *src_set)) {
      return ValidationResult::kOk;
    }

    SetVisitedFailureAttr(func, arg_num);

    // Only emit error if none of the users have already been marked as having
    // failed validation.
    if (llvm::any_of(func.getArgument(arg_num).getUses(),
                     IsFailureAndOpVisited)) {
      return ValidationResult::kErrorButDontEmit;
    }

    emitError(func->getLoc(),
              CalleeArgError(func, arg_num, use_set.getArrayRef(),
                             src_set->getArrayRef()));

    return ValidationResult::kError;
  }

  std::string CalleeArgError(FuncOp callee, int arg_num,
                             ArrayRef<StringRef> use_set,
                             ArrayRef<StringRef> src_set) {
    std::string error_str;
    llvm::raw_string_ostream error_stream(error_str);

    // Log errors for the op.
    error_stream << "Mesh assignment is not possible for arg" << arg_num
                 << " of mpmd.call \""
                 << SanitizeCallName(callee.getSymName()).str()
                 << "\" as it is used in {" << llvm::join(use_set, ",")
                 << "} but it can only be placed on {"
                 << llvm::join(src_set, ",")
                 << "} without introducing a transfer. Add an explicit "
                    "transfer to fix this.\n";

    error_stream << "To handle this automatically, set "
                    "`mpmd_infer_transfers` in the partitioning options.";

    error_stream << kErrorMessageSeparator;

    BlockArgument arg = callee.getArgument(arg_num);
    error_stream
        << "\nmpmd.call was called "
        << GetMaybeCachedCallOps(callee, lazy_call_ops_by_callee_).size()
        << " times and can be assigned to meshes: {" << llvm::join(src_set, ",")
        << "} which originates from the intersection of:";
    error_stream << PrintLocToOrigins(
        GetSrcOriginsWithLocs(arg, lazy_call_ops_by_callee_));

    error_stream << "\n\nArg is used in meshes: {" << llvm::join(use_set, ",")
                 << "} which originates from the union of:";
    error_stream << PrintLocToOrigins(
        GetUseOriginsWithLocs(arg, lazy_call_ops_by_callee_));

    error_stream << "\n\nmpmd.call Stack trace:\n"
                 << PrintStackTraceFromLoc(callee.getLoc()) << "\n";

    error_stream << "\n\nSample arg users:";
    for (Operation* user : arg.getUsers()) {
      error_stream << "\n - " << PrintOperationForLog(user)
                   << PrintStackTraceFromLoc(user->getLoc());
    }

    error_stream << kErrorMessageSeparator;

    return error_str;
  }

  DenseMap<StringRef, SmallVector<CallOp>> lazy_call_ops_by_callee_;
  int error_count_ = 0;
  int emitted_error_count_ = 0;
};

}  // namespace

}  // namespace mlir::mpmd
