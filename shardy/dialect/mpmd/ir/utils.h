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

#ifndef SHARDY_DIALECT_MPMD_IR_UTILS_H_
#define SHARDY_DIALECT_MPMD_IR_UTILS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir::mpmd {

// Globsl sdy mesh name.
constexpr StringRef kGlobalMeshName = "mesh";

// The function attribute that holds the SPMD mesh.
constexpr StringRef kMeshShapeAttr = "mesh_shape";
// The function attribute that holds the MPMD topology.
constexpr StringRef kTopologyAttr = "topology";

// TODO(b/428336749): Remove these attributes once gspmd path is gone.

// When set in the module, it means that the module fragments have been SDY
// sharded by Shardy.
inline constexpr StringRef kIsSdyPartitioned = "mpmd.is_sdy_partitioned";

// When set in the module, it means that the module fragments have been SPMD
// sharded by GSPMD.
// NOTE: once set, this attribute needs to be preserved throughout our lowering
// pipelines, so that it survives lowering to IFRT-IR and can be used to build
// compiling options.
inline constexpr StringRef kIsGspmdPartitioned = "mpmd.is_gspmd_partitioned";

// The suffix of the mesh name for a CPU mesh.
// LINT.IfChange
constexpr StringRef kCpuMeshSuffix = "/cpu";
// LINT.ThenChange(
//   https://github.com/openxla/shardy/blob/main/shardy/integrations/python/jax/mpmd/types.py
// )

// Memory kind attributes.
// Attr on func args and results to indicate whether the value lives on host or
// device. If not present, it means it lives on device.
inline constexpr StringRef kMemoryKindAttr = "mhlo.memory_kind";
// Attr value to indicate whether the value is pinned on the host.
inline constexpr StringRef kMemoryKindPinnedHost = "pinned_host";
// Attr value to indicate whether the value is unpinned on the host.
inline constexpr StringRef kMemoryKindUnpinnedHost = "unpinned_host";
// Attr value to indicate whether the value is on the device.
inline constexpr StringRef kMemoryKindDevice = "device";

// Layout-related attributes.
inline constexpr StringRef kLayoutModeAttr = "mhlo.layout_mode";
// Attr value to use the default compact layout.
inline constexpr StringRef kLayoutModeDefault = "default";
// Attr value to let compiler choose layout.
inline constexpr StringRef kLayoutModeAuto = "auto";

// A module attribute to indicate the module has been lowered with JAX using sdy
// config. The dialect name prefix is needed.
inline constexpr StringRef kIsSdyLowered = "mpmd.sdy_lowered";

inline constexpr StringRef kRematAttributeName = "remat";

inline constexpr StringRef kJaxResultInfoAttr = "jax.result_info";

template <typename... Args>
std::string StrCat(Args&&... args) {
  std::string result;
  llvm::raw_string_ostream os(result);

  // C++17 fold expression
  (os << ... << std::forward<Args>(args));

  return result;
}

// Specifies for a tensor how each of its dimensions are sharded. This is
// equivalent to "PartitionSpec" in JAX.
//
// For example, given
//   sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>
//   sdy.sharding = <@mesh, [{}, {"x", "z"}]>
// the corresponding `SpmdTensorShardingSpec` would be
//   [[], ["x", "z"]]
using SpmdTensorPartitionSpec = std::vector<std::vector<std::string>>;

// Holds the mesh name, sharding specs, and memory kind of a mesh tensor.
struct NamedSpmdShardingSpec {
  std::string mesh_name;
  SpmdTensorPartitionSpec tensor_spec;
  std::optional<std::string> memory_kind;
};

// Holds the mesh names and sharding specs of each input and output.
struct FunctionIOShardingSpecsAndMeshes {
  std::vector<NamedSpmdShardingSpec> input_specs;
  std::vector<NamedSpmdShardingSpec> output_specs;
};

// Given a function, extracts the mesh names, sharding specs and the memory kind
// of each input and output. We assume that the function is a valid MPMD
// program, and therefore we are guaranteed to get the mesh names and sharding
// specs.
//
// If an input or output of the function has both a type with memory kind and an
// attribute with memory kind, we use the memory kind from the type, even if
// they are different. If no there's no memory_kind on value or attribute then
// the memory kind is not set.
//
// TODO: b/374994155 - Instead we should drop the memory kind from the
// attributes and move it to types only. If for some reason that's not possible
// we should at least be consistent and use the memory kind from the type if
// then we need at least some verification of their consistency.
FunctionIOShardingSpecsAndMeshes ExtractFunctionIOShardingSpecsAndMeshes(
    func::FuncOp func_op);

// Retrieves the function named "main" from the given module, if it exists, and
// fails otherwise.
func::FuncOp GetMainFunction(ModuleOp module);

// Returns true iff the module has a function which is annotated with a
// topology.
bool IsMpmdModule(ModuleOp module);

// Returns true iff the function is named "main".
bool IsMainFunction(func::FuncOp func_op);

// Returns true if the function is annotated with a topology.
bool IsMpmdFunction(func::FuncOp func_op);

// Returns true if the function is annotated with a mesh.
bool IsSpmdFunction(func::FuncOp func_op);

// Returns true if the function is either a SPMD or an MPMD function (see
// above).
bool IsDistributedFunction(func::FuncOp func_op);

// A function is considered an entry point function if it is distributed and
// public. Note: this means that other functions which are distributed but
// not intended to be entry points must be set to private (e.g. MPMD CallOp
// callees).
bool IsEntryPointFunction(func::FuncOp func_op);

// Returns the topology attribute of a function.
// Precondition: `IsMpmdFunction(func_op)` must be true.
TopologyAttr GetTopology(func::FuncOp func_op);

// Returns the topology attribute of a module.
// Precondition: `IsMpmdModule(module_op)` must be true.
TopologyAttr GetTopology(ModuleOp module_op);

// Removes the mesh attribute from the function.
void RemoveMesh(func::FuncOp func_op);

// A flat mesh is a vector of (axis_name, axis_size) pairs.
using FlatMesh = std::vector<std::pair<std::string, int>>;

// Sets the topology used by the function enclosing op (which can be a function
// itself).
void SetTopology(
    const std::vector<std::pair<std::string, FlatMesh>>& topology_shape,
    Operation* op);

// Returns the meshes of the topology attribute of an mpmd function.
// Precondition: `IsMpmdFunction()` must be true.
ArrayRef<NamedMeshAttr> GetTopologyMeshes(func::FuncOp func_op);

// Converts a list of NamedMeshAttr to a name-to-mesh-attr map.
llvm::DenseMap<StringRef, sdy::MeshAttr> GetMeshesByName(
    ArrayRef<NamedMeshAttr> meshes);

// Casts the type of `value` into a MeshTensorType and returns its local type.
Type GetLocalTensorTypeFromMeshType(Value value, sdy::MeshAttr mesh_attr);

// Casts the type of `value` into a MeshTensorType and returns its global type.
Type GetGlobalTensorTypeFromMeshType(Value value, sdy::MeshAttr mesh_attr);

// If `op` is an intra-mesh TransferOp returns it, otherwise returns nullptr.
TransferOp DynCastIntraMeshTransfer(Operation* op);

// If `op` is an inter-mesh TransferOp returns it, otherwise returns nullptr.
TransferOp DynCastInterMeshTransfer(Operation* op);

// Return true if `op` is an inter-mesh TransferOp.
bool IsInterMeshTransfer(Operation* op);

// Returns a vector with all functions in `module_op` that have a topology.
SmallVector<func::FuncOp> GetMpmdFunctions(ModuleOp module_op);

inline func::FuncOp GetCalleeFunc(CallOp call_op) {
  return cast<func::FuncOp>(cast<CallOpInterface>(*call_op).resolveCallable());
}

// A dataflow edge with multiple sources and multiple targets used when
// propagating info (e.g., mesh assignment, SPMD sharding propagation, etc)
// through control-flow like ops (e.g., `mpmd.call` and `mpmd.for` ops).
//
// We allow for multiple sources in an edge, whenever we want to guarantee some
// form of consistency among the different sources (even if they are different
// values). For example, the first operand of all call_ops to the same function
// must be assigned to meshes consistently. Similarly for multiple targets.
struct MpmdDataflowEdge {
  SmallVector<OpOperand*> sources;
  SmallVector<Value> targets;
};

// Returns a vector of dataflow edges that connect block arguments of the
// function with the operands of its call_ops, or an empty list if the function
// isn't referenced by a mpmd.call_op.
SmallVector<MpmdDataflowEdge> GetMpmdDataflowEdgesForFuncArgs(
    func::FuncOp func_op);

// Returns a vector of dataflow edges that connect operands of the function's
// return op with the results of its call_ops, or an empty list if the function
// isn't referenced by a mpmd.call_op.
SmallVector<MpmdDataflowEdge> GetMpmdDataflowEdgesForFuncResults(
    func::FuncOp func_op);

// Returns all the call ops that with `func_op` as a target, ignoring unused
// ones.
SmallVector<CallOp> GetCallOps(func::FuncOp func_op);

// Returns a vector of dataflow edges that connect block arguments of the
// function with the operands of its call_ops and operands of the function's
// return op with the results of its call_ops.
//
// Returns an empty list if the function isn't referenced by a
// mpmd.call_op.
//
// See GetMpmdDataflowEdge.FuncOp test in utils_test for an example.
//
// Note: we collect edges on func_ops, instead of call_ops, because we need to
// guarantees assignment/sharding consistency across different call_ops (e.g.,
// the i-th operand of all call_ops to function `f` must be assigned to the
// same mesh).
SmallVector<MpmdDataflowEdge> GetMpmdDataflowEdges(func::FuncOp func_op);

// Replaces the given `op` with a new FragmentOp that is assigned to `mesh_name`
// and has a clone of `op` in its region whose results are returned.
//
// The created FragmentOp will take as operands all operands of `op`, and any
// free tensor variables' in its regions, after assigning them to the same mesh,
// and the results of the FragmentOp will be unassigned from the mesh so they
// can replace the uses of `op`.
//
// This method only replaces uses of the original op for which
// `should_replace_use` returns true.
FragmentOp WrapOpWithFragment(
    Operation* op, StringRef mesh_name, RewriterBase& rewriter,
    std::function<bool(OpOperand&)> should_replace_use = [](OpOperand&) {
      return true;
    });

// Returns all the transpose counts of a fragment.
std::vector<int64_t> GetTransposeCounts(FragmentOp fragment);

// If the fragment has one and only one transpose count, returns that value.
// Otherwise, returns `nullopt`.
std::optional<int64_t> TryToFindSingleTransposeCount(FragmentOp fragment);

// If the fragment has at least one transpose count, returns the maximum value.
// Otherwise, returns `nullopt`.
//
// A merged fragment may have multiple transpose counts. We take the max
// transpose count to ensure that we are reflecting the highest order of
// differentiation the fragment is involved in, e.g., when merging a
// rematerialized forward fragment (transpose_count=0) with a backward fragment
// (transpose_count=1).
std::optional<int64_t> TryToFindMaxTransposeCount(FragmentOp fragment);

// If the fragment is a user-defined fragment, returns its single transpose
// count. If it is a merged remat fragment, returns the max transpose count.
// Otherwise, returns `nullopt`.
std::optional<int64_t> TryToFindFragmentTransposeCount(FragmentOp fragment);

constexpr StringRef kCallCounterAttrName = "call_counter";

// Returns the call counter of a fragment if defined. Otherwise, returns
// `nullopt`.
std::optional<uint32_t> TryToFindCallCounter(FragmentOp fragment);

// Checks if `fragment2` appears immediately after `fragment1` in the program
// respective to their mesh.
// Assumes that both fragments are assigned to the same mesh.
bool IsExecutedImmediatelyAfter(FragmentOp fragment1, FragmentOp fragment2);

// Checks if all the meshes in the topology of `func` are identical to each
// other.
bool HasHomogeneousTopology(func::FuncOp func);

// Returns the union of the origins of two different fragments.
ArrayAttr GetFragmentOriginUnion(FragmentOp fragment1, FragmentOp fragment2,
                                 RewriterBase& rewriter);

// Returns whether the module has been lowered with sdy config in JAX.
bool IsLoweredWithSdy(ModuleOp module);

// Checks if an operation is the result of rematerialization (e.g.,
// created during during `mpmd-loop-remat`).
bool IsRemat(Operation* op);

// Marks an operation as being the result of rematerialization.
void MarkAsRemat(Operation* op, RewriterBase& rewriter);

// Extracts the memory kind from the mesh name if it exists (e.g., "mesh#device"
// or "mesh#pinned_host"), returning a pair with the mesh name and the memory
// kind, or a pair with the mesh name and nullopt if the latter is not present
// (e.g., "mesh").
// Note: we do not check whether the memory kind is valid.
std::pair<StringRef, std::optional<StringRef>>
TryToExtractMemoryKindFromMeshName(StringRef mesh_name);

// Updates the type of the value given a sharding, overriding the existing
// sharding if present.
void UpdateValueTypeWithSharding(Value value, sdy::TensorShardingAttr sharding);

// Returns the location of the result info attribute of the given result, if
// present, otherwise returns `nullopt`.
std::optional<Location> GetResultInfoLoc(func::FuncOp func,
                                         int64_t result_index);

// Lookup the mesh attribute in a function that contains the operation.
//
// Returns an error if `op` isn't a fragment or the enclosing function doesn't
// have a topology attribute.
FailureOr<sdy::MeshAttr> GetMeshAttr(Operation* op);

// Lookup the mesh attribute by its name in the topology in a function that
// contains the operation.
//
// Returns an error if the enclosing function doesn't have a topology attribute.
FailureOr<sdy::MeshAttr> GetMeshAttr(Operation* op, StringRef mesh_name);

// Same as `GetMeshAttr(op)` but hards fail if an error is returned.
sdy::MeshAttr GetMeshOrFail(Operation* op);

// Same as `GetMeshAttr(op, mesh_name)` but hards fail if an error is returned.
sdy::MeshAttr GetMeshOrFail(Operation* op, StringRef mesh_name);

// Finds an operation inside `module` that carries an attribute named
// `annotation`. Returns `nullptr` if such operation does not exist.
Operation* FindAnnotatedOperation(ModuleOp module, StringRef annotation);

// Checks whether given op is a supported binary reduction op and if so, returns
// the corresponding type.
std::optional<ReductionType> GetReductionOpType(Operation* op);

// Creates an stablehlo binary reduction op, given a binary reduction type.
Operation* CreateStablehloReduceOp(ReductionType reduction_type,
                                   ValueRange values, Location loc,
                                   OpBuilder& builder);

// Checks that a closure that is the argument of a reduction computation, such
// as the one encapsulated in stablehlo::ReduceOp or stablehlo:ScatterOp, is of
// a specific op type (e.g. stablehlo::AddOp or stablehlo::MaxOp) per result,
// and returns the corresponding ReductionType.
std::optional<ReductionType> ComputeReductionType(Block& block);

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_IR_UTILS_H_
