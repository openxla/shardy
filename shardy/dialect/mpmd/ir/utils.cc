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

#include "shardy/dialect/mpmd/ir/utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::mpmd {

namespace {

using ::mlir::func::FuncOp;

inline constexpr StringRef kJaxResultInfoAttr = "jax.result_info";

SpmdTensorShardingSpec ExtractTensorShardingSpec(MeshTensorType type,
                                                 sdy::MeshAttr mesh_attr) {
  if (!type.getSharding()) {
    return {};
  }
  SpmdTensorShardingSpec spec;
  spec.reserve(type.getSharding().getDimShardings().size());
  llvm::SmallDenseMap<StringRef, int> axis_name_to_index;
  for (auto [index, axis] : llvm::enumerate(mesh_attr.getAxes())) {
    axis_name_to_index[axis.getName()] = index;
  }

  for (sdy::DimensionShardingAttr dim_sharding :
       type.getSharding().getDimShardings()) {
    std::vector<int> indices;
    indices.reserve(dim_sharding.getAxes().size());
    // SDY sharding axes are in major-to-minor, but `SpmdTensorShardingSpec` is
    // expected to be minor-to-major. So we reverse the order.
    for (sdy::AxisRefAttr axis : llvm::reverse(dim_sharding.getAxes())) {
      indices.push_back(axis_name_to_index[axis.getName()]);
    }
    spec.emplace_back(indices);
  }
  return spec;
}

NamedSpmdShardingSpec GetNamedShardingSpec(MeshTensorType mesh_tensor,
                                           sdy::MeshAttr mesh_attr) {
  SpmdTensorShardingSpec spec =
      ExtractTensorShardingSpec(mesh_tensor, mesh_attr);
  std::optional<std::string> memory_kind;
  if (mesh_tensor.getMemoryKind()) {
    memory_kind = mesh_tensor.getMemoryKind().getValue().str();
  }
  return NamedSpmdShardingSpec{mesh_tensor.getMeshName().str(), spec,
                               memory_kind};
}

}  // namespace

FunctionIOShardingSpecsAndMeshes ExtractFunctionIOShardingSpecsAndMeshes(
    FuncOp func_op) {
  FunctionIOShardingSpecsAndMeshes io_sharding_and_mesh;
  io_sharding_and_mesh.input_specs.reserve(func_op.getNumArguments());
  for (BlockArgument arg : func_op.getArguments()) {
    MeshTensorType arg_type = cast<MeshTensorType>(arg.getType());
    sdy::MeshAttr mesh_attr = GetMeshOrFail(func_op, arg_type.getMeshName());
    NamedSpmdShardingSpec mesh_and_spec =
        GetNamedShardingSpec(arg_type, mesh_attr);
    if (auto memory_kind = func_op.getArgAttrOfType<StringAttr>(
            arg.getArgNumber(), kMemoryKindAttr)) {
      if (!mesh_and_spec.memory_kind.has_value()) {
        mesh_and_spec.memory_kind = memory_kind.getValue().str();
      }
      // TODO: b/374994155 - We should drop the memory kind from the
      // attributes and move it to types only. If for some reason that's not
      // possible we should at least be consistent and use the memory kind from
      // the type if then we need at least some verification of their
      // consistency.
    }
    io_sharding_and_mesh.input_specs.push_back(mesh_and_spec);
  }

  for (auto [index, type] : llvm::enumerate(func_op.getResultTypes())) {
    MeshTensorType result_type = cast<MeshTensorType>(type);
    sdy::MeshAttr mesh_attr = GetMeshOrFail(func_op, result_type.getMeshName());
    NamedSpmdShardingSpec mesh_and_spec =
        GetNamedShardingSpec(result_type, mesh_attr);
    if (auto memory_kind =
            func_op.getResultAttrOfType<StringAttr>(index, kMemoryKindAttr)) {
      if (!mesh_and_spec.memory_kind.has_value()) {
        mesh_and_spec.memory_kind = memory_kind.getValue().str();
      }
      // TODO: b/374994155 - We should drop the memory kind from the
      // attributes and move it to types only. If for some reason that's not
      // possible we should at least be consistent and use the memory kind from
      // the type if then we need at least some verification of their
      // consistency.
    }
    io_sharding_and_mesh.output_specs.push_back(mesh_and_spec);
  }

  return io_sharding_and_mesh;
}

FuncOp GetMainFunction(ModuleOp module) {
  FuncOp func = dyn_cast_or_null<FuncOp>(module.lookupSymbol("main"));
  SDY_CHECK(func);
  return func;
}

bool IsMpmdModule(ModuleOp module) { return !GetMpmdFunctions(module).empty(); }

bool IsMpmdFunction(FuncOp func_op) { return func_op->hasAttr(kTopologyAttr); }

bool IsSpmdFunction(FuncOp func_op) { return func_op->hasAttr(kMeshShapeAttr); }

bool IsDistributedFunction(FuncOp func_op) {
  return IsSpmdFunction(func_op) || IsMpmdFunction(func_op);
}

bool IsEntryPointFunction(FuncOp func_op) {
  return IsDistributedFunction(func_op) && func_op.isPublic();
}

TopologyAttr GetTopology(FuncOp func_op) {
  SDY_CHECK(IsMpmdFunction(func_op));
  return cast<TopologyAttr>(func_op->getAttr(kTopologyAttr));
}

TopologyAttr GetTopology(ModuleOp module_op) {
  SmallVector<FuncOp> mpmd_funcs = GetMpmdFunctions(module_op);
  SDY_CHECK(!mpmd_funcs.empty());
  return GetTopology(mpmd_funcs.front());
}

namespace {

void SetTopologyImpl(ArrayRef<NamedMeshAttr> meshes, FuncOp func) {
  // Make sure the MPMD dialect is loaded before we insert an MPMD attribute.
  func.getContext()->loadDialect<MpmdDialect>();
  func->setAttr(kTopologyAttr, TopologyAttr::get(func.getContext(), meshes));
}

}  // namespace

void SetTopology(
    const std::vector<std::pair<std::string, FlatMesh>>& topology_shape,
    Operation* op) {
  MLIRContext* context = op->getContext();

  // Make sure the MPMD dialect is loaded before we create an MPMD attribute.
  FuncOp func = sdy::getEnclosingOfType<FuncOp>(op);
  func.getContext()->loadDialect<MpmdDialect>();

  auto make_mesh_axis_attr = [&](const std::pair<std::string, int> p) {
    return sdy::MeshAxisAttr::get(context, p.first, p.second);
  };
  auto make_mesh_attr = [&](const FlatMesh& flat_mesh) {
    SmallVector<sdy::MeshAxisAttr> attr_axes(flat_mesh.size());
    llvm::transform(flat_mesh, attr_axes.begin(), make_mesh_axis_attr);
    return sdy::MeshAttr::get(context, attr_axes);
  };

  SmallVector<NamedMeshAttr> named_meshes(topology_shape.size());
  llvm::transform(topology_shape, named_meshes.begin(),
                  [&](const std::pair<std::string, FlatMesh> name_mesh_pair) {
                    return NamedMeshAttr::get(
                        context, name_mesh_pair.first,
                        make_mesh_attr(name_mesh_pair.second));
                  });
  SetTopologyImpl(named_meshes, func);
}

ArrayRef<NamedMeshAttr> GetTopologyMeshes(FuncOp func_op) {
  SDY_CHECK(IsMpmdFunction(func_op));
  ArrayRef<NamedMeshAttr> meshes = GetTopology(func_op).getMeshes();
  return meshes;
}

llvm::DenseMap<StringRef, sdy::MeshAttr> GetMeshesByName(
    ArrayRef<NamedMeshAttr> meshes) {
  llvm::DenseMap<StringRef, sdy::MeshAttr> meshes_by_name;
  meshes_by_name.reserve(meshes.size());
  for (NamedMeshAttr mesh : meshes) {
    meshes_by_name[mesh.getName()] = mesh.getMesh();
  }
  return meshes_by_name;
}

Type GetLocalTensorTypeFromMeshType(Value value, sdy::MeshAttr mesh_attr) {
  return cast<MeshTensorType>(value.getType()).getLocalTensorType(mesh_attr);
}

Type GetGlobalTensorTypeFromMeshType(Value value, sdy::MeshAttr) {
  return cast<MeshTensorType>(value.getType()).getGlobalTensorType();
}

TransferOp DynCastIntraMeshTransfer(Operation* op) {
  if (auto transfer_op = dyn_cast_or_null<TransferOp>(op);
      transfer_op && transfer_op.isIntraMesh()) {
    return transfer_op;
  }
  return nullptr;
}

TransferOp DynCastInterMeshTransfer(Operation* op) {
  if (auto transfer_op = dyn_cast_or_null<TransferOp>(op);
      transfer_op && transfer_op.isInterMesh()) {
    return transfer_op;
  }
  return nullptr;
}

bool IsInterMeshTransfer(Operation* op) { return DynCastInterMeshTransfer(op); }

namespace {

// Creates M multiple-source single-target edges, where M is the number of
// inputs of func_op and of operands of all CallOps in `call_ops`.
void CreateEdgesForFuncArguments(FuncOp func_op, ArrayRef<CallOp> call_ops,
                                 SmallVector<MpmdDataflowEdge>& all_edges) {
  for (BlockArgument arg : func_op.getArguments()) {
    SmallVector<OpOperand*> sources;
    sources.reserve(call_ops.size());
    for (CallOp call_op : call_ops) {
      OpOperand& operand = call_op->getOpOperand(arg.getArgNumber());
      sources.push_back(&operand);
    }
    all_edges.push_back(
        MpmdDataflowEdge{std::move(sources), /*targets=*/{arg}});
  }
}

// Creates N single-source multiple-target edges, where N is the number of
// outputs of all CallOps in `call_ops`.
void CreateEdgesForFuncResults(FuncOp func_op, ArrayRef<CallOp> call_ops,
                               SmallVector<MpmdDataflowEdge>& all_edges) {
  auto return_op =
      cast<func::ReturnOp>(func_op.getBody().front().getTerminator());
  for (OpOperand& operand : return_op->getOpOperands()) {
    SmallVector<Value> targets;
    targets.reserve(call_ops.size());
    for (CallOp call_op : call_ops) {
      targets.push_back(call_op.getResult(operand.getOperandNumber()));
    }
    all_edges.push_back(MpmdDataflowEdge{/*sources=*/{&operand},
                                         /*targets=*/std::move(targets)});
  }
}

}  // namespace

SmallVector<FuncOp> GetMpmdFunctions(ModuleOp module_op) {
  SmallVector<FuncOp> funs;
  for (auto func_op : module_op.getOps<FuncOp>()) {
    if (IsMpmdFunction(func_op)) {
      funs.push_back(func_op);
    }
  }
  return funs;
}

SmallVector<CallOp> GetCallOps(FuncOp func_op) {
  SmallVector<CallOp> call_ops;
  StringRef func_name = func_op.getSymName();

  // We walk through the ops before their regions, so we can skip over regions
  // which aren't relevant. The only regions which are relevant are mpmd funcs
  // and module ops (because they could contain mpmd funcs).
  func_op->getParentOfType<ModuleOp>()->walk<WalkOrder::PreOrder>(
      [&call_ops, func_name](Operation* op) {
        if (auto call_op = dyn_cast<CallOp>(op);
            call_op && call_op.getCallee() == func_name &&
            !call_op->use_empty()) {
          call_ops.push_back(call_op);
        } else if (isa<ModuleOp>(op)) {
          return WalkResult::advance();
        } else if (auto func = dyn_cast<FuncOp>(op);
                   func && IsMpmdFunction(func)) {
          return WalkResult::advance();
        } else if (dyn_cast<ForOp>(op)) {
          return WalkResult::advance();
        }
        return WalkResult::skip();
      });
  return call_ops;
}

SmallVector<MpmdDataflowEdge> GetMpmdDataflowEdgesForFuncArgs(FuncOp func_op) {
  // Find all CallOps that refer to this function.
  SmallVector<CallOp> call_ops = GetCallOps(func_op);

  if (call_ops.empty()) {
    return {};
  }

  SmallVector<MpmdDataflowEdge> all_edges;
  all_edges.reserve(func_op.getNumArguments());
  CreateEdgesForFuncArguments(func_op, call_ops, all_edges);
  return all_edges;
}

SmallVector<MpmdDataflowEdge> GetMpmdDataflowEdgesForFuncResults(
    FuncOp func_op) {
  // Find all CallOps that refer to this function.
  SmallVector<CallOp> call_ops = GetCallOps(func_op);

  if (call_ops.empty()) {
    return {};
  }

  SmallVector<MpmdDataflowEdge> all_edges;
  all_edges.reserve(func_op.getNumResults());
  CreateEdgesForFuncResults(func_op, call_ops, all_edges);
  return all_edges;
}

SmallVector<MpmdDataflowEdge> GetMpmdDataflowEdges(FuncOp func_op) {
  // Find all CallOps that refer to this function.
  SmallVector<CallOp> call_ops = GetCallOps(func_op);

  if (call_ops.empty()) {
    return {};
  }

  SmallVector<MpmdDataflowEdge> all_edges;
  all_edges.reserve(func_op.getNumResults() + func_op.getNumArguments());
  CreateEdgesForFuncResults(func_op, call_ops, all_edges);
  CreateEdgesForFuncArguments(func_op, call_ops, all_edges);
  return all_edges;
}

FragmentOp WrapOpWithFragment(
    Operation* op, StringRef mesh_name, RewriterBase& rewriter,
    std::function<bool(OpOperand&)> should_replace_use) {
  // We set the insertion point right before `op` so assigns of operands will be
  // in the right place regardless of previous insertion point.
  rewriter.setInsertionPoint(op);

  llvm::SetVector<Value> operands_and_free_vars(op->operand_begin(),
                                                op->operand_end());
  getUsedValuesDefinedAbove(op->getRegions(), operands_and_free_vars);

  MLIRContext* ctx = rewriter.getContext();
  Location loc = op->getLoc();

  sdy::MeshAttr mesh_attr = GetMeshOrFail(op, mesh_name);

  // Assign all operands and free tensor vars to fully replicated mesh tensors
  // of the same mesh as `other_mesh_tensor`, which would become the operands to
  // the fragment op.
  SmallVector<Value> fragment_operands;
  fragment_operands.reserve(operands_and_free_vars.size());
  for (Value value : operands_and_free_vars) {
    fragment_operands.push_back(
        rewriter.create<AssignOp>(loc, value, mesh_name, mesh_attr));
  }

  // The fragment result types are fully replicated mesh tensors of the same
  // mesh as `other_mesh_tensor`, with a local type corresponding to the
  // respective result type of `op`.
  SmallVector<Type> fragment_result_types;
  fragment_result_types.reserve(op->getNumResults());
  for (Type result_type : op->getResultTypes()) {
    auto local_type = cast<RankedTensorType>(result_type);
    fragment_result_types.push_back(MeshTensorType::getFullyReplicated(
        ctx, mesh_name, mesh_attr, local_type));
  }

  FragmentOp fragment_op = FragmentOp::createMeshFragmentWithGlobalBody(
      loc, /*user_origin=*/{}, mesh_name, fragment_operands,
      fragment_result_types, rewriter,
      [&](ArrayRef<Value> args,
          OpBuilder& block_builder) -> SmallVector<Value> {
        // Map the original operand or free tensor var to the corresponding
        // argument.
        IRMapping mapping;
        mapping.map(operands_and_free_vars, args);
        // Clone the original `op` inside the fragment's block using the
        // populated IRMapping, and return the results of the cloned op.
        return block_builder.clone(*op, mapping)->getResults();
      });

  // Unassign all fragment results and replace all uses of `op` with the
  // corresponding unassign op for which `should_replace_use` returns true.
  for (auto [original_result, fragment_result] :
       llvm::zip(op->getResults(), fragment_op.getResults())) {
    auto unassign_op = rewriter.create<UnassignOp>(loc, fragment_result);
    rewriter.replaceUsesWithIf(original_result, unassign_op,
                               should_replace_use);
  }
  return fragment_op;
}

std::vector<int64_t> GetTransposeCounts(FragmentOp fragment) {
  ArrayAttr origin = fragment.getOrigin();
  std::vector<int64_t> transpose_counts;
  for (Attribute origin_attr : origin) {
    auto user_origin = cast<UserOriginAttr>(origin_attr);
    transpose_counts.push_back(user_origin.getTransposeCount());
  }
  return transpose_counts;
}

std::optional<int64_t> TryToFindSingleTransposeCount(FragmentOp fragment) {
  std::vector<int64_t> transpose_counts = GetTransposeCounts(fragment);
  if (!transpose_counts.empty() &&
      std::adjacent_find(transpose_counts.begin(), transpose_counts.end(),
                         std::not_equal_to<>()) == transpose_counts.end()) {
    return transpose_counts.front();
  }
  return std::nullopt;
}

std::optional<uint32_t> TryToFindCallCounter(FragmentOp fragment) {
  if (auto count = fragment->getAttrOfType<IntegerAttr>(kCallCounterAttrName)) {
    SDY_CHECK(count.getType().isUnsignedInteger(32));
    return count.getUInt();
  }
  return std::nullopt;
}

bool IsExecutedImmediatelyAfter(FragmentOp fragment1, FragmentOp fragment2) {
  Operation* current = fragment1->getNextNode();
  while (current) {
    if (auto fragment = dyn_cast<FragmentOp>(current);
        fragment && fragment.getMeshName() == fragment1.getMeshName()) {
      return fragment2 == fragment;
    }
    current = current->getNextNode();
  }
  return false;
}

bool HasHomogeneousTopology(FuncOp func) {
  ArrayRef<NamedMeshAttr> named_meshes = GetTopologyMeshes(func);
  DenseSet<sdy::MeshAttr> meshes;
  for (NamedMeshAttr named_mesh : named_meshes) {
    meshes.insert(named_mesh.getMesh());
  }
  return meshes.size() == 1;
}

ArrayAttr GetFragmentOriginUnion(FragmentOp fragment1, FragmentOp fragment2,
                                 RewriterBase& rewriter) {
  std::vector<Attribute> merged_origin(fragment1.getOrigin().begin(),
                                       fragment1.getOrigin().end());
  for (auto attr : fragment2.getOrigin()) {
    if (!llvm::is_contained(fragment1.getOrigin().getValue(), attr)) {
      merged_origin.push_back(attr);
    }
  }
  return rewriter.getArrayAttr(merged_origin);
}

bool IsLoweredWithSdy(ModuleOp module) {
  return module->hasAttr(kIsSdyLowered);
}

bool IsRemat(mlir::Operation* op) { return op->hasAttr(kRematAttributeName); }

void MarkAsRemat(mlir::Operation* op, RewriterBase& rewriter) {
  op->setAttr(kRematAttributeName, rewriter.getUnitAttr());
}

std::pair<StringRef, std::optional<StringRef>>
TryToExtractMemoryKindFromMeshName(StringRef mesh_name) {
  std::pair<StringRef, StringRef> mesh_and_memory_kind = mesh_name.split('#');
  if (mesh_and_memory_kind.second.empty()) {
    return {mesh_name, std::nullopt};
  }
  return mesh_and_memory_kind;
}

void UpdateValueTypeWithSharding(Value value,
                                 sdy::TensorShardingAttr sharding) {
  if (!sharding) {
    return;
  }
  value.setType(
      cast<MeshTensorType>(value.getType()).replaceSharding(sharding));
}

std::optional<Location> GetResultInfoLoc(FuncOp func, int64_t result_index) {
  auto result_info =
      func.getResultAttrOfType<StringAttr>(result_index, kJaxResultInfoAttr);
  if (!result_info) {
    return std::nullopt;
  }
  return NameLoc::get(result_info);
}

FailureOr<sdy::MeshAttr> GetMeshAttr(Operation* op) {
  if (auto fragmentOp = sdy::getEnclosingOfType<FragmentOp>(op)) {
    return GetMeshAttr(op, fragmentOp.getMeshName());
  }

  FuncOp func = sdy::getEnclosingOfType<FuncOp>(op);
  auto meshAttr = func->getAttrOfType<sdy::MeshAttr>(kMeshShapeAttr);
  if (!meshAttr) {
    return op->emitError("Function does not have a ")
           << kMeshShapeAttr << " attribute: " << func.getSymName();
  }
  return meshAttr;
}

FailureOr<sdy::MeshAttr> GetMeshAttr(Operation* op, StringRef mesh_name) {
  FuncOp func = sdy::getEnclosingOfType<FuncOp>(op);
  if (!func->hasAttr(kTopologyAttr)) {
    return op->emitError("Function does not have a ")
           << kTopologyAttr << " attribute: " << func.getSymName();
  }
  for (NamedMeshAttr named_mesh_attr :
       cast<TopologyAttr>(func->getAttr(kTopologyAttr)).getMeshes()) {
    if (named_mesh_attr.getName() == mesh_name) {
      return named_mesh_attr.getMesh();
    }
  }
  return op->emitError("Topology doesn't have a mesh with name: ") << mesh_name;
}

sdy::MeshAttr GetMeshOrFail(Operation* op) {
  FailureOr<sdy::MeshAttr> mesh_attr = GetMeshAttr(op);
  SDY_CHECK(succeeded(mesh_attr));
  return *mesh_attr;
}

sdy::MeshAttr GetMeshOrFail(Operation* op, StringRef mesh_name) {
  FailureOr<sdy::MeshAttr> mesh_attr = GetMeshAttr(op, mesh_name);
  SDY_CHECK(succeeded(mesh_attr));
  return *mesh_attr;
}

Operation* FindAnnotatedOperation(ModuleOp module, StringRef annotation) {
  Operation* result = nullptr;
  module->walk([&](Operation* op) {
    if (op->hasAttr(annotation)) {
      result = op;
    }
  });
  return result;
}

std::optional<ReductionType> GetReductionOpType(Operation* op) {
  if (!op) {
    return std::nullopt;
  }
  if (isa<stablehlo::AddOp>(op)) {
    return ReductionType::kAdd;
  }
  if (isa<stablehlo::MaxOp>(op)) {
    return ReductionType::kMax;
  }
  if (isa<stablehlo::MinOp>(op)) {
    return ReductionType::kMin;
  }
  if (isa<stablehlo::MulOp>(op)) {
    return ReductionType::kMul;
  }
  if (isa<stablehlo::OrOp>(op)) {
    return ReductionType::kOr;
  }
  if (isa<stablehlo::AndOp>(op)) {
    return ReductionType::kAnd;
  }
  return std::nullopt;
}

Operation* CreateStablehloReduceOp(ReductionType reduction_type,
                                   ValueRange values, Location loc,
                                   OpBuilder& builder) {
  switch (reduction_type) {
    case ReductionType::kAdd:
      return builder.create<stablehlo::AddOp>(loc, values);
    case ReductionType::kMul:
      return builder.create<stablehlo::MulOp>(loc, values);
    case ReductionType::kMax:
      return builder.create<stablehlo::MaxOp>(loc, values);
    case ReductionType::kMin:
      return builder.create<stablehlo::MinOp>(loc, values);
    case ReductionType::kOr:
      return builder.create<stablehlo::OrOp>(loc, values);
    case ReductionType::kAnd:
      return builder.create<stablehlo::AndOp>(loc, values);
    case ReductionType::kNone:
      return nullptr;
  }
  llvm_unreachable("unknown ReductionType");
}

std::optional<ReductionType> ComputeReductionType(Block& block) {
  Operation* ret = block.getTerminator();
  if (!ret || block.getNumArguments() != (ret->getNumOperands() * 2)) {
    return std::nullopt;
  }

  std::optional<ReductionType> reduction_type;
  for (OpOperand& ret_operand : ret->getOpOperands()) {
    Operation* reduction_op = ret_operand.get().getDefiningOp();
    std::optional<ReductionType> inner_type = GetReductionOpType(reduction_op);
    if (!inner_type.has_value() ||
        (reduction_type.has_value() && reduction_type.value() != inner_type)) {
      return std::nullopt;
    }
    reduction_type = inner_type;

    int result_idx = ret_operand.getOperandNumber();
    BlockArgument arg1 = block.getArgument(result_idx);
    BlockArgument arg2 = block.getArgument(result_idx + ret->getNumOperands());
    if (!((reduction_op->getOperand(0) == arg1 &&
           reduction_op->getOperand(1) == arg2) ||
          (reduction_op->getOperand(0) == arg2 &&
           reduction_op->getOperand(1) == arg1))) {
      return std::nullopt;
    }
  }

  return reduction_type;
}

}  // namespace mlir::mpmd

