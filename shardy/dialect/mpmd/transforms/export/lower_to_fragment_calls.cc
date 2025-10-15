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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/fragment_arg_res_attrs.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/export/naming_utils.h"
#include "shardy/dialect/mpmd/transforms/export/passes.h"  // IWYU pragma: keep
#include "shardy/dialect/mpmd/transforms/export/utils.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "mlir/Support/WalkResult.h"

namespace mlir::mpmd {

#define GEN_PASS_DEF_LOWERTOFRAGMENTCALLSPASS
#include "shardy/dialect/mpmd/transforms/export/passes.h.inc"

namespace {

using ::mlir::func::FuncOp;

constexpr StringRef kFragmentNamePrefix = "p";
constexpr StringRef kGroupId = "group_id";
constexpr StringRef kGroupName = "group_name";

void SetIntegerAttr(Operation* op, StringRef name, int64_t value,
                    IRRewriter& rewriter) {
  op->setAttr(name, rewriter.getI64IntegerAttr(value));
}

std::optional<int64_t> GetIntegerAttr(Operation* op, StringRef name) {
  if (auto attr = dyn_cast_or_null<IntegerAttr>(op->getAttr(name))) {
    return attr.getInt();
  }
  return std::nullopt;
}

// Two fragments are considered to be equivalent if their bodies are equivalent
// and they have the same mesh name. Note that using the mesh name rather than
// the mesh attribute in the topology is important to avoid false sharing in
// systems where we have the same mesh attribute bound to different names.
// That said, there may be situations where two meshes are equal as mesh
// attributes, and spanning over the same devices, but have different names,
// in which case we may want to still merge (to reduce code bloat). We do not,
// currently, have such a mechanism to know whether two equal meshes span over
// the same devices and hence we pick the more conservative choice of comparing
// mesh names, which is important for heterogeneous systems.
//
// TODO(dvytin): consider merging even across different equivalent meshes.
struct FragmentBodyEquivalenceBaseInfo : public DenseMapInfo<FragmentOp> {
  static unsigned getHashValue(FragmentOp fragment_op,
                               bool group_across_meshes) {
    llvm::hash_code hash;
    if (group_across_meshes) {
      // Hash the mesh shape.
      hash = hash_value(*mpmd::GetMeshAttr(fragment_op));
    } else {
      // Hash the mesh name to avoid deduping fragments executed on different
      // meshes (which may be unsafe in heterogeneous settings).
      hash = llvm::hash_value(fragment_op.getMeshName());
    }
    // Hash the fragment argument attributes.
    hash = llvm::hash_combine(hash, fragment_op->getAttr(kArgAttrName));
    hash = llvm::hash_combine(hash, fragment_op->getAttr(kResAttrName));

    // Hash the fragment body.
    //
    // Note that since we don't hash the operands of operations in the body or
    // take the number of block arguments into account, there might be
    // collisions between the hash value of two fragments that aren't truly
    // equivalent, e.g. they have exactly the same operations but the order of
    // operands is different. This is ok because the hash value is only used to
    // lookup the bucket for a fragment, where hashing the operations' signature
    // is strong enough for minimising collisions, and those that do occur are
    // resolved with the isEqual check below.
    fragment_op.getRegion().walk([&](Operation* op) {
      hash = llvm::hash_combine(
          hash, OperationEquivalence::computeHash(
                    op, /*hashOperands=*/OperationEquivalence::ignoreHashValue,
                    /*hashResults=*/OperationEquivalence::ignoreHashValue,
                    OperationEquivalence::IgnoreLocations));
    });

    return hash;
  }

  static bool isEqual(FragmentOp lhs, FragmentOp rhs,
                      bool group_across_meshes) {
    if (lhs == rhs) {
      return true;
    }
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey()) {
      return false;
    }

    bool equal_meshes;
    if (group_across_meshes) {
      equal_meshes = *mpmd::GetMeshAttr(lhs) == *mpmd::GetMeshAttr(rhs);
    } else {
      // Compare the mesh names to avoid deduping fragments executed on
      // different meshes (which may be unsafe in heterogeneous settings).
      equal_meshes = lhs.getMeshName() == rhs.getMeshName();
    }

    return equal_meshes &&
           lhs->getAttr(kArgAttrName) == rhs->getAttr(kArgAttrName) &&
           lhs->getAttr(kResAttrName) == rhs->getAttr(kResAttrName) &&
           OperationEquivalence::isRegionEquivalentTo(
               &lhs.getRegion(), &rhs.getRegion(),
               OperationEquivalence::IgnoreLocations);
  }
};

struct FragmentBodyEquivalenceCrossMeshGroupingInfo
    : FragmentBodyEquivalenceBaseInfo {
  static unsigned getHashValue(FragmentOp fragment_op) {
    return FragmentBodyEquivalenceBaseInfo::getHashValue(
        fragment_op, /*group_across_meshes=*/true);
  }

  static bool isEqual(FragmentOp lhs, FragmentOp rhs) {
    return FragmentBodyEquivalenceBaseInfo::isEqual(
        lhs, rhs, /*group_across_meshes=*/true);
  }
};

struct FragmentBodyEquivalenceSameMeshGroupingInfo
    : FragmentBodyEquivalenceBaseInfo {
  static unsigned getHashValue(FragmentOp fragment_op) {
    return FragmentBodyEquivalenceBaseInfo::getHashValue(
        fragment_op, /*group_across_meshes=*/false);
  }

  static bool isEqual(FragmentOp lhs, FragmentOp rhs) {
    return FragmentBodyEquivalenceBaseInfo::isEqual(
        lhs, rhs, /*group_across_meshes=*/false);
  }
};

// Auxiliary data structure for fragment grouping.
struct FragmentGroupInfo {
  std::optional<int64_t> hbm_bytes;
  // The unique identifier for this group.
  int64_t group_id;
  // Call-sites for all fragments in this group, grouped by mesh.
  MeshToCallSites mesh_call_sites = MeshToCallSites();
};

StringRef DropJitPrefix(StringRef name) {
  const std::string_view kJitPrefix = "jit_";
  // jax jitted modules have a jit_ prefix, which we drop for simplicity.
  if (name.starts_with(kJitPrefix)) {
    return name.substr(kJitPrefix.size());
  }
  return name;
}

StringRef GetModuleName(ModuleOp module_op) {
  return module_op.getName().value_or("main");
}

bool IsAllForward(ModuleOp module_op) {
  bool is_all_forward = true;
  module_op.walk([&](FragmentOp fragment) {
    for (Attribute attr : fragment.getOrigin().getValue()) {
      if (cast<UserOriginAttr>(attr).getTransposeCount() != 0) {
        is_all_forward = false;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return is_all_forward;
}

std::string PrettyPrintUserOrigin(ArrayRef<Attribute> origins, bool all_forward) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  auto concat_origin = [&](UserOriginAttr origin) -> void {
    stream << origin.getUserName().getValue();
    if (!all_forward) {
      stream << "(";
      if (origin.getTransposeCount() == 0) {
        stream << "fwd";
      } else if (origin.getTransposeCount() == 1) {
        stream << "bwd";
      } else {
        stream << "TransposeCount=" << origin.getTransposeCount();
      }
      stream << ")";
    }
  };

  stream << "[";
  if (origins.empty()) {
    stream << "]";
    return result;
  }
  Attribute tail = origins.back();
  for (Attribute attr : origins.drop_back(1)) {
    auto origin_attr = cast<UserOriginAttr>(attr);
    concat_origin(origin_attr);
    stream << ", ";
  }
  concat_origin(cast<UserOriginAttr>(tail));
  stream << "]";
  return result;
}

// Groups fragments by body and mesh shape equivalence, and gives each group a
// unique identifier, and an updated hbm reserved bytes number. Marks each
// fragment with its group id, name, and hbm bytes id.
//
// Returns all fragments in the module.
template <typename FragmentEquivalenceInfo>
std::vector<FragmentOp> GroupFragmentsAndMarkWithGroupName(
    ModuleOp module_op, IRRewriter& rewriter, bool is_all_forward) {
  DenseMap<FragmentOp, FragmentGroupInfo, FragmentEquivalenceInfo> fragment_map;

  // Step 1: Group all fragments by body and mesh shape equivalence, and give
  // each group a unique identifier, and an updated hbm reserved bytes number.
  int64_t group_id = 0;
  std::vector<FragmentOp> all_fragments;
  // Walk the module, collecting fragments in program order, as we rely on that
  //  below to log the schedule.
  module_op.walk([&](FragmentOp fragment) {
    all_fragments.push_back(fragment);

    std::optional<int64_t> hbm_reserved_bytes =
        GetIntegerAttr(fragment, kReservedHbmBytes);

    auto [it, inserted] =
        fragment_map.try_emplace(fragment, FragmentGroupInfo{});
    FragmentGroupInfo& fragment_group = it->getSecond();
    if (inserted) {
      fragment_group.group_id = group_id++;
    }
    // TODO(dvytin): Experiment with different policies.
    // std::nullopt < any int64_t, hence std::max works with std::nullopt.
    fragment_group.hbm_bytes =
        std::max(fragment_group.hbm_bytes, hbm_reserved_bytes);

    std::string name =
        GetFullNameFromMetadata(fragment.getOrigin().getValue(),
                                fragment.getStageId(), is_all_forward);
    std::optional<uint32_t> call_counter = TryToFindCallCounter(fragment);
    fragment_group.mesh_call_sites[fragment.getMeshName()].emplace_back(
        std::move(name), call_counter);
  });

  // Step 2: Mark all fragments with their calculated group ids, names, and
  // hbm bytes.
  for (auto fragment : all_fragments) {
    const auto& [hbm_bytes, group_id, call_sites] = fragment_map[fragment];
    if (hbm_bytes.has_value()) {
      SetIntegerAttr(fragment, kReservedHbmBytes, *hbm_bytes, rewriter);
    }
    SetIntegerAttr(fragment, kGroupId, group_id, rewriter);
    std::string group_name;

    llvm::raw_string_ostream stream(group_name);
    std::string fragment_name = GetCallSitesSummaryName(call_sites);
    // Append a unique id. We do this first to guarantee it isn't affected by
    // truncation.
    stream << kFragmentNamePrefix << group_id << "_";

    // Append the fragment name.
    stream << fragment_name << ".";

    // Find function name in the module.
    StringRef module_name = GetModuleName(module_op);
    // Drop the jit_ prefix if present.
    module_name = DropJitPrefix(module_name);
    // Truncate the function name to 32 characters.
    std::string truncated_module_name = Truncate(module_name, 32);
    stream << truncated_module_name;

    int max_group_name_length = 64 - truncated_module_name.size();
    group_name = Truncate(group_name, max_group_name_length);
    fragment->setAttr(kGroupName, rewriter.getStringAttr(group_name));
  }

  return all_fragments;
}

class LowerToFragmentCallsPass
    : public impl::LowerToFragmentCallsPassBase<LowerToFragmentCallsPass> {
  using LowerToFragmentCallsPassBase::LowerToFragmentCallsPassBase;

  void runOnOperation() final {
    ModuleOp module_op = getOperation();
    MLIRContext& ctx = getContext();
    bool is_sdy_partitioned = mpmd::IsLoweredWithSdy(module_op);
    bool is_all_forward = IsAllForward(module_op);

    IRRewriter rewriter(&ctx);


    std::vector<FragmentOp> all_fragments =
        groupAcrossMeshes
            ? GroupFragmentsAndMarkWithGroupName<
                  FragmentBodyEquivalenceCrossMeshGroupingInfo>(
                  module_op, rewriter, is_all_forward)
            : GroupFragmentsAndMarkWithGroupName<
                  FragmentBodyEquivalenceSameMeshGroupingInfo>(
                  module_op, rewriter, is_all_forward);


    // Step 3: Log the fragment naming per mesh, for debugging purposes.
    if (auto func = dyn_cast_or_null<FuncOp>(module_op.lookupSymbol("main"))) {
      ArrayRef<mpmd::NamedMeshAttr> meshes = mpmd::GetTopologyMeshes(func);
      for (mpmd::NamedMeshAttr mesh : meshes) {
        if (verboseLogging) {
          SDY_LOG(INFO) << "Module "
                        << std::string_view(GetModuleName(module_op))
                        << " on mesh " << std::string_view(mesh.getName())
                        << " will execute the following XLA programs:";
        }
        for (FragmentOp fragment : all_fragments) {
          if (fragment.getMeshName() != mesh.getName()) {
            continue;
          }
          std::string stage_and_call_counter;
          llvm::raw_string_ostream stream(stage_and_call_counter);
          if (std::optional<int64_t> stage_id = fragment.getStageId()) {
            stream << ", stage id " << *stage_id;
          }
          if (std::optional<uint32_t> call_counter =
                  TryToFindCallCounter(fragment)) {
            stream << ", call counter " << *call_counter;
          }
          StringRef group_name =
              fragment->getAttrOfType<StringAttr>(kGroupName).getValue();
          if (verboseLogging) {
            SDY_LOG(INFO) << "\t- " << group_name.str()
                          << " from program with origins "
                          << PrettyPrintUserOrigin(
                                 fragment.getOrigin().getValue(),
                                 is_all_forward)
                          << stage_and_call_counter << ".";
          }
        }
      }
    }

    // Step 4: For each fragment, extract a function if not already done.
    SymbolTableCollection symbol_table_collection;
    SymbolTable& symbol_table =
        symbol_table_collection.getSymbolTable(module_op);

    for (FragmentOp fragment : all_fragments) {
      // We use the marked attributes instead of looking up in fragment_map
      // because we will be doing rewriter replacements.
      std::optional<int64_t> hbm_bytes =
          GetIntegerAttr(fragment, kReservedHbmBytes);
      StringRef group_name =
          fragment->getAttrOfType<StringAttr>(kGroupName).getValue();
      if (!symbol_table.lookup(group_name)) {
        // This is the first encountered fragment for the equivalence group.
        Block& block = *fragment.getBody();
        auto func_op = FuncOp::create(
            fragment.getLoc(), group_name,
            FunctionType::get(&ctx, block.getArgumentTypes(),
                              block.getTerminator()->getOperandTypes()));
        // TODO(b/425890780): Remove references to kMeshShapeAttr.
        func_op->setAttr(mpmd::kMeshShapeAttr, *mpmd::GetMeshAttr(fragment));
        // TODO(b/298362694): Try to experiment with different policies when
        // selecting a value for the flag.
        if (hbm_bytes.has_value()) {
          SetIntegerAttr(func_op, kReservedHbmBytes, *hbm_bytes, rewriter);
        }
        // Set the argument and result attributes, for now it only has aliasing
        // and host offload information.
        if (Attribute arg_attr = fragment->getAttr(kArgAttrName)) {
          func_op.setArgAttrsAttr(cast<ArrayAttr>(arg_attr));
        }
        if (Attribute res_attr = fragment->getAttr(kResAttrName)) {
          func_op.setResAttrsAttr(cast<ArrayAttr>(res_attr));
        }

        // Set the argument and result attr to include the sharding in the type.
        // This is needed for shardy XLA to read the sharding later when
        // importing.
        if (is_sdy_partitioned) {
          for (OpOperand& arg : fragment->getOpOperands()) {
            auto arg_type = dyn_cast<MeshTensorType>(arg.get().getType());
            if (!arg_type) {
              continue;
            }
            if (sdy::TensorShardingAttr arg_sharding = arg_type.getSharding()) {
              func_op.setArgAttr(arg.getOperandNumber(), sdy::kShardingAttr,
                                 arg_sharding);
            }
          }
          for (auto result : fragment->getResults()) {
            auto res_type = dyn_cast<MeshTensorType>(result.getType());
            if (!res_type) {
              continue;
            }
            if (sdy::TensorShardingAttr res_sharding = res_type.getSharding()) {
              sdy::setFuncResultSharding(func_op, result.getResultNumber(),
                                         res_sharding);
            }
          }
        }

        sdy::inlineRegionAndConvertTerminatorOp<func::ReturnOp>(
            fragment.getRegion(), func_op.getBody());
        symbol_table.insert(func_op);
      }
      rewriter.setInsertionPoint(fragment);
      bool is_remat = mpmd::IsRemat(fragment);
      bool is_gspmd_partitioned = fragment->hasAttr(kIsGspmdPartitioned);
      auto fragment_call_op = rewriter.replaceOpWithNewOp<FragmentCallOp>(
          fragment, fragment.getResultTypes(), fragment.getOperands(),
          fragment.getOrigin(), fragment.getMeshName(), group_name);
      // The following is just for debugging (nb: remat is not an explicit
      // attribute of fragment calls.).
      if (is_remat) {
        mpmd::MarkAsRemat(fragment_call_op, rewriter);
      }
      if (is_gspmd_partitioned) {
        fragment_call_op->setAttr(kIsGspmdPartitioned, rewriter.getUnitAttr());
      }
      if (is_sdy_partitioned) {
        fragment_call_op->setAttr(kIsSdyPartitioned, rewriter.getUnitAttr());
      }
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
