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

#ifndef SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESHES_WITH_ORIGINS_H_
#define SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESHES_WITH_ORIGINS_H_

#include <stdbool.h>

#include <optional>

#include "llvm/ADT/MapVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace mlir::mpmd {

inline constexpr StringRef kWildcardMesh = "*";

using MeshToOrigins = llvm::MapVector<StringRef, SetVector<OriginAttr>>;

// Wrapper around MeshesWithOriginsAttr to facilitate merging.
//
// This is used for mesh inference, for propagation of use and src sets. If
// `has_meshes_specified_` is false, then the set is considered to be all
// meshes.
class MeshesWithOrigins {
 public:
  MeshesWithOrigins() = default;

  explicit MeshesWithOrigins(MeshesWithOriginsAttr meshes_with_origins) {
    if (!meshes_with_origins) {
      return;
    }
    mesh_to_origins_ = MeshToOrigins();
    for (MeshWithOriginsAttr origin : meshes_with_origins.getValue()) {
      insert(origin);
    }

    if (!wildcard_origins_.empty()) {
      // For now, we only allow the wildcard mesh if there are also other meshes
      // specified. This makes the logic simpler for the rest of the code.
      SDY_CHECK(!mesh_to_origins_->empty())
          << "Wildcard mesh should only be specified if there are also other "
             "meshes specified.";
    }
  }

  explicit MeshesWithOrigins(
      mlir::mpmd::MeshWithOriginsAttr mesh_with_origins) {
    SDY_CHECK(mesh_with_origins)
        << "MeshesWithOrigins should not be initialized "
           "with a null MeshWithOriginsAttr.";
    SDY_CHECK(mesh_with_origins.getMeshName() != kWildcardMesh)
        << "Wildcard origin not allowed for this constructor.";
    insert(mesh_with_origins);
  }

  // Constructs a UseSet from a MeshesWithOriginsAttr. If the attr is empty,
  // then meshes_with_origins_ will be initialized to an empty set to make clear
  // that the set is empty.
  static MeshesWithOrigins CreateUseSet(
      MeshesWithOriginsAttr meshes_with_origins) {
    MeshesWithOrigins use_set(meshes_with_origins);
    if (!use_set.mesh_to_origins_) {
      use_set.mesh_to_origins_ = MeshToOrigins();
    }
    return use_set;
  }

  // Takes the union of the meshes with origins from the other set. Ignores
  // unspecified sets. If the current set is unspecified, it will be
  // initialized to the other set.
  void Union(const MeshesWithOrigins& other);

  // Takes the intersection of the meshes with origins from the other set.
  // Ignores unspecified sets. If the current set is unspecified, it will be
  // initialized to the other set.
  void Intersect(const MeshesWithOrigins& other);

  MeshesWithOriginsAttr ToAttr(OpBuilder& builder) const;
  SmallVector<MeshWithOriginsAttr> ToArray(MLIRContext* context) const;

  SetVector<StringRef> MeshNames(bool include_wildcard_mesh = false) const;
  std::optional<SetVector<StringRef>> MaybeMeshNames(
      bool include_wildcard_mesh = false) const;
  SetVector<StringRef> MeshNamesOrEmpty(
      bool include_wildcard_mesh = false) const;
  bool HasSameMeshes(const MeshesWithOrigins& other) const;

  // Returns the prioritized mesh name. Useful for making mesh assignments.
  // Assumes that meshes are specified.
  // Prioritizes as follows (in order):
  // 1. Prioritizes meshes from `preferred_mesh_names` if any.
  // 2. Prioritizes meshes which are not low priority (see `LowPriorityOrigin`).
  // 3. Prioritizes meshes based on lexicographic order of the mesh name.
  std::optional<StringRef> GetPrioritizedMeshName(
      const SetVector<StringRef>& preferred_mesh_names = {}) const;

  void insert(MeshWithOriginsAttr origin);

  int size() const { return mesh_to_origins_ ? mesh_to_origins_->size() : -1; }
  bool empty() const { return mesh_to_origins_ && mesh_to_origins_->empty(); }
  bool has_meshes_specified() const { return mesh_to_origins_.has_value(); }
  bool has_wildcard_mesh() const { return !wildcard_origins_.empty(); }

  explicit operator bool() const { return has_meshes_specified(); }

 protected:
  std::optional<MeshToOrigins> mesh_to_origins_;
  SetVector<OriginAttr> wildcard_origins_;
};

}  // namespace mlir::mpmd

#endif  // SHARDY_DIALECT_MPMD_TRANSFORMS_IMPORT_MESHES_WITH_ORIGINS_H_
