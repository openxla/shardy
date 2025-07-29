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

#include "shardy/dialect/mpmd/transforms/import/meshes_with_origins.h"

#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "shardy/common/logging.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/transforms/import/mesh_inference_origins.h"

namespace mlir::mpmd {

void MeshesWithOrigins::Union(const MeshesWithOrigins& other) {
  if (!other.has_meshes_specified()) {
    return;
  }
  if (!has_meshes_specified()) {
    mesh_to_origins_ = other.mesh_to_origins_;
    return;
  }

  for (const auto& [mesh, origins] : *other.mesh_to_origins_) {
    (*mesh_to_origins_)[mesh].insert_range(origins);
  }

  // There's no point adding the wildcard origins to every mesh for the Union,
  // and it would just be noise. So we just keep track of the wildcard origins
  // separately and don't add them to the mesh_to_origins_ map.
  wildcard_origins_.insert_range(other.wildcard_origins_);
}

void MeshesWithOrigins::Intersect(const MeshesWithOrigins& other) {
  if (!other.has_meshes_specified()) {
    return;
  }

  if (!has_meshes_specified()) {
    mesh_to_origins_ = other.mesh_to_origins_;
    wildcard_origins_ = other.wildcard_origins_;
    return;
  }

  SmallVector<StringRef> meshes_to_remove;

  // For each mesh in our current set, we do the following:
  // - If the mesh is in the current set, then we merge the origins.
  // - If the mesh is not in the current set, then we use the wildcard mesh
  //   (if it exists) or remove the mesh.
  for (const auto& [mesh, origins] : *mesh_to_origins_) {
    if (other.mesh_to_origins_->contains(mesh)) {
      (*mesh_to_origins_)[mesh].insert_range(
          other.mesh_to_origins_->lookup(mesh));
    } else if (other.has_wildcard_mesh()) {
      (*mesh_to_origins_)[mesh].insert_range(other.wildcard_origins_);
    } else {
      meshes_to_remove.push_back(mesh);
    }
  }

  // Now we handle the case where the current set has a wildcard mesh, and add
  // all the meshes from (other set - current set) to the current set.
  if (has_wildcard_mesh()) {
    // We need to add all the meshes from the other set.
    for (const auto& [mesh, origins] : *other.mesh_to_origins_) {
      if (!mesh_to_origins_->contains(mesh)) {
        (*mesh_to_origins_)[mesh].insert_range(origins);
        (*mesh_to_origins_)[mesh].insert_range(wildcard_origins_);
      }
    }
  }

  for (const auto& mesh : meshes_to_remove) {
    mesh_to_origins_->erase(mesh);
  }

  // If both have wildcard mesh, then we join the origins.
  // If neither has wildcard mesh, then it doesn't matter since both are
  // empty.
  // If only one has wildcard mesh, then we need to clear the wildcard
  // origins.
  if (has_wildcard_mesh() == other.has_wildcard_mesh()) {
    wildcard_origins_.insert_range(other.wildcard_origins_);
  } else {
    wildcard_origins_.clear();
  }
}

void MeshesWithOrigins::insert(MeshWithOriginsAttr origin) {
  if (!mesh_to_origins_) {
    mesh_to_origins_ = MeshToOrigins();
  }

  if (origin.getMeshName() == kWildcardMesh) {
    SDY_CHECK(!origin.getOrigins().empty())
        << "Wildcard mesh must have an origin.";
    wildcard_origins_.insert_range(origin.getOrigins());
  } else {
    (*mesh_to_origins_)[origin.getMeshName()].insert_range(origin.getOrigins());
  }
}

MeshesWithOriginsAttr MeshesWithOrigins::ToAttr(
    OpBuilder& builder) const {
  if (!has_meshes_specified()) {
    return nullptr;
  }

  return MeshesWithOriginsAttr::get(builder.getContext(),
                                    ToArray(builder.getContext()));
}

SmallVector<MeshWithOriginsAttr> MeshesWithOrigins::ToArray(
    MLIRContext* context) const {
  if (!has_meshes_specified()) {
    return {};
  }

  SmallVector<MeshWithOriginsAttr> meshes_with_origins;
  meshes_with_origins.reserve(mesh_to_origins_->size());
  for (const auto& [mesh, origins] : *mesh_to_origins_) {
    meshes_with_origins.push_back(
        MeshWithOriginsAttr::get(context, mesh, origins.getArrayRef()));
  }

  if (has_wildcard_mesh()) {
    meshes_with_origins.push_back(MeshWithOriginsAttr::get(
        context, kWildcardMesh, wildcard_origins_.getArrayRef()));
  }

  return meshes_with_origins;
}

SetVector<StringRef> MeshesWithOrigins::MeshNames(
    bool include_wildcard_mesh) const {
  SDY_CHECK(has_meshes_specified()) << "MeshesWithOrigins is unspecified.";

  SetVector<StringRef> mesh_names;
  for (const auto& [mesh, origins] : *mesh_to_origins_) {
    mesh_names.insert(mesh);
  }

  if (include_wildcard_mesh && has_wildcard_mesh()) {
    mesh_names.insert(kWildcardMesh);
  }

  return mesh_names;
}

std::optional<SetVector<StringRef>> MeshesWithOrigins::MaybeMeshNames(
    bool include_wildcard_mesh) const {
  if (!has_meshes_specified()) {
    return std::nullopt;
  }
  return MeshNames(include_wildcard_mesh);
}

SetVector<StringRef> MeshesWithOrigins::MeshNamesOrEmpty(
    bool include_wildcard_mesh) const {
  if (!has_meshes_specified()) {
    return {};
  }
  return MeshNames(include_wildcard_mesh);
}

bool MeshesWithOrigins::HasSameMeshes(const MeshesWithOrigins& other) const {
  auto meshes = MeshNamesOrEmpty(/*include_wildcard_mesh=*/true);
  auto other_meshes = other.MeshNamesOrEmpty(/*include_wildcard_mesh=*/true);

  return meshes.size() == other_meshes.size() &&
         llvm::set_is_subset(meshes, other_meshes);
}

namespace {
// Returns true if the origin is considered low priority.
//
// This is used for picking the best mesh to assign.
bool LowPriorityOrigin(OriginAttr origin) {
  return origin.getOriginLabel() == kBroadcastInputOrigin;
}

StringRef GetHighestPriorityMeshName(MeshToOrigins mesh_to_origins) {
  SmallVector<StringRef> high_priority_mesh_names;
  SmallVector<StringRef> low_priority_mesh_names;
  for (const auto& [mesh, origins] : mesh_to_origins) {
    if (none_of(origins, LowPriorityOrigin)) {
      high_priority_mesh_names.push_back(mesh);
    } else {
      low_priority_mesh_names.push_back(mesh);
    }
  }

  if (!high_priority_mesh_names.empty()) {
    return *min_element(high_priority_mesh_names);
  }

  return *min_element(low_priority_mesh_names);
}
}  // namespace

std::optional<StringRef> MeshesWithOrigins::GetPrioritizedMeshName(
    const SetVector<StringRef>& preferred_mesh_names) const {
  SDY_CHECK(has_meshes_specified())
      << "GetPrioritizedMeshName is only allowed when meshes are specified.";
  if (empty()) {
    return std::nullopt;
  }

  // Get preferred meshes that are also in the current set.
  MeshToOrigins valid_preferred_meshes;
  for (const auto& mesh : preferred_mesh_names) {
    if (mesh_to_origins_->contains(mesh)) {
      valid_preferred_meshes.try_emplace(mesh, mesh_to_origins_->lookup(mesh));
    } else if (has_wildcard_mesh()) {
      valid_preferred_meshes.try_emplace(mesh, wildcard_origins_);
    }
  }

  if (!valid_preferred_meshes.empty()) {
    return GetHighestPriorityMeshName(valid_preferred_meshes);
  }

  return GetHighestPriorityMeshName(*mesh_to_origins_);
}

}  // namespace mlir::mpmd
