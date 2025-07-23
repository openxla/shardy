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

#ifndef MPMD_TYPES
#define MPMD_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "shardy/dialect/mpmd/ir/dialect.td"

// Base class for MPMD types.
class Mpmd_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Mpmd_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Mpmd_MeshTensorType : Mpmd_Type<"MeshTensor", "mesh_tensor",
  [DeclareTypeInterfaceMethods<ShapedTypeInterface>]> {
  let summary = "mesh tensor type";
  let description = [{
    Assigns a RankedTensorType to a specific SPMD mesh in the program's
    MPMD topology of meshes.
    The type holds an optional sharding that specifies how the tensor is
    sharded w.r.t to the SPMD mesh.
    If the sharding is not present the tensor is fully replicated.
  }];

  let parameters = (ins
    StringRefParameter<"mesh name">:$mesh_name,
    TypeParameter<"::mlir::RankedTensorType", "ranked tensor type">:$ranked_tensor_type,
    OptionalParameter<"::mlir::sdy::TensorShardingAttr">:$sharding,
    OptionalParameter<"::mlir::StringAttr">:$memory_kind
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilder<(ins
      "::llvm::StringRef":$mesh_name,
      "::mlir::RankedTensorType":$ranked_tensor_type),
    [{
      return $_get($_ctxt, mesh_name, ranked_tensor_type,
                   /*sharding=*/nullptr, /*memory_kind=*/nullptr);
    }]>,

    TypeBuilder<(ins
      "::llvm::StringRef":$mesh_name,
      "::mlir::RankedTensorType":$ranked_tensor_type,
      "::mlir::sdy::TensorShardingAttr":$sharding),
    [{
      return $_get($_ctxt, mesh_name, ranked_tensor_type,
                   /*sharding=*/sharding, /*memory_kind=*/nullptr);
    }]>,

    TypeBuilder<(ins
      "::llvm::StringRef":$mesh_name,
      "::mlir::RankedTensorType":$ranked_tensor_type,
      "::mlir::StringAttr":$memory_kind),
    [{
      return $_get($_ctxt, mesh_name, ranked_tensor_type,
                   /*sharding=*/nullptr, /*memory_kind=*/memory_kind);
    }]>
  ];

  let extraClassDeclaration = [{
    // Verifies the MeshTensorType in the context of a topology, by locating the
    // TopologyAttr in the main FuncOp via `op`.
    LogicalResult verifyForTopology(Operation* op);
    // Verifies the MeshTensorType in the context of the given `mesh`.
    LogicalResult verifyForMesh(sdy::MeshAttr mesh, Operation* op);

    // Returns the local tensor type of the MeshTensorType wrt the given `mesh`.
    RankedTensorType getLocalTensorType(sdy::MeshAttr sdy_mesh);

    // Returns the local tensor type of the MeshTensorType wrt
    // the mesh (of the op) and sharding specified.
    // Assumes that the sharding is valid wrt the mesh and tensor type.
    RankedTensorType getLocalTensorType(Operation* op);

    RankedTensorType getGlobalTensorType() {
      return getRankedTensorType();
    }

    // Builds a MeshTensorType with a fully replicated sharding,
    // whose local type is `local_type`.
    static MeshTensorType getFullyReplicated(
        MLIRContext* ctx, StringRef mesh_name, sdy::MeshAttr mesh,
        RankedTensorType local_type);

    // Returns a MeshTensorType with only the sharding replaced.
    MeshTensorType replaceSharding(sdy::TensorShardingAttr sharding);

    // Returns true if the MeshTensorType is fully replicated.
    bool isFullyReplicated() {
      sdy::TensorShardingAttr sharding = getSharding();
      return !sharding || sharding.isFullyReplicated();
    }

    // Returns true if the MeshTensorType is on the host.
    bool isOnHost();
  }];
}

def LocalOrMeshTensor : AnyTypeOf<[AnyTensor, Mpmd_MeshTensorType]>;

#endif
