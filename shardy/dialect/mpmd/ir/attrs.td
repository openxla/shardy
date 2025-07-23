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

#ifndef MPMD_ATTRS
#define MPMD_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "shardy/dialect/mpmd/ir/dialect.td"
include "shardy/dialect/mpmd/ir/enums.td"

def Mpmd_NamedMesh : AttrDef<Mpmd_Dialect, "NamedMesh"> {
  let mnemonic = "named_mesh";
  let summary = "A pair with a name and a Mesh.";
  let parameters = (ins
      StringRefParameter<"name">:$name,
      AttrParameter<"sdy::MeshAttr", "mesh">:$mesh
  );
  let assemblyFormat = "`<` $name `:` $mesh `>`";
}

// A topology of SPMD meshes. This is used for MPMD partition where a program
// is composed of many programs, each running in a different mesh.
def Mpmd_Topology : AttrDef<Mpmd_Dialect, "Topology"> {
  let mnemonic = "topology";
  let summary = "Topology of named meshes.";
  let parameters = (ins
    ArrayRefParameter<"NamedMeshAttr", "topology meshes">:$meshes
  );
  // TODO(b/425894364): consider combining the assemblyFormat with custom
  // parse/print methods to remove unnecessary whitespaces.
  let assemblyFormat = "`<` $meshes `>`";
  let genVerifyDecl = 1;
}

// Attribute specifying the origin of named computations.
def Mpmd_UserOrigin : AttrDef<Mpmd_Dialect, "UserOrigin"> {
  let mnemonic = "user_origin";
  let summary = "Origin of user-specified computation.";
  let parameters = (ins "::mlir::StringAttr":$userName,
                        "int64_t":$transposeCount);
  let assemblyFormat = "`<` $userName `` custom<OptionalTransposeCount>($transposeCount) `>`";

  let extraClassDeclaration = [{
    void printShort(llvm::raw_ostream& os);
    void printShort(AsmPrinter& printer);
    static mlir::Attribute parseShort(AsmParser& parser);
  }];
}

// TODO: b/415235792 - Properly implement this and use this for the Origins in
// the AssignOp and UnassignOp.
//
// The plan is to store the debugging information in this Attr.
def Mpmd_Origin : AttrDef<Mpmd_Dialect, "Origin"> {
  let mnemonic = "origin";
  let summary = "Origin of mesh assignment.";
  let description = [{
    The origin of a mesh assignment.

    `origin_label` is a human-readable label for the origin.
    It is intended to be used for debugging purposes.
  }];

  let parameters = (ins
    StringRefParameter<"origin_label">:$origin_label
  );

  let assemblyFormat = "`` $origin_label";
}

def Mpmd_MeshWithOrigins : AttrDef<Mpmd_Dialect, "MeshWithOrigins"> {
  let mnemonic = "mesh_with_origins";
  let summary = "Mesh with its origins.";

  let parameters = (ins
    StringRefParameter<"mesh_name">:$mesh_name,
    OptionalArrayRefParameter<"OriginAttr", "origins">:$origins
  );

  let assemblyFormat = "`` $mesh_name (```[``` $origins^ `]`)?";

  let builders = [
    AttrBuilder<(ins
      "::mlir::StringRef":$mesh_name
    ), [{
      return $_get($_ctxt, mesh_name,
        /*origin=*/(mlir::ArrayRef<OriginAttr>){});
    }]>,
  ];
}

def Mpmd_MeshesWithOrigins : ArrayOfAttr<Mpmd_Dialect,
                                 "MeshesWithOrigins", "meshes_with_origins",
                                 "MeshWithOriginsAttr"> {
  let summary = "A list of meshes with their origins.";
  let assemblyFormat = "`<` (`>`) : ($value^ `>`)?";
}

def Mpmd_Reduction : AttrDef<Mpmd_Dialect, "Reduction"> {
  let mnemonic = "reduction";
  let summary = "Denotes a reduction.";
  let parameters = (ins EnumParameter<Mpmd_ReductionType>:$reduction_type);
  let constBuilderCall = "ReductionAttr::get($_builder.getContext(), $0)";
  let assemblyFormat = "`<` $reduction_type `>`";
}

#endif  // MPMD_ATTRS
