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

#ifndef MPMD_OPS
#define MPMD_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "shardy/dialect/mpmd/ir/attrs.td"
include "shardy/dialect/mpmd/ir/dialect.td"
include "shardy/dialect/mpmd/ir/enums.td"
include "shardy/dialect/mpmd/ir/types.td"
include "shardy/dialect/sdy/ir/op_interface.td"
include "shardy/dialect/sdy/ir/attrs.td"
include "stablehlo/dialect/Base.td"

// Base class for MPMD operations.
class Mpmd_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mpmd_Dialect, mnemonic, traits>;

// NOTE: any MPMD op is expected to live at the top-level of a function.
// If this changes in the future, then we need to revisit certain assumptions in
// the code base (e.g., in merge passes).

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Mpmd_Op<"return", [Pure, Terminator]> {
  let summary = [{
    The `mpmd.return` operation terminates the regions attached to mpmd
    region-based ops. It is variadic: it takes as arguments a list of values
    whose types can be any (but of the same kind, e.g. `AnyTensor`) and
    therefore can be reused at various levels of the MPMD IR stack.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "attr-dict $results (`:` type($results)^)?";

  let builders = [
    OpBuilder<(ins), [{ }]>,  // needed for ensureTerminator() during parsing.
  ];
}

//===----------------------------------------------------------------------===//
// NamedComputationOp
//===----------------------------------------------------------------------===//

def NamedComputationOp : Mpmd_Op<"named_computation", [
       SingleBlockImplicitTerminator<"ReturnOp">,
       RecursiveMemoryEffects, RecursivelySpeculatable, IsolatedFromAbove]> {
  let summary = "named scope operation";
  let description = [{
    Groups a computation, i.e. a block of operations, and gives it a name and
    a transpose count via the UserOrigin attribute. This NamedComputation can be
    used to assign a mesh to the computation in MPMD or for optimizations.

    The transpose count (default=0) denotes whether the named computation has
    been produced by a certain number of JAX AD transpose transformations.

    The op's region shouldn't have any free variables, and the type of
    each block arguments and returned values in the region must be the same as
    the type of the inputs and the return type of the op.
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrToken>:$tensors,
    Mpmd_UserOrigin:$origin
  );

  let results = (outs Variadic<HLO_TensorOrToken>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    StringRef getName() { return getOrigin().getUserName(); }
    int64_t getTransposeCount() { return getOrigin().getTransposeCount(); }
  }];

}

//===----------------------------------------------------------------------===//
// NamedTensorOp
//===----------------------------------------------------------------------===//

def NamedTensorOp : Mpmd_Op<"named_tensor", [Pure, SameOperandsAndResultType]> {
  let summary = "Assign a tensor to a mesh";
  let description = [{
    An identity op that associates the result of the tensor with a given name.
    This NamedTensor can be used to assign a mesh to the tensor in MPMD.

    NOTE: this is different than TagOp in that TagOp is used for naming a tensor
    and can be used to partition that tensor. NamedTensorOp is for MPMD programs
    for tensors that may be explicitly assigned to meshes.
  }];

  let arguments = (ins
    AnyTensor:$tensor,
    StrAttr:$name
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$tensor `name````=```$name attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// FragmentOp
//===----------------------------------------------------------------------===//

def FragmentOp : Mpmd_Op<"fragment", [
       SingleBlockImplicitTerminator<"ReturnOp">,
       RecursiveMemoryEffects, RecursivelySpeculatable, IsolatedFromAbove,
       ParentOneOf<["::mlir::func::FuncOp", "ForOp"]>,
       DeclareOpInterfaceMethods<Sdy_ShardableDataFlowOpInterface,
       /*methodOverrides=*/["setBlockArgumentEdgeOwnerSharding", "getBlockArgumentEdgeOwnerShardings",
       "setBlockArgumentEdgeOwnerShardings", "setOpResultEdgeOwnerShardings",
       "getBlockArgumentEdgeOwners", "getOpResultEdgeOwnerShardings",
       "getOpResultEdgeOwners", "getEdgeSources", "getEdgeOwnerFromTarget",
       "getEdgeOwnerFromSource", "shouldKeepEdgeOwnerShardingsDivisible"]>]> {
  let summary = "fragment operation";
  let description = [{
    Assigns a computation, i.e. a block of operations, to a specific mesh in an
    MPMD topology, that is intended to be executed as an individual SPMD program
    fragment.

    The fragment takes and returns only mesh tensors that are assigned to the
    same mesh as the fragment.

    The mesh name of the fragment should correspond to a mesh in the topology.

    The fragment includes a list of origins, i.e., metadata with information re
    the original named_computations that formed this fragment, and a staged_id
    defined _iff_ it is a user defined fragment, i.e., it has a non-empty list
    of origins. The optional in_shardings specifies the sharding of the
    block arguments of a fragment, which correspond to the operands.
    The optional out_shardings specifies the shardings of the results.

    The fragment's region shouldn't have any free variables, and the type of
    each block arguments and returned values in the region is the global tensor
    type of the corresponding mesh tensor.
  }];

  let arguments = (ins
    Variadic<Mpmd_MeshTensorType>:$inputs,
    TypedArrayAttrBase<Mpmd_UserOrigin, "array of origin infos">:$origin,
    StrAttr:$mesh_name,
    OptionalAttr<I64Attr>:$stage_id,
    OptionalAttr<Sdy_TensorShardingPerValue>:$in_shardings,
    OptionalAttr<Sdy_TensorShardingPerValue>:$out_shardings
  );
  let results = (outs Variadic<Mpmd_MeshTensorType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;


  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$results,
      "::mlir::ValueRange":$inputs,
      "::mlir::ArrayAttr":$origin,
      "::mlir::StringRef":$mesh_name,
      "::mlir::IntegerAttr":$stage_id),
      [{ build($_builder, $_state,
             results, inputs, origin, mesh_name,
             stage_id, /*in_shardings=*/nullptr, /*out_shardings=*/nullptr); }]>
  ];

  let extraClassDeclaration = [{

    // Returns true if the fragment originates (e.g. is a merge of) some user
    // named computation.
    bool isUserFragment() { return !getOrigin().empty(); }

    // Prints the metadata (e.g., origin, mesh name, stage) of the fragment.
    void printFragmentMetadata(llvm::raw_ostream& os);

    // Sets the sharding of an input of the fragment.
    void setInputSharding
    (unsigned input_index, sdy::TensorShardingAttr sharding);

    // Sets the sharding of a result of the fragment specified by the user.
    void setUserSpecifiedResultSharding
    (unsigned result_index, sdy::TensorShardingAttr sharding);

    // Creates a new FragmentOp with `tensors` as inputs (that must have
    // mesh types), returning `result_types` (that must be mesh types), and
    // a body populator `body_populator` that will populate the global region
    // of the fragment.
    static FragmentOp createMeshFragmentWithGlobalBody(
      Location loc,
      llvm::ArrayRef<Attribute> user_origin,
      llvm::StringRef mesh_name,
      ValueRange tensors,
      TypeRange result_types,
      OpBuilder& builder,
      FragmentOpBodyPopulator body_populator);

  }];
}

//===----------------------------------------------------------------------===//
// FragmentCallOp
//===----------------------------------------------------------------------===//

def FragmentCallOp : Mpmd_Op<"fragment_call",
    [HasParent<"::mlir::func::FuncOp">, CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "fragment call operation";
  let description = [{
    Represents a call to a function that holds an MPMD fragment body, i.e. a
    computation assigned to a specific mesh in an MPMD topology, that is
    intended to be executed as an individual SPMD program fragment.

    The mesh name of the fragment should correspond to a mesh in the topology of
    the enclosing function, and that mesh shape should match that of the callee.

    The origin specifies the user named computations that contributed to this
    fragment call e.g. through merging.

    The function input and result types of the callee must be the local tensor
    types of the corresponding mesh tensors of this op's operands and results
    respectively.

    Example:

    ```mlir
    %2 = mpmd.fragment_call<mesh="m1", origin=[]> @my_fragment(%0, %1) :
      (mesh_tensor<...>, mesh_tensor<...>) -> mesh_tensor<...>
    ```
  }];

  let arguments = (ins
    Variadic<Mpmd_MeshTensorType>:$tensors,
    TypedArrayAttrBase<Mpmd_UserOrigin, "array of origin infos">:$origin,
    StrAttr:$mesh_name,
    FlatSymbolRefAttr:$callee);
  let results = (outs Variadic<Mpmd_MeshTensorType>);

  let extraClassDeclaration = [{
    // Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return getTensorsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    // Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }
    // Set the callee from the callable.
    void setCalleeFromCallable(CallInterfaceCallable callee);

    Attribute removeArgAttrsAttr() { return nullptr; }
    Attribute removeResAttrsAttr() { return nullptr; }
    ArrayAttr getArgAttrsAttr() { return nullptr; }
    ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(ArrayAttr) { return; }
    void setResAttrsAttr(ArrayAttr) { return; }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

def TransferOp : Mpmd_Op<"transfer", [Pure, HasParent<"::mlir::func::FuncOp">, DeclareOpInterfaceMethods<Sdy_ShardingRuleOpInterface,
       /*methodOverrides=*/["shouldKeepOutputShardingsDivisible"]>]> {
  let summary = "transfer operation";
  let description = [{
    Transfers a distributed tensor from one mesh to another.

    The mesh names of the operand and result types should correspond to meshes
    in the topology, and their global types should be identical.
  }];

  let arguments = (ins Mpmd_MeshTensorType:$tensor);
  let results = (outs Mpmd_MeshTensorType:$result);

  let assemblyFormat = "attr-dict $tensor `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    // Returns true if this transfer is between two tensors of the same mesh.
    bool isIntraMesh();

    // Returns true if this transfer is between two tensors of a different mesh.
    bool isInterMesh() {
      return !isIntraMesh();
    }
  }];
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

def AssignOp : Mpmd_Op<"assign",
    [Pure, HasParent<"::mlir::func::FuncOp, ForOp">]> {
  let summary = "assign operation";
  let description = [{
    Assigns a local tensor to a mesh as fully replicated within that mesh.

    This is a temporary op that is introduced when lowering jax ops, to move
    from local types to mesh types. These ops will be eliminated during import,
    when the inputs and results of the func op become mesh tensors.

    The mesh name of the result type should correspond to a mesh in the
    topology, and its global type should be identical to the operand type.

    The origin of the assign op is the origin of mesh, e.g. named_computation,
    mesh inference, etc.
  }];

  let arguments = (
    ins AnyTensor:$tensor,
    OptionalAttr<StrAttr>:$origin  // TODO: b/396601755 - Make required.
  );
  let results = (outs Mpmd_MeshTensorType:$result);

  let assemblyFormat = "attr-dict $tensor `:` functional-type(operands, results)";
  let hasVerifier = 1;

  let builders = [
    // Standard builders but with optional origin, or origin as StringRef.
    OpBuilder<(ins "::mlir::Type":$result_type, "::mlir::Value":$tensor),
    [{ build($_builder, $_state, result_type, tensor,
             /*origin=*/(::mlir::StringAttr)nullptr); }]>,
    OpBuilder<(ins "::mlir::Type":$result_type, "::mlir::Value":$tensor,
               "::mlir::StringRef":$origin),
    [{ build($_builder, $_state, result_type, tensor,
             ::mlir::StringAttr::get($_builder.getContext(), origin)); }]>,

    // Builds an AssignOp whose result type is a MeshTensorType with a fully
    // replicated distributed tensor type.
    OpBuilder<(ins "::mlir::Value":$tensor, "::mlir::StringRef":$mesh_name,
                   "sdy::MeshAttr":$mesh),
    [{ build($_builder, $_state,
             MeshTensorType::getFullyReplicated($_builder.getContext(),
                mesh_name, mesh, cast<RankedTensorType>(tensor.getType())),
             tensor); }]>,
    OpBuilder<(ins "::mlir::Value":$tensor, "::mlir::StringRef":$mesh_name,
                   "sdy::MeshAttr":$mesh, "::mlir::StringRef":$origin),
    [{ build($_builder, $_state,
             MeshTensorType::getFullyReplicated($_builder.getContext(),
                mesh_name, mesh, cast<RankedTensorType>(tensor.getType())),
             tensor, origin); }]>,
  ];

  let extraClassDeclaration = [{
    StringRef getDestinationMeshName() {
      return getType().getMeshName();
    }
    MeshWithOriginsAttr getMeshWithOrigin() {
      return MeshWithOriginsAttr::get(getContext(),
              getDestinationMeshName(),
              getOriginAttr() ?
                OriginAttr::get(getContext(), getOriginAttr()) :
                ArrayRef<OriginAttr>());
    }
  }];
}

//===----------------------------------------------------------------------===//
// UnassignOp
//===----------------------------------------------------------------------===//

def UnassignOp : Mpmd_Op<"unassign",
    [Pure, HasParent<"::mlir::func::FuncOp, ForOp">, InferTensorType]> {
  let summary = "unassign operation";
  let description = [{
    Unassigns a fully replicated tensor from a mesh.

    This is a temporary op that is introduced when lowering jax ops, to move
    from local types to mesh types. These ops will be eliminated during import,
    when the inputs and results of the func op become mesh tensors.

    The mesh name of the operand type should correspond to a mesh in the
    topology, and its global type should be identical to the result type.
  }];

  let arguments = (
    ins Mpmd_MeshTensorType:$tensor,
    OptionalAttr<StrAttr>:$origin
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "attr-dict $tensor `:` functional-type(operands, results)";
  let hasVerifier = 1;

  let builders = [
    // Standard builders but with optional origin, or origin as StringRef.
    OpBuilder<(ins "::mlir::Value":$tensor),
    [{ build($_builder, $_state, tensor,
             /*origin=*/(::mlir::StringAttr)nullptr); }]>,
    OpBuilder<(ins "::mlir::Value":$tensor, "::mlir::StringRef":$origin),
    [{ build($_builder, $_state, tensor,
            ::mlir::StringAttr::get($_builder.getContext(), origin)); }]>,
  ];

  let extraClassDeclaration = [{
    StringRef getSourceMeshName() {
      return getTensor().getType().getMeshName();
    }
    MeshWithOriginsAttr getMeshWithOrigin() {
      return MeshWithOriginsAttr::get(getContext(), getSourceMeshName(),
              getOriginAttr() ?
                OriginAttr::get(getContext(), getOriginAttr()) :
                ArrayRef<OriginAttr>());
    }
  }];
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Mpmd_Op<"call",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, CallOpInterface]> {
  let summary = "MPMD specific call function";
  let description = [{
    A function call operation. Useful to wrap the body of loops in function
    declarations to reduce code size, for example.
  }];

  let arguments = (ins
    Variadic<LocalOrMeshTensor>:$tensors,
    FlatSymbolRefAttr:$callee);
  let results = (outs Variadic<LocalOrMeshTensor>);

  let extraClassDeclaration = [{
    // Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    MutableOperandRange getArgOperandsMutable() {
      return getTensorsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    // Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }
    // Set the callee from the callable.
    void setCalleeFromCallable(CallInterfaceCallable callee);

    Attribute removeArgAttrsAttr() { return nullptr; }
    Attribute removeResAttrsAttr() { return nullptr; }
    ArrayAttr getArgAttrsAttr() { return nullptr; }
    ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(ArrayAttr) { return; }
    void setResAttrsAttr(ArrayAttr) { return; }
  }];

  let assemblyFormat = [{
    $callee `(` $tensors `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

def ForOp : Mpmd_Op<"for",
    [HLO_PairwiseSameOperandAndResultType,
     RecursiveMemoryEffects, RecursivelySpeculatable,
     SingleBlockImplicitTerminator<"ReturnOp">,
     OpAsmOpInterface,
     LoopLikeOpInterface,
     DeclareOpInterfaceMethods<Sdy_ShardableDataFlowOpInterface,
     /*methodOverrides=*/ ["getNonEdgeOwnerTargets"]>]> {
  let summary = "For operator";
  let description = [{
    Returns the result of executing a body function for a fixed number of
    iterations, with the iteration index available in the body.

    An optional unroll factor, that must divide the number of iterations,
    can be specified to unroll the body of the op by that factor, i.e. for
    unroll factor N, the body is replicated to create N copies and the number of
    iterations is reduced by a factor of 1/N. Each copy except the first uses
    the results of the previous copy instead of the block arguments, and the
    iteration index is multiplied by the unroll factor and incremented after
    every copy.

    A for operator can accept and return any types, but the TypeID of these
    must be the same -- e.g. all tensor types or all MPMD mesh types etc. This
    allows us to use the op at various levels, sharing implementation and
    transformations.
  }];

  let arguments = (ins
    Variadic<AnyType>:$tensors,
    UI32Attr:$iterations,
    OptionalAttr<UI32Attr>:$unroll_factor
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getAsmBlockArgumentNames(Region &region,
                                  OpAsmSetValueNameFn setNameFn) {
      setNameFn(getIndexArg(), "index");
    }

    // Returns the index block argument, assuming it was already added.
    BlockArgument getIndexArg() {
      return *getRegion().args_rbegin();
    }

    // Creates a new ForOp with a block argument for each result type and the
    // iteration index as the last block argument, populates the block using the
    // given `body_populator` and adds a ReturnOp at the end of the block with
    // the values returned from `body_populator`.
    static ForOp create(Location loc,
                        ValueRange tensors,
                        uint32_t iterations,
                        OpBuilder& builder,
                        ForOpBodyPopulator body_populator,
                        uint32_t unroll_factor = 1);

    //===------------------------------------------------------------------===//
    // LoopLikeOpInterface methods
    //===------------------------------------------------------------------===//
    SmallVector<Region*> getLoopRegions() {
      return {&getRegion()};
    }
    std::optional<::llvm::SmallVector<Value>> getLoopInductionVars() {
      return llvm::SmallVector<Value>{getIndexArg()};
    }
    MutableArrayRef<OpOperand> getInitsMutable() {
      return getOperation()->getOpOperands();
    }
    Block::BlockArgListType getRegionIterArgs() {
      return getRegion().getArguments().drop_back();
    }
    std::optional<::llvm::MutableArrayRef<OpOperand>>
      getYieldedValuesMutable() {
      return getRegion().front().getTerminator()->getOpOperands();
    }
   std::optional<ResultRange> getLoopResults() {
      return getResults();
    }
  }];

}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

def BroadcastOp : Mpmd_Op<"broadcast",
    [SameOperandsAndResultType, Pure]> {
  let summary = "broadcast operation";
  let description = [{
    Allows for a tensor to be transferred (or replicated) in any mesh where it's
    used. Whenever transferred, the origin of the transfer is the current
    location of the operand.
  }];

  let arguments = (ins AnyTensor:$tensor);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "attr-dict $tensor `:` type($tensor)";
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

def ReduceOp : Mpmd_Op<"reduce",
    [Pure, SameOperandsAndResultType]> {
  let summary = "cross-mesh reduce operation";
  let description = [{
    Allows for a tensor to be reduced across different meshes, and then
    broadcast to wherever it needs to be used.
  }];

  let arguments = (ins
    Variadic<AnyTensor>:$tensors,
    DefaultValuedAttr<
      Mpmd_Reduction, "ReductionType::kNone">:$reduction
  );
  let results = (outs AnyTensor:$result);
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    ReductionType getReductionType() {
      return getReduction().getReductionType();
    }
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = "`` $reduction attr-dict $tensors `:` functional-type(operands, results)";
}

#endif
