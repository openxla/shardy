/* Copyright 2024 The Shardy Authors.

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

#ifndef SDY_OPS
#define SDY_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "shardy/dialect/sdy/ir/attrs.td"
include "shardy/dialect/sdy/ir/dialect.td"
include "shardy/dialect/sdy/ir/enums.td"
include "shardy/dialect/sdy/ir/op_interface.td"

class Sdy_Op<string mnemonic, list<Trait> traits = []> :
   Op<Sdy_Dialect, mnemonic, traits>;

def Sdy_MeshOp : Sdy_Op<"mesh", [Symbol, HasParent<"ModuleOp">]> {
  let summary = "Named mesh";
  let description = [{
    Defines a new named mesh. All meshes in a module must have the same number
    of devices (except for meshes with a single device_id).
    The mesh is a `Symbol` operation that appears in the module's
    `SymbolTable` and can be referenced by its `name`.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    Sdy_Mesh:$mesh);

  let assemblyFormat = "$sym_name `=` $mesh attr-dict";
  let hasVerifier = 1;
}

def Sdy_ShardingConstraintOp : Sdy_Op<"sharding_constraint",
  [Elementwise, SameOperandsAndResultType, InferTypeOpInterface]> {
  let summary = "Constrains a tensor to the specified sharding";
  let description = [{
    Attaches a sharding to an intermediate tensor (e.g. the result of a matmul)
    to indicate that this is how that tensor, or a subset of its uses, should be
    sharded.

    If the sharding has open dimensions and unconstraint axes, it means the
    tensor can be further sharded along the open dimensions.

    This op can either:
    - Have no uses (dangling) - which means the attached sharding is how the
      input tensor itself should be sharded.
    - Have uses - which means the attached sharding is how the uses of the
      sharding constraint op should be sharded, while other uses of the input
      tensor might have a different sharding (if the input tensor has no other
      uses then the behavior is the same as the no uses case).
  }];

  let arguments = (ins
    AnyTensor:$input,
    Sdy_TensorSharding:$sharding);

  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input $sharding attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def Sdy_ReshardOp : Sdy_Op<"reshard",
  [Pure, Elementwise, SameOperandsAndResultType, InferTypeOpInterface]> {
  let summary = "Reshards a tensor to a different sharding";
  let description = [{
    Reshards the input tensor with the specified sharding, which is different
    from the input tensor's existing sharding.

    Both ShardingConstraintOp and ReshardOp attach a sharding to a tensor. Their
    lifespan is:
    1. Before sharding propagation, ShardingConstraintOp is added by users.
    2. Sharding propagation consumes ShardingConstraintOp. There is no
       ShardingConstraintOp in the results of sharding propagation. Instead,
       ReshardOp may be added if needed.
    3. A partitioner converts a ReshardOp into a collective op (or an identity
       op). There should be no ReshardOp in the results of the partitioner.

  // TODO(b/331680067). Add a canonicalization pattern to remove redundant
  // reshard ops.
  }];

  let arguments = (ins
    AnyTensor:$input,
    Sdy_TensorSharding:$sharding);

  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$input $sharding attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Sdy_ReturnOp : Sdy_Op<"return", [Pure, Terminator]> {
  let summary = [{
    The `sdy.return` operation terminates the regions attached to
    `sdy` region-based ops and any other Shardy region-based ops. It is
    variadic: it takes as arguments a list of values whose types can be any (but
    of the same kind, e.g. `AnyTensor`) and therefore can be reused at various
    levels of the Shardy IR stack.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

def Sdy_ManualComputationOp : Sdy_Op<"manual_computation",
  [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"ReturnOp">,
   IsolatedFromAbove, DeclareOpInterfaceMethods<
       ShardableDataFlowOpInterface,
       /*methodOverrides=*/["getOpResultEdgeOwnerShardings",
                            "setOpResultEdgeOwnerShardings",
                            "transformTargetSharding",
                            "getBlockArgumentEdgeOwners",
                            "setBlockArgumentEdgeOwnerShardings",
                            "getBlockArgumentEdgeOwnerShardings"]>]> {
  let summary = "Multi-device parallelism operation with manual collectives";
  let description = [{
    Jump into a region written in terms of per-device local code with explicit
    collectives, where logical shapes match local per-device physical buffer
    shapes and collectives correspond exactly to physical cross-device
    communication.

    The body is local wrt the manual_axes. Propagation will occur through
    the body on any free axes - those not in the manual_axes list.

    **Constraints:**
    - Elements in `in_shardings` and `out_shardings` must satisfy the constraints listed in `TensorShardingAttr`.
    - The number of global and local tensor inputs/outputs of the op region must match.
    - The manual axes must come before any free axes in each dim sharding.
    - The global and local shapes of the op regions arguments/results must match.
    - No manual axes are split.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$tensors,
    Sdy_TensorShardingPerValue:$in_shardings,
    Sdy_TensorShardingPerValue:$out_shardings,
    Sdy_ManualAxes:$manual_axes
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `(`operands`)`
    `in_shardings````=```custom<StrippedTensorShardingPerValueAttr>($in_shardings)
    `out_shardings````=```custom<StrippedTensorShardingPerValueAttr>($out_shardings)
    `manual_axes````=```$manual_axes
    custom<SingleBlockRegionNoBlockId>($body)
    attr-dict
    `:`
    functional-type(operands, results)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    TensorShardingAttr getInSharding(int64_t operandIndex) {
      return getInShardings().getSharding(operandIndex);
    }
    TensorShardingAttr getOutSharding(int64_t resultIndex) {
      return getOutShardings().getSharding(resultIndex);
    }

    void setInShardings(ArrayRef<TensorShardingAttr> shardings);
    void setOutShardings(ArrayRef<TensorShardingAttr> shardings);

    void setInSharding(int64_t operandIndex, TensorShardingAttr sharding);
    void setOutSharding(int64_t resultIndex, TensorShardingAttr sharding);

    // Same as `getInSharding`, but removes any prefixed manual axes from each
    // dim sharding.
    TensorShardingAttr getInShardingWithoutManualAxes(int64_t operandIndex);

    // Same as `getOutSharding`, but removes any prefixed manual axes from each
    // dim sharding.
    TensorShardingAttr getOutShardingWithoutManualAxes(int64_t resultIndex);
  }];
}

def Sdy_ShardingGroupOp : Sdy_Op<"sharding_group",
  // Op is non-pure since it modifies the internal representation of the
  // sharding group.
  [DeclareOpInterfaceMethods<InferTypeOpInterface>]>{
  let summary = "Constrains tensors in the group to have the same sharding.";
  let description = [{
    This op provides an interface to assign tensors to sharding groups (
    groups of tensors that will be enforced to have identical shardings).
    During propagation, as soon as one group element is sharded, all other
    members will be sharded in exactly the same way. This operation takes the
    argument group ID and returns no result, but instead modifies the internal
    sharding group representation to add the input tensor to the group with the
    given ID.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$group_id);

  // Dangling op has no results.
  let results = (outs);

  let assemblyFormat = "$input `group_id````=```$group_id attr-dict `:` type($input)";
  let hasCanonicalizer = 1;
}

def Sdy_ConstantOp : Sdy_Op<"constant",
  [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    See:
    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#constant

    NOTE: SDY defines its own constant op that isn't ConstantLike and doesn't
    have a folder, so that we'll be able to duplicate constants without any
    greedy pattern rewriter folding them back into a single constant. In this
    way, constants can be sharded differently for every use, and no propagation
    is done between constants (or constant expressions).

    Example:
    ```mlir
    %output = sdy.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    ```
  }];
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    AnyStaticShapeTensor:$output
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(TypeRange l, TypeRange r);
  }];
}

//===----------------------------------------------------------------------===//
// DataFlowEdgeOp
//===----------------------------------------------------------------------===//

// TODO(tomnatan): consider moving this to an sdy_internal dialect
// TODO((b/330339693): update doc based on the interface we define with dialects
// like stablehlo.
def Sdy_DataFlowEdgeOp : Sdy_Op<"data_flow_edge",
  [SameOperandsAndResultType]> {
  let summary = "data flow edge op.";

  let description = [{
    A data flow edge of some op X defines a bridge between a set of sources
    (each is either an operand of X or an operand of X's block terminator) and
    a set of targets (each is either a result of X or a block argument of X),
    such that all sources and targets should be sharded in the same way.

    An op can have multiple data flow edges that are orthogonal to one another.

    For example:

    ```mlir
      y_0, ..., y_n = while (x_0, ..., x_n)
                      ((pred_arg_0,... , pred_arg_n) { ... })
                      ((body_arg_0,..., body_arg_n) {
                        ...
                        return return_value_0, ..., return_value_n
                      })
    ```

    This while op has n data flow edges, the i-th data flow edges is between
    sources `x_i`, `return_value_i` and targets `y_i`, `pred_arg_i`,
    `body_arg_i`.

    An `sdy.data_flow_edge` takes as input the root target of an edge (can be
    any of the targets, but preferably an op result rather than a block
    argument), which shouldn't have any other uses. This op isn't pure because
    it can take an input that originally didn't have any uses.

    The `sdy.data_flow_edge` also holds an optional sharding for all targets of
    the edge, and that sharding should be updated instead of the targets'
    sharding (if can be attached) during propagation. This is useful when an op
    has many edges, as it's much more efficient to:
    - propagate through each edge separately.
    - update the sharding of each edge separately instead of all targets at once
      (e.g. an op has a single immutable `TensorShardingPerValueAttr` for result
      shardings).
    - add each edge to the worklist separately when the sharding of a source has
      changed.

    Propagation will propagate shardings between all sources and targets of a
    `sdy.data_flow_edge` as if it was a regular op with the sources as operands
    and targets as results, and an identity `sdy.op_sharding_rule`. That means
    that forward propagation is from sources to targets and backwards
    propagation is from targets to sources.

    We don't allow the input of a `sdy.data_flow_edge` to be defined by an
    `SdyDialect` op, so we can assume that it's defined by an op that has
    unregistered `sdy.sharding` attribute.

    NOTE: it's NOT the responsibility of the `sdy.data_flow_edge` to link
    between sources and targets, it's simply attached to the root target of the
    edge. The op that this edge is bound to (while in the example above) is
    responsible for providing this information.
  }];

  let arguments = (ins
    AnyShaped:$input,
    OptionalAttr<Sdy_TensorSharding>:$sharding);

  let results = (outs AnyShaped:$result);

  let assemblyFormat = "$input (`sharding````=``` $sharding^)? attr-dict `:` type($result)";

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input),
    [{ build($_builder, $_state, input, /*sharding=*/nullptr); }]>
  ];

  let extraClassDeclaration = [{
    // If `root` has a single use which is by a `DataFlowEdgeOp`, returns that
    // `DataFlowEdgeOp`, otherwise returns `nullptr`.
    static DataFlowEdgeOp getDataFlowEdgeUser(Value root);
  }];
}

//===----------------------------------------------------------------------===//
// PropagationBarrierOp
//===----------------------------------------------------------------------===//

def Sdy_PropagationBarrierOp : Sdy_Op<"propagation_barrier",
  [Pure, Elementwise, SameOperandsAndResultType,
   DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Propagation barrier operation";

  let description = [{
    This op operates like an identity op, outputting the same value it took as
    input. But in terms of propagation, this will only allow propagation to flow
    through it in a certain direction.

    This prevents shardings from being propagated between the uses of the result
    of the barrier op and its operand.

    - `FORWARD` means shardings can only flow from the operand to the result.
    - `BACKWARD` means shardings can only flow from the result to the operand.
    - `NONE` means no sharding can propagate through this op.
    - Cannot specify `BOTH`, as this op would be redundant.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Sdy_PropagationDirection:$allowed_direction
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input `allowed_direction````=```$allowed_direction attr-dict `:` type($input)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// NamedComputationOp
//===----------------------------------------------------------------------===//

def Sdy_NamedComputationOp : Sdy_Op<"named_computation",
    [RecursiveMemoryEffects, SingleBlockImplicitTerminator<"ReturnOp">,
     RecursivelySpeculatable, IsolatedFromAbove,
     DeclareOpInterfaceMethods<InferTypeOpInterface>,
     DeclareOpInterfaceMethods<
         ShardableDataFlowOpInterface,
         /*methodOverrides=*/["getOpResultEdgeOwnerShardings",
                              "setOpResultEdgeOwnerShardings",
                              "getBlockArgumentEdgeOwners",
                              "setBlockArgumentEdgeOwnerShardings",
                              "getBlockArgumentEdgeOwnerShardings"]>]> {
  let summary = "named computation operation";
  let description = [{
    Groups a computation, i.e. a block of operations, and gives it a name.
    Propagation will flow in/out of the region as if everything was inlined.

    This can be used to handle propagating through call instructions to other
    functions. Any users of Shardy should write an import/export pass that
    converts their call ops to `sdy.named_computation` ops, duplicating/copying
    the body of the called function into the body of the `named_computation`.

    The type of each block arguments and returned values in the region must be
    the same as the type of the operands and results type of the op.

    Example:

    ```mlir
    %1 = sdy.named_computation<"foo">(%0) (%arg1: tensor<16x32xf32>) {
      sdy.return %arg1 : tensor<16x32xf32>
    } : (tensor<16x32xf32>) -> tensor<16x32xf32>
    ```
  }];

  let arguments = (ins
    StrAttr:$name,
    Variadic<AnyType>:$operands,
    OptionalAttr<Sdy_TensorShardingPerValue>:$in_shardings,
    OptionalAttr<Sdy_TensorShardingPerValue>:$out_shardings
  );
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<`$name`>` `` `(` $operands `)`
    (`in_shardings````=```custom<StrippedTensorShardingPerValueAttr>($in_shardings)^)?
    (`out_shardings````=```custom<StrippedTensorShardingPerValueAttr>($out_shardings)^)?
    custom<SingleBlockRegionNoBlockId>($body)
    attr-dict
    `:` functional-type($operands, results)
  }];

  let hasVerifier = 1;
}


def Sdy_AllGatherOp : Sdy_Op<"all_gather",
    [SameOperandsAndResultType,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Gathers chunks of a tensor along axes";
  let description = [{
    Gathers chunks of a tensor along axes specified in `gatheringAxes`.

    The `gatheringAxes` is a list of lists of axes. Each inner list specifies
    the axes along which a separate gather should be performed. The outer list
    is over the dimensions of the tensor. It will be applied to the sharding of
    the operand (`tensor`) to obtain the sharding of the result (`outSharding`).

    Note that `outSharding` is not used to determine the sharding of the result.
    Instead, the sharding of the result is determined by the sharding of the
    operand and the `gatheringAxes`, and `outSharding` must match this inferred
    sharding.

    Example:
    ```mlir
    %1 = stablehlo.tanh(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b", "c"}, {}, {"d"}\]>]>} : tensor<8x8xf32>
    %2 = sdy.all_gather gathering_axes=[{"b", "c"}, {}, {"d"}\] %1 to_sharding=<@mesh, [{"a"}, {}, {}\]> : tensor<8x8xf32>
    ```

    **Constraints:**
    - Elements in `gatheringAxes` must satisfy the constraints listed in `AxisRefListAttr`.
    - Both operand and result shardings should be bound to the same `MeshAttr`.
  }];

  let arguments = (ins
    AnyTensor:$tensor,
    Sdy_ListOfAxisRefLists:$gatheringAxes,
    Sdy_TensorSharding:$outSharding
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$gatheringAxes $tensor `out_sharding````=```$outSharding attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}


#endif  // SDY_OPS
