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

include "mlir/Pass/PassBase.td"

def Mpmd_UserAssignmentMapOption :
    Option<"assignment", "assignment", "UserAssignmentMapOption",
           /*default=*/"UserAssignmentMapOption()",
           "Mapping between names (of computations and tensors) and mesh "
           "names, and optionally stage ids. E.g., 'n0@m0,n1@m1' defines that "
           "names n0 and n1 will be assigned to meshes m0 and m1, "
           "respectively. Alternatively 'n0@m0/0,n1@m1/1' means that these "
           "names are also assigned to the stages 0 and 1.">;

def Mpmd_InputOutputEquishardingConstraintsOption :
      ListOption<"constraints", "constraints",
                 "InputOutputEquishardingConstraint",
                 "A list of constraint, each enforcing that an input and "
                 "output should be assigned to the same mesh.">;

def Mpmd_InferTransfersOption :
    Option<"inferTransfers", "infer-transfers", "bool", /*default=*/"false",
           "Whether to create transfers when needed, instead of erroring.">;

def Mpmd_ErrorLimitOption :
    Option<"errorLimit", "error-limit", "int", /*default=*/"5",
           "The number of errors to emit. Set to -1 to emit all errors. Cannot "
           "be 0.">;

def CopyTopologyFromMainPass : Pass<"mpmd-copy-topology-from-main", "ModuleOp"> {
  let summary = "Copies the topology from the main function to functions "
                "referred by mpmd.call.";
  let description = [{
    Copies the topology attribute from the main function to any function
    referred by a mpmd.call. This also sets the mpmd.call callee to private
    visibility, to avoid being mistaken for an entry point function.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

// TODO(jupvfranco): Rename to EnforceEquiassignmentPass.
def EnforceEquishardingPass :
    Pass<"mpmd-enforce-input-output-equisharding", "func::FuncOp"> {
  let summary = "Enforces equisharding constraints for MPMD functions.";
  let description = [{
    Enforces input-output equisharding constraints for MPMD functions by
    introducing TransferOps when necessary.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Mpmd_InputOutputEquishardingConstraintsOption
  ];
}

def MapNamedOpsToMpmdOpsPass :
    PassBase<"mpmd-map-named-ops-to-mpmd-ops", "DistributedFunctionPass"> {
  let summary = "Assigns meshes to user defined operations.";
  let description = [{
    Creates a pass optionally assigning mpmd.named_tensor to
    Assign(Unassign(%v)) (depends if there is an entry in `assignment`), and to
    map each named_computation to a mesh, using a user-defined mapping between
    named_computations and mesh names. This means replacing each
    named_computation with a Fragment and creating AssignOps for the operands
    and UnassignOps for the results of these Fragments. The now introduced
    pattern Assign(Unassign(%v)) is rewritten into a Transfer(%v).
    No named_computation/named_tensor ops will exist after this pass.

    Requires: all named_computations and named_tensors to live at the top-level
    of the function.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Mpmd_UserAssignmentMapOption
  ];
}

def MapInputOutputToMeshPass :
    Pass<"mpmd-map-input-output-to-mesh", "ModuleOp"> {
  let summary = "Assigns meshes to function inputs and output.";
  let description = [{
    Creates a pass that maps function inputs/outputs to meshes given a
    user-defined mesh assignment.

    For the input arguments, this pass:
    1. Casts the input tensors that should be put on a mesh to a mesh
       tensor.
    2. Updates the function signature.
    3. Adds mpmd.unassign before the tensor is used.

    For the output arguments, this pass adds mpmd.assign before the tensor
    is returned and updates the function signature.

    Requires: Each input/output index is valid and each mapped mesh
    is a valid mesh in the topology.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Option<"inputAssignment", "input-assignment", "IndexedAssignmentMapOption",
           /*default=*/"IndexedAssignmentMapOption()",
           "Mapping between function input indices and assigned mesh names."
           "E.g., '0@m0,1@m1' defines that input with index 0 will be assigned "
           "to mesh m0 and input with index 1 will be assigned to mesh m1.">,
    Option<"outputAssignment", "output-assignment", "IndexedAssignmentMapOption",
           /*default=*/"IndexedAssignmentMapOption()",
           "Mapping between function output indices and assigned mesh names."
           "E.g., '0@m0,1@m1' defines that output with index 0 will be "
           "assigned to mesh m0 and output with index 1 will be assigned to "
           "mesh m1.">,
  ];
}

def InlineNestedUserExposedOpsPass :
    PassBase<"mpmd-inline-nested-user-exposed-ops", "DistributedFunctionPass"> {
  let summary = "Inlines any user-exposed mpmd op nested in a named_computation.";
  let description = [{
    Inlines any named_computation, named_tensor, broadcast and reduce op that is
    nested in a named_computation, checking that its mesh assignment (when
    defined) matches that of the parent.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Mpmd_UserAssignmentMapOption
  ];
}

def IntroduceTransfersPass : Pass<"mpmd-introduce-transfers", "ModuleOp"> {
  let summary = "Creates data transfers based on user mesh assignments.";
  let description = [{
    Creates a pass that introduces transfer operations based on user mesh
    assignments. This includes:

    1. Push in UnassignOp to mpmd calls if the result of UnassignOp is later
       assigned in the callee.
    2. Replaces the AssignOp of an UnassignOp with a TransferOp.
    3. Assign the addition to the consuming mesh and introduce a transfer if
       there is a meshless addition between fragments.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];
}

def InsertNamelessCloneOfNeglibleOpsPass :
    PassBase<"mpmd-insert-nameless-clone-of-negligible-ops",
             "DistributedFunctionPass"> {
  let summary = "Clones negligible ops outside of named computations.";
  let description = [{
    Clones negligible operations, i.e., single result, zero operand operations,
    outside named computations whenever they are used by the computation's
    return op, replacing the named_computation's result with the clone. This is
    needed because if such results are used by named computations assigned to
    different meshes, this could cause a mesh inference conflict. By applying
    this pass, we allow mesh inference to clone these negligible ops.

    This pass does NOT change the named computation at all.
  }];
}

// TODO: b/430024003 - Replace with simpler dedup and dce passes.
def SimplifyNamedComputationsPass :
    PassBase<"mpmd-simplify-named-computations", "DistributedFunctionPass"> {
  let summary = "Simplifies the inputs and outputs of named computation ops.";
  let description = [{
    Simplifies each named computation independently. In particular, it:
    - deduplicates results, and their corresponding return values;
    - deduplicates operands, and their corresponding block arguments;
    - removes results whose corresponding return operand is a block
      argument of the op;
    - removes operands whose corresponding block argument has no more uses (or
      didn't have any to begin with); and
    - removes results that are unused.
    - replaces the pattern `arg -> stablehlo.optimization_barrier -> return`
      within a named computation with the pattern `arg -> return`, allowing
      further simplification.
  }];
}

def ValidateNamedOpsInMpmdFuncPass :
    Pass<"mpmd-validate-named-ops-in-mpmd-func", "func::FuncOp"> {
  let summary = "Validates that named ops are only nested in mpmd functions.";
  let description = [{
    Validates that NamedComputationOp and NamedTensorOp are only nested in
    mpmd functions, i.e., functions with a topology attr.
  }];
}

//===----------------------------------------------------------------------===//
// Start of - Infer Mesh assignment passes
//===----------------------------------------------------------------------===//

def InferMeshPopulateUseSetPass :
    Pass<"mpmd-infer-mesh-populate-use-set", "ModuleOp"> {
  let summary = "Initializes the use_set for AssignOps and propagates the use_set.";
  let description = [{
    This pass initializes the use_set and propagates it backwards, populating the
    graph with the use_set info.

    Initialization: the use_set of an AssignOp is set to the mesh that it
    assigns to.

    Propagation: use_sets propagate backwards from users to the op itself, taking
    the union of users. An op's use_set is the union of its users' use_sets by
    definition, since the use_set is the set of transitive uses.
  }];
}

def InferMeshPopulateSrcSetPass :
    Pass<"mpmd-infer-mesh-populate-src-set", "ModuleOp"> {
  let summary = "Initializes the src_set for UnassignOps and func args and "
                "propagates the src_set.";
  let description = [{
    This pass initializes the src_set and propagates it, populating the graph
    with the src_set info.

    Pre-condition: for func args to have src_sets, the use_set must be populated.

    Initialization: the src_set of an UnassignOp is set to the mesh that it
    assigns to. The src_set of a func arg is set to its use_set.

    Propagation: src_sets propagate forwards from operands to the op itself,
    taking the intersection of operands. See `PropagateSrcSet` for details.
  }];
}

def InferMeshAssignUsingInputOutputConstraintsPass :
    Pass<"mpmd-infer-mesh-assign-using-input-output-constraints", "ModuleOp"> {
  let summary = "Assigns a mesh to the inputs and outputs according to the "
                "input-output assignment constraints.";
  let description = [{
    This pass uses the input-output equi-assignment constraints to assign both
    the input and output to the same mesh.

    Note that this guarantees the input is on the same mesh, regardless of
    whether it is subsequently transferred to other meshes. But this means that
    we should not run `populate-src-set` after this pass.

    *Requires:*
      * for any input `i` of the entry-point function that may be part of an
      equi-assignment constraint: `i` has a MeshTensorType, OR a well-defined
      use-set.
      * for any output `o` of the entry-point function that may be part of an
      equi-assignment constraint: `o` has type MeshTensorType, OR well-defined
      src- and use- sets.

    Where a well-defined use-set of a value includes all the meshes the value is
    (transitively) assigned to, via mpmd.assign ops and no other mesh. A
    well-defined src-set includes all the meshes where the tensor is allowed to
    live and no other mesh..

    Although this only runs on entry point functions, we make this a module op
    pass because this requires all passes on existing functions to complete
    before it runs. E.g. if we make this an `EntryPointFunctionPass` then the
    pass manager might run this pass before validation completes on the non-entry
    point functions.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Option<"verboseLogging", "verbose-logging", "bool", /*default=*/"false",
           "Whether to enable verbose logging">,
    Mpmd_InputOutputEquishardingConstraintsOption
  ];
}

def InferMeshAssignMeshForFuncLeavesPass :
    Pass<"mpmd-infer-mesh-assign-mesh-func-leaves", "ModuleOp"> {
  let summary = "Assigns a mesh to each unused computation, function output, "
                "and function input using the use_set and src_set analysis.";
  let description = [{
    This pass assigns meshes to the func body leaves (i.e., the results of unused
    computations, unused function arguments, and function outputs) by creating
    AssignOps or changing the type, using the use_set and src_set information.

    We also treat certain intermediate values as leaves, for the sake of
    analysis. Namely: the operands of an mpmd.reduce and mpmd.broadcast are
    treated as leaves and an assign-unassign pair will be created on them.

    This assignment will clear the use_set of all non-leaf ops, as the previously
    annotated uses will be stale since inferring reduce ops will change the
    use_set of some values: the initial use_set propagation is unaware of reduce
    ops, but now that we've inferred reduce ops, the propagation will be
    different.

    This pass will fail, emitting errors, if use- and src-sets aren't correctly
    populated for leaf ops.

    Pre-condition: every op has a non-empty src-set, or we infer transfers.
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Mpmd_InferTransfersOption,
    Mpmd_ErrorLimitOption
  ];
}

def InferMeshConvertReduceOpsPass :
    Pass<"mpmd-infer-mesh-convert-reduce-ops", "func::FuncOp"> {
  let summary = "Converts annotated reduce ops to mpmd.reduce ops and flattens "
                "chains of reduce ops.";
  let description = [{
    Converts the annotated reduce ops to mpmd.reduce ops, and also flattens
    chains of these reduce ops.

    In symbols:

    x = add(w0, w1) {mpmd.reduce = #mpmd.reduce<add>}
    y = add(x, w2) {mpmd.reduce = #mpmd.reduce<add>}
      ~~>
    r = mpmd.reduce<add>(w0,w1,w2)
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Option<"inferCrossMeshReductions", "infer-cross-mesh-reductions", "bool",
           /*default=*/"false",
           "Whether to infer cross-mesh reductions. Will be enabled by default "
           "once stable.">
  ];
}

def InferMeshRewriteUsingAnalysisPass :
    Pass<"mpmd-infer-mesh-rewrite-using-analysis", "ModuleOp"> {
  let summary = "Rewrites ops according to the use_set.";
  let description = [{
    This pass assigns meshless ops by wrapping them in fragments, using the
    use_set and src_set analyses.

    It also removes the use_set and src_set attributes as part of clean up, since
    the analyses are no longer needed after this.

    Pre-condition: every op has a use_set, i.e. analysis is complete.
    Pre-condition: every argument of a non-entry point function is used at least
    by one non-terminator op.

    TODO: jupvfranco - consider renaming this pass given that it doesn't depend
    on the analysis so much anymore.
  }];
  let dependentDialects = [
    "mlir::mpmd::MpmdDialect",
    // This pass creates stablehlo reduction ops.
    "mlir::stablehlo::StablehloDialect"
  ];

  let options = [
    Option<"maxClones", "max-clones", "int", /*default=*/"1",
           "How many copies of a meshless operation we allow. Setting it to 1 "
           "means we never clone the op.">
  ];
}

def InferMeshFinalizePass :
    Pass<"mpmd-infer-mesh-finalize", "ModuleOp"> {
  let summary = "Applies final clean up after patterns mesh inference.";
  let dependentDialects = ["mlir::mpmd::MpmdDialect"];

  let options = [
    Mpmd_InferTransfersOption
  ];
}

def InferMeshValidateSrcSetNotEmptyPass :
    Pass<"mpmd-infer-mesh-validate-src-set-not-empty", "ModuleOp"> {
  let summary = "Validates that every meshless op has a non-empty src_set.";
  let description = [{
    This pass validates all meshless ops, checking that the op can be assigned
    somewhere. I.e. for meshless ops which aren't func ops, it errors when the
    src_set is empty on an op or if it was inferred to be a cross-mesh reduction
    but it is not converted. This is a prerequisite for func leaves assignment.

    For func ops, it suffices to check the above conditions for the func args,
    since the func returns either meshless ops, or block args.

    This needs to be a pass on the module level, since in the case of an error
    on the callee, we want to print the callers.

    Pre-condition: cross-mesh reductions should be converted to reduce ops before
    this pass is run.
  }];

  let options = [
    Mpmd_ErrorLimitOption
  ];
}

def InferMeshValidateNoAdditionalTransfersNeededPass :
    Pass<"mpmd-infer-mesh-validate-no-additional-transfers-needed", "ModuleOp"> {
  let summary = "Validates no additional transfers are needed for mesh assignment.";
  let description = [{
    This pass validates that mesh assignment is possible for all meshless ops
    without introducing any additional transfers.

    For meshless ops which aren't func ops, it errors when:
    1. use_set is not contained in the src_set for a given op, i.e. a transfer
    is needed.

    For func ops, it suffices to check the above conditions for the func args,
    since the func returns either meshless ops, or block args.
  }];

  let options = [
    Mpmd_ErrorLimitOption
  ];
}

//===----------------------------------------------------------------------===//
// End of - Infer Mesh assignment passes
//===----------------------------------------------------------------------===//
