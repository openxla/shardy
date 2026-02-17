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

def ConvertSdyShardingsToMpmdTypesPass :
    PassBase<"mpmd-convert-sdy-shardings-to-mpmd-types",
             "DistributedFunctionPass"> {
  let summary = "Moves shardings from op attrs to `!mpmd.mesh_tensor` types.";
  let description = [{
    Moves shardings from the attributes of MPMD ops (e.g. fragments, transfer)
    to the `MeshTensorType` of their results. Assuming we apply SDY propagation
    before this pass, the SPMD shardings are attached to the op's attributes.
    This pass moves the sharding to `MeshTensorType`s since later passes require
    the type to contain a sharding.

    This pass also removes any sharding from ops that don't have a
    `MeshTensorType`, i.e., ops inside `mpmd.fragment` ops.
  }];

  let dependentDialects = ["mlir::mpmd::MpmdDialect"];
}

def ConvertSdyConstantsPass :
    Pass<"mpmd-convert-sdy-constants", "func::FuncOp"> {
  let summary = "Converts `sdy.constant` ops into `stablehlo.constant`.";
  let description = [{
    Converts any `sdy.constant` op, that isn't foldable, into a
    `stablehlo.constant` op. There is no reason to prevent constant folding
    since we are stripping shardings away from constants in
    `mpmd-convert-sdy-shardings-to-mpmd-types`.
  }];

  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def PopulateUnreducedOutShardingPass :
    PassBase<"mpmd-populate-unreduced-out-sharding", "DistributedFunctionPass"> {
  let summary = "Populates unreduced out shardings for fragments.";
  let description = [{
    Iterates through each fragment, looks at its return values, checks if that
    value has an unreduced axis sharding constraint, and if so, amends the
    fragment `out_sharding` attribute to have the unreduced sharding.

    If the `out_sharding` attribute doesn't exist, it is added with default
    replicated shardings (with the exception of the unreduced axes).

    This is needed because the SDY propagation pipeline does not populate the
    unreduced out shardings, only the in shardings. Since technically Shardy
    is not responsible for unreduced axes propagation, this pass lives in MPMD
    instead of SDY.
  }];

  let dependentDialects = ["mlir::sdy::SdyDialect", "mlir::mpmd::MpmdDialect"];
}

def EnforceUserShardingsPass :
    PassBase<"mpmd-enforce-user-shardings", "DistributedFunctionPass"> {
  let summary = "Enforces the user specified shardings for inputs and outputs.";
  let description = [{
    Enforces the input and output shardings of fragments that take function
    arguments or produce function results respectively, to be the ones specified
    by the user, i.e., the input and outputs shardings of the function.

    After this pass, fragment and transfer users of function arguments and
    producers of function results should have the same shardings as the ones
    specified by the user. If the user did not specify a sharding for an input
    or output, this pass keeps the sharding that propagation assigned.

    Precondition:
    - The user shardings are set on the function's arguments and results as
      attributes.
    - The fragment shardings are set on `in_shardings` and `out_shardings`
      attributes.
  }];

  let dependentDialects = ["mlir::sdy::SdyDialect"];
}

// TODO: b/329842439 - Longer term, we need a better strategy or better support
// for this.
def ExtractReshardsFromInterMeshTransfersPass :
    PassBase<"mpmd-extract-reshards-from-inter-mesh-transfers",
             "DistributedFunctionPass"> {
  let summary = "Moves SPMD resharding around an inter-mesh transfer to inside "
                "a fragment.";
  let description = [{
    Ensures that all inter-mesh transfers do not (SPMD) reshard the array (their
    in and out shardings are the same), by updating the types of
    producer/consumer fragments or by creating inferred fragments for
    non-fragment producers/consumers.

    This is needed as MPMD runtimes have limitations w.r.t. supported
    reshardings.

    This pass is only applied to MPMD functions in global view and with a
    homogeneous topology.

    Precondition: all shardings are specified as op attributes and not in types.
  }];

  let dependentDialects = ["mlir::mpmd::MpmdDialect", "mlir::sdy::SdyDialect"];
}

def SimplifyProgramPass :
    Pass<"mpmd-simplify-program", "func::FuncOp"> {
  let summary = "Removes redundant arg/results from fragments.";
  let description = [{
    Simplifies a fragment or loop, its operands and results, and their
    corresponding block arguments and return values.
  }];

  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}
