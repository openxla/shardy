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

def LowerToFragmentCallsPass : Pass<"mpmd-lower-to-fragment-calls", "ModuleOp"> {
  let summary = "Lowers MPMD fragments to fragment calls and functions.";
  let description = [{
    Replaces all fragments with fragment calls.

    This pass creates a function for every group of fragments that have an
    identical body and mesh shape, with the name of the first encountered
    fragment in the group as the symbol name, and adds it to the symbol table.

    The body of each function is extracted from the body of the first
    encountered fragment in the respective group, and the mesh shape is taken
    from the topology by the mesh name of that fragment. If the fragment has
    argument attributes about input-output aliasing, they will be assigned to
    the argument attributes of the lowered function.

    Since functions must have unique names, this pass appends an index to the
    name of all but the first function with the same original name, i.e., the
    ith function with name "some_name" for i > 0 will have the name
    "some_name_i".
  }];
  let dependentDialects = ["mlir::mpmd::MpmdDialect", "mlir::sdy::SdyDialect"];

  let options = [
    Option<"groupAcrossMeshes", "group-across-meshes", "bool",
           /*default=*/"false",
           "Whether to do more aggressive fragment grouping, across meshes. "
           "This may not be desirable for heterogeneous systems.">,
    Option<"verboseLogging", "verbose-logging", "bool", /*default=*/"false",
           "Whether to enable verbose logging">,
  ];
}

def MarkAliasingAndDonationPass :
        PassBase<"mpmd-mark-aliasing-and-donation", "DistributedFunctionPass"> {
  let summary = "Marks each fragment with aliasing or donation information.";
  let description = [{
    Sets an `arg_attrs` attribute to Fragment ops when any of their inputs
    can be aliased with an output or donated. Each input that can be aliased
    will have a `tf.aliasing_output` attribute. Otherwise, a
    `jax.buffer_donor = true` attribute. For example, `{arg_attrs =
    \[{tf.aliasing_output = 0 : i32}, {jax.buffer_donor = true}, {}\]}` shows
    the first input can be aliased with output 0, the second input can be
    donated so that XLA will find an aliased output, and the third input can't
    be aliased or donated.
  }];
}

def MarkFragmentReservedMemoryPass :
        PassBase<"mpmd-mark-fragment-reserved-memory", "DistributedFunctionPass"> {
  let summary = "Mark each fragment with the amount of memory that needs to be "
                "reserved for compilation.";
  let description = [{
    Assigns an `xla_tpu_user_reserved_hbm_bytes` attribute to each fragment
    which will tell XLA how many bytes to keep around while compiling each
    fragment. By keeping track of live tensors on a mesh, XLA will know of
    actual minimum memory usage at the time of execution, and we can prevent
    it from applying optimizations in the executable that would increase
    memory usage beyond the device capacity.
    NOTE: this pass assumes that fragments are executed in program order.
  }];
}

def MarkInputOutputWithLayoutsPass :
        PassBase<"mpmd-mark-input-output-with-layouts", "DistributedFunctionPass"> {
  let summary = "Propagates layouts from func args/results to fragment "
                "args/results.";
  let description = [{
    Propagates `mhlo.layout_mode` attributes from program inputs to fragments
    that are consumers of program input, and propagate `mhlo.layout_mode`
    attributes from program outputs to fragments that are output producers.
    If a program argument is returned then `mhlo.layout_mode` is propagated
    to/from program results. Inputs, outputs and fragment arguments/results
    that are connected with a transfer op are always overwritten with the
    DEFAULT layout, because transfers only support default layouts.

    If a program argument is set to AUTO layout and is used in multiple
    fragments, then we set it to be the DEFAULT layout to setup consistent
    layout across fragments.

    If program output, and also is a fragment result, is set to AUTO layout
    and used in other fragments as an input, we also set it to be the DEFAULT
    layout to setup consistent layout across fragments.
  }];
}

// TODO: b/374994155 - Consider using memory-kind field of mesh types instead.
def MarkOffloadedInputOutputPass :
        PassBase<"mpmd-mark-offloaded-input-output", "DistributedFunctionPass"> {
  let summary = "Marks offloaded input and output values so the compiler knows "
                "they are in host memory.";
  let description = [{
    Marks fragment args and results with attributes to identify which values
    live in host memory, so that this information can be used by XLA. Also marks
    the entrypoint func args and results so that Pathways can use this
    information.
  }];
}

def DelayInferredFragmentsPass :
        PassBase<"mpmd-delay-inferred-fragments", "DistributedFunctionPass"> {
  let summary = "Delays inferred fragments to be executed as late as possible.";
  let description = [{
     Moves inferred fragments to be executed as late as possible, i.e., right
     before their first consumer.
  }];
}

def DelayTransfersFromCpuPass :
        PassBase<"mpmd-delay-transfers-from-cpu", "DistributedFunctionPass"> {
  let summary = "Delays transfers from CPU to be executed as late as possible.";
  let description = [{
    Moves cpu-to-device transfer ops right before their first consumer. This
    means postponing the allocation of memory for the transferred buffers, which
    can be beneficial for HBM usage.
  }];
}

def ValidateNoReshardsPass :
        PassBase<"mpmd-validate-no-reshards", "DistributedFunctionPass"> {
  let summary = "Validates that no reshard-only fragments exist.";
  let description = [{
    A reshard-only fragment is a fragment that contains only a `mpmd.return` of
    one of its arguments. These fragments usually indicate an unexpected reshard.

    This pass identifies such fragments and emits an error.
  }];
}
