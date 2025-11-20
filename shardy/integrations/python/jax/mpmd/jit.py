# Copyright 2025 The MPMD Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of sdy.mpmd.jit and auxiliary classes/functions."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any

from absl import logging
import jax
from jax.experimental import layout as jax_layout
from jax.interpreters import mlir
from jax.jaxlib import _sdy_mpmd as jaxlib_mpmd
import jaxtyping
import numpy as np
import typing_extensions

from shardy.integrations.python.jax.mpmd import ops
from shardy.integrations.python.jax.mpmd import stages
from shardy.integrations.python.jax.mpmd import types
from shardy.integrations.python.jax.mpmd import utils

PyTree = jaxtyping.PyTree


@dataclasses.dataclass(frozen=True)
class _MpmdPartitioningArgs:
  """Arguments for mpmd_py.apply_mpmd_partitioning.

  This is essentially a processed version of a MpmdConfig dataclass, but in a
  format that is more convenient for the C++ function. Note that users should
  interface with MpmdConfig instead of this class.

  Attributes:
    func_name: The name of the function to partition, used for logging and
      debugging.
    named_meshes: A sequence of mesh names and their shapes (e.g., `[('mesh1',
      [('x', 2)])]`), representing the MPMD topology.
    assignment: A mapping from named computations/tensors to their assigned mesh
      and optional stage ID.
    input_meshes: A flat sequence of mesh names for each function input, indexed
      by the flattened input argument index. Assigning a mesh to an input tensor
      means that the tensor will be placed on that mesh. `None` indicates no
      assignment. Also see `input_mesh_assignment` in `types.MpmdConfig`.
    output_meshes: A flat sequence of mesh names for each function output,
      indexed by the flattened output argument index. Assigning a mesh to an
      output tensor means that the tensor will be placed on that mesh. `None`
      indicates no assignment. Also see `output_mesh_assignment` in
      `types.MpmdConfig`.
    donate_argnums: A sequence of indices of function arguments whose buffers
      can be donated (reused).
    input_output_constraints: Constraints that enforce that certain function
      inputs and outputs have the same mesh assignment and sharding. Each
      `tuple[int, int]` represents a constraint where the first `int` is the
      flattened input index and the second `int` is the flattened output index
      that it is constrained to. Also see `input_output_constraints` in
      `types.MpmdConfig`.
    device_type: See `types.MpmdConfig.device_type`.
    tpu_topology_args_proto: See `types.MpmdConfig.tpu_info`. This is required
      for TPUs when using GSPMD partitioning.
    partitioning_options: See `types.MpmdConfig.partitioning_options`.
    fragment_merge_rules: See `types.MpmdConfig.fragment_merge_rules`.
    fragment_schedule_rules: See `types.MpmdConfig.fragment_schedule_rules`.
  """

  func_name: str
  named_meshes: Sequence[tuple[str, list[tuple[str, int]]]]
  assignment: types.NameToMeshAssignment
  input_meshes: Sequence[str | None]
  output_meshes: Sequence[str | None]
  donate_argnums: Sequence[int]
  partitioning_options: types.PartitioningOptions | None
  fragment_merge_rules: Sequence[jaxlib_mpmd.FragmentMergeRule]
  fragment_schedule_rules: Sequence[jaxlib_mpmd.FragmentScheduleRule]


@dataclasses.dataclass(frozen=True)
class MpmdLoweredArgs:
  """Arguments for constructing a stages.MpmdLowered object."""

  stablehlo_mlir_module: mlir.ir.Module
  jax_fn_info: utils.JaxFunctionInfo
  args_info: PyTree[jax.stages.ArgInfo]
  topology: types.Topology
  name_to_mesh_map: types.NameToMeshAssignment
  flat_input_mesh_assignment: Sequence[str] | None = None


def _apply_partitioning(
    mlir_module: mlir.ir.Module,
    partitioning_args: _MpmdPartitioningArgs,
    phases: jaxlib_mpmd.PartitioningPhase,
) -> jaxlib_mpmd.PartitioningResult:
  """Applies MPMD partitioning to the MLIR module."""
  return jaxlib_mpmd.apply_mpmd_partitioning(
      mlir_module,
      func_name=partitioning_args.func_name,
      named_meshes=partitioning_args.named_meshes,
      assignment=partitioning_args.assignment,
      input_meshes=partitioning_args.input_meshes,
      output_meshes=partitioning_args.output_meshes,
      donate_argnums=partitioning_args.donate_argnums,
      partitioning_options=partitioning_args.partitioning_options,
      fragment_merge_rules=partitioning_args.fragment_merge_rules,
      # TODO: b/424385447 - Reenable fragment_schedule_rules once
      # we update jaxlib.
      phases=phases,
  )


class MpmdGspmdTraced(jax.stages.Traced):
  """A `Traced` object for MPMD parallelism."""

  def __init__(
      self,
      func: Callable[..., Any],
      mpmd_config: types.MpmdConfig,
      donate_argnums: Sequence[int],
      traced: jax.stages.Traced,
      traced_args: Any,
      lowering_platforms: tuple[str, ...] | None = None,
  ):
    """Initializes an MpmdGspmdTraced object."""
    self._traced = traced

    self.func = func
    self._fun_name = utils.get_func_name(self.func, prefix='')
    self._donate_argnums = donate_argnums
    self._traced_args = traced_args
    self._mpmd_config = mpmd_config
    self._lowering_platforms = lowering_platforms

  fun_name = property(lambda self: self._fun_name)

  @typing_extensions.override
  @property
  def out_info(self) -> Any:
    return self._traced.out_info

  def _prepare_partitioning_args(
      self, _private_parameters: mlir.LoweringParameters | None = None  # pytype: disable=signature-mismatch  # pylint: disable=invalid-name
  ) -> tuple[mlir.ir.Module, _MpmdPartitioningArgs, MpmdLoweredArgs]:
    """Prepares arguments for the partitioning pipeline."""
    args = self._traced_args
    func_name = self.fun_name
    if _private_parameters is None:
      _private_parameters = mlir.LoweringParameters()
    _private_parameters = dataclasses.replace(
        _private_parameters,
        override_lowering_rules=ops.jit_lowerings,
    )

    logging.info('Lowering function %s via jax.jit.', func_name)
    lowered_computation = self._traced.lower(
        lowering_platforms=self._lowering_platforms,
        _private_parameters=_private_parameters,
    )
    logging.info('Lowering function %s via jax.jit completed.', func_name)

    def _shaped_abstractify(x):
      if isinstance(x, (jax.ShapeDtypeStruct, np.ndarray)):
        return jax.core.ShapedArray(x.shape, x.dtype)
      if isinstance(x, jax.Array):
        return x.aval
      if isinstance(x, (int, bool, float, complex)):
        return jax.core.ShapedArray(shape=np.shape(x), dtype=np.dtype(type(x)))
      raise NotImplementedError(f'Unsupported type: {type(x)}')

    # Flatten in the inputs and outputs, preserving the structures of the
    # respective trees. Note the flattened input shapes contain all inputs,
    # including the unused ones.
    flat_in_shaped, in_tree = jax.tree.flatten(args)
    global_flat_in_abstract_values = [
        _shaped_abstractify(shaped) for shaped in flat_in_shaped
    ]
    flat_out_shaped, out_tree = jax.tree.flatten(lowered_computation.out_info)
    global_flat_out_abstract_values = [
        _shaped_abstractify(shaped) for shaped in flat_out_shaped
    ]

    # TODO(joelwee): Use public interface for moving resources when
    # available.
    # pylint:disable-next=protected-access
    preserved_input_indices = lowered_computation._lowering.compile_args.get(
        'kept_var_idx'
    )
    erased_inputs_from_signature = [
        i not in preserved_input_indices for i, _ in enumerate(flat_in_shaped)
    ]
    erased_input_tree = jax.tree.unflatten(
        in_tree, erased_inputs_from_signature
    )

    erased_paths = ', '.join(
        jax.tree_util.keystr(path)
        for path, is_erased in jax.tree.leaves_with_path(erased_input_tree)
        if is_erased
    )
    if erased_paths:
      logging.debug(
          'Erased inputs from MLIR function signature at paths: \n%s',
          erased_paths,
      )

    jax_fn_info = utils.JaxFunctionInfo(
        func_name,
        global_flat_in_abstract_values,
        global_flat_out_abstract_values,
        in_tree,
        out_tree,
        kept_inputs_indices=preserved_input_indices,
    )

    # Flatten the input to mesh assignment pytree.
    flat_input_mesh_assignment = ()
    if self._mpmd_config.input_mesh_assignment:
      # TODO: b/339592751 - Use a more suitable function to flatten the
      # input mesh assignment.
      flat_input_mesh_assignment = jax.api_util.flatten_axes(
          'flat_input_mesh_assignment',
          in_tree,
          self._mpmd_config.input_mesh_assignment,
      )
    kept_inputs_mesh_assignment = ()
    if flat_input_mesh_assignment:
      kept_inputs_mesh_assignment = [
          mesh_assignment
          for i, mesh_assignment in enumerate(flat_input_mesh_assignment)
          if i in jax_fn_info.kept_inputs_indices
      ]
    # Flatten the output to mesh assignment pytree.
    flat_output_mesh_assignment = ()
    if self._mpmd_config.output_mesh_assignment:
      # TODO: b/339592751 - Use a more suitable function to flatten the
      # output mesh assignment.
      flat_output_mesh_assignment = jax.api_util.flatten_axes(
          'flat_output_mesh_assignment',
          out_tree,
          self._mpmd_config.output_mesh_assignment,
      )

    topology_shape = [
        (mesh_name, list(mesh.shape.items()))
        for mesh_name, mesh in self._mpmd_config.topology.items()
    ]
    stablehlo_mlir_module = lowered_computation.compiler_ir(dialect='stablehlo')
    # Clone the module to avoid modifying the original module, which might be
    # cached by JAX.
    assert (
        jax.config.jax_use_shardy_partitioner
    ), 'sdy.mpmd can only be used with Shardy partitioner.'
    mlir_module = jaxlib_mpmd.clone_mlir_module(
        stablehlo_mlir_module, ['mpmd.sdy_lowered']
    )

    args_info = lowered_computation.args_info
    flat_args_info, _ = jax.tree_util.tree_flatten(args_info)
    # `flat_args_info` contains args that have been removed by JAX (if any), so
    # we need to filter them out.
    kept_flat_args_info = [
        arg_info
        for i, arg_info in enumerate(flat_args_info)
        if i in jax_fn_info.kept_inputs_indices
    ]
    donate_argnums = [
        argnum
        for argnum, arg_info in enumerate(kept_flat_args_info)
        if arg_info.donated
    ]

    assert not self._mpmd_config.fragment_merge_rules
    fragment_merge_rules = []
    assert not self._mpmd_config.fragment_schedule_rules
    fragment_schedule_rules = []

    partitioning_args = _MpmdPartitioningArgs(
        func_name=func_name,
        named_meshes=topology_shape,
        assignment=self._mpmd_config.mesh_and_stage_assignment,
        input_meshes=kept_inputs_mesh_assignment,
        output_meshes=flat_output_mesh_assignment,
        donate_argnums=donate_argnums,
        partitioning_options=self._mpmd_config.partitioning_options,
        fragment_merge_rules=fragment_merge_rules,
        fragment_schedule_rules=fragment_schedule_rules,
    )
    lowered_args = MpmdLoweredArgs(
        stablehlo_mlir_module=stablehlo_mlir_module,
        jax_fn_info=jax_fn_info,
        args_info=args_info,
        topology=self._mpmd_config.topology,
        name_to_mesh_map=self._mpmd_config.mesh_and_stage_assignment,
        flat_input_mesh_assignment=flat_input_mesh_assignment,
    )
    return mlir_module, partitioning_args, lowered_args

  @typing_extensions.override
  def lower(
      self,
      _private_parameters: mlir.LoweringParameters | None = None,  # pytype: disable=signature-mismatch  # pylint: disable=invalid-name
  ) -> stages.MpmdLowered:
    """Lowers the Jax function wrapped in this object to MLIR.

    Args:
      _private_parameters: Lowering parameters for JAX.

    Returns:
      A `stages.MpmdLowered` object.

    This includes MPMD partition and GSPMD propagation across different MPMD
    programs.
    """
    mlir_module, partitioning_args, lowered_args = (
        self._prepare_partitioning_args(_private_parameters)
    )

    partitioning_result = _apply_partitioning(
        mlir_module, partitioning_args, jaxlib_mpmd.PartitioningPhase.ALL
    )
    ifrt_ir_module = jaxlib_mpmd.clone_mlir_module(
        partitioning_result.mpmd_module
    )
    jaxlib_mpmd.lower_to_ifrt(ifrt_ir_module)

    return stages.MpmdLowered(
        stablehlo_mlir_module=lowered_args.stablehlo_mlir_module,
        partitioning_result=stages.PartitioningResult(
            module_io_sharding_specs_and_meshes=partitioning_result.module_io_sharding_specs_and_meshes,
            ifrt_ir_module=ifrt_ir_module,
            mpmd_module=partitioning_result.mpmd_module,
        ),
        jax_fn_info=lowered_args.jax_fn_info,
        args_info=lowered_args.args_info,
        lowering_metadata={},
        topology=lowered_args.topology,
        name_to_mesh_map=lowered_args.name_to_mesh_map,
        flat_input_mesh_assignment=lowered_args.flat_input_mesh_assignment,
    )


def _replace_mesh_on_sharding(
    path: jax.tree_util.KeyPath,
    sharding: jax.sharding.NamedSharding,
    mesh: jax.sharding.Mesh,
) -> jax.sharding.Sharding | None:
  """Replaces the mesh on a sharding, if possible. Returns None if invalid."""
  if (
      sharding.mesh.axis_names != mesh.axis_names
      or sharding.mesh.axis_sizes != mesh.axis_sizes
  ):
    # We only check the mesh names and sizes, as the underlying devices are
    # allowed to be different.
    logging.error(
        'Sharding at path %s has mesh mismatch. Expected mesh: %s, got'
        ' mesh: %s',
        jax.tree_util.keystr(path),
        mesh,
        sharding.mesh,
    )
    return None

  return jax.sharding.NamedSharding(
      mesh, sharding.spec, memory_kind=sharding.memory_kind
  )


def _standardize_sharding_mesh(
    shardings: PyTree[jax.sharding.Sharding],
    mesh: jax.sharding.Mesh,
    debug_str: str,
) -> PyTree[jax.sharding.Sharding]:
  """Updates all shardings to use the same mesh, if possible.

  Args:
    shardings: A tree of shardings.
    mesh: the mesh to use.
    debug_str: A string to debug the error.

  Returns:
    PyTree of shardings with the mesh replaced with `mesh`, if it is a
    NamedSharding.

  Raises:
    ValueError: If any sharding has a mesh that is incompatible with the mesh,
    i.e. different mesh names or sizes.
  """
  has_error = False

  def replace_mesh(
      path: jax.tree_util.KeyPath, sharding: jax.sharding.Sharding
  ) -> jax.sharding.Sharding:
    nonlocal has_error

    if isinstance(sharding, jax_layout.Format):
      return jax_layout.Format(
          sharding.layout,
          sharding=replace_mesh(path, sharding.sharding),
      )

    if not isinstance(sharding, jax.sharding.NamedSharding):
      # This isn't expected, but could happen. We allow it for now, but log some
      # info.
      logging.info(
          'Sharding at path %s is not a NamedSharding: %s.',
          jax.tree_util.keystr(path),
          sharding,
      )
      return sharding
    res = _replace_mesh_on_sharding(path, sharding, mesh)
    if res is None:
      has_error = True
      return sharding
    return res

  res = jax.tree_util.tree_map_with_path(replace_mesh, shardings)

  if has_error:
    raise ValueError(
        'Sharding mismatch when standardizing sharding to use the same mesh'
        f' for "{debug_str}". See error logs for details.'
    )

  return res


def _to_sharded_shape(
    lowering_args: PyTree[jax.ShapeDtypeStruct | jax.Array],
    mesh: jax.sharding.Mesh,
) -> PyTree[jax.ShapeDtypeStruct]:
  """Shapifies the lowering args, unifying the mesh on the sharding if possible.

  Args:
    lowering_args: A tree of shardings.
    mesh: the mesh to use.

  Returns:
    PyTree of ShapeDtypeStructs with the sharding mesh replaced with `mesh`, if
    its sharding is a NamedSharding.

  Raises:
    ValueError: If any sharding has a mesh that is incompatible with the mesh,
    i.e. different mesh names or sizes.
  """
  has_error = False

  def sharded_shape(
      path: jax.tree_util.KeyPath,
      x: jax.ShapeDtypeStruct | jax.Array,
  ) -> jax.ShapeDtypeStruct | jax.Array:
    nonlocal has_error
    if not isinstance(x, (jax.Array, jax.ShapeDtypeStruct)):
      # Could be an np.array or a plain int.
      return x

    sharding = x.sharding
    layout = None
    if hasattr(x, 'format'):
      layout = x.format.layout

    if isinstance(sharding, jax_layout.Format):
      layout = sharding.layout
      sharding = sharding.sharding

    if isinstance(sharding, jax.sharding.NamedSharding):
      sharding = _replace_mesh_on_sharding(path, x.sharding, mesh)
      if sharding is None:
        has_error = True
    else:
      sharding = None

    if layout is not None and sharding is not None:
      sharding = jax_layout.Format(
          layout,
          sharding=sharding,
      )

    return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding)

  res = jax.tree_util.tree_map_with_path(sharded_shape, lowering_args)
  if has_error:
    raise ValueError(
        'Sharding mismatch when standardizing sharding to use the same mesh'
        ' for "lowering_args". See error logs for details.'
    )
  return res


class MpmdWrapped(jax.stages.Wrapped):
  """A `Wrapped` object for MPMD parallelism."""

  def __init__(
      self,
      func: Callable[..., Any],
      mpmd_config: types.MpmdConfig,
      in_shardings: PyTree[jax.sharding.Sharding | None] | None,
      out_shardings: PyTree[jax.sharding.Sharding | None] | None,
      donate_argnums: int | Sequence[int] | None,
      keep_unused: bool = False,
      override_func_name: str | None = None,
  ):
    """Initializes an MpmdWrapped object."""

    if override_func_name:
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
      wrapper.__name__ = override_func_name
      self.func = wrapper
    else:
      self.func = func

    jit_kwargs = {}

    # For in_shardings and out_shardings, don't override the `jax.jit` default
    # unless explicitly provided.
    if in_shardings:
      jit_kwargs['in_shardings'] = _standardize_sharding_mesh(
          in_shardings, mpmd_config._spmd_mesh, debug_str='in_shardings'
      )
      if mpmd_config.read_input_output_mesh_from_shardings:
        assert not mpmd_config.input_mesh_assignment
        mpmd_config = dataclasses.replace(
            mpmd_config,
            input_mesh_assignment=types.mesh_names(
                in_shardings, mpmd_config.topology
            ),
        )
    if out_shardings:
      jit_kwargs['out_shardings'] = _standardize_sharding_mesh(
          out_shardings, mpmd_config._spmd_mesh, debug_str='out_shardings'
      )
      if mpmd_config.read_input_output_mesh_from_shardings:
        assert not mpmd_config.output_mesh_assignment
        mpmd_config = dataclasses.replace(
            mpmd_config,
            output_mesh_assignment=types.mesh_names(
                out_shardings, mpmd_config.topology
            ),
        )

    self._jax_jit_wrapped = jax.jit(
        self.func,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        **jit_kwargs,
    )
    if donate_argnums is None:
      self._donate_argnums = ()
    elif isinstance(donate_argnums, int):
      self._donate_argnums = (donate_argnums,)
    else:
      self._donate_argnums = donate_argnums
    self._mpmd_config = mpmd_config

  def trace(self, *args) -> MpmdGspmdTraced:
    """Traces the computation via `jax.jit` and returns a `MpmdGspmdTraced`."""
    # TODO: b/377706756 - Maybe read the mesh assignment from the args shardings
    # too. This is more complicated, since args may be a mix of jax.Arrays and
    # jax.ShapeDtypeStructs, and also we will need to enforce that the
    # assignments are the same for in_shardings and args.
    # We cannot simply pass `*args` to jax.jit as these may be allocated in
    # different meshes (e.g., via device_put), which would cause jax to throw
    # an error, as jax requires all inputs to have the same device assignment.
    # Given that we only use jax.jit for tracing and lowering, it's enough to
    # pass in shapes, dtypes, and shardings.
    # pylint: disable-next=protected-access
    args = _to_sharded_shape(args, self._mpmd_config._spmd_mesh)
    func_name = utils.get_func_name(self.func, prefix='')
    logging.info('Tracing function %s via jax.jit.', func_name)
    traced = self._jax_jit_wrapped.trace(*args)
    logging.info('Tracing function %s via jax.jit completed.', func_name)

    return MpmdGspmdTraced(
        func=self.func,
        mpmd_config=self._mpmd_config,
        donate_argnums=self._donate_argnums,
        traced=traced,
        traced_args=args,
    )

  def lower(self, *args) -> stages.MpmdLowered:
    """See `MpmdGspmdTraced.lower`."""
    return self.trace(*args).lower()

  def __call__(self, *args):
    """See base class."""
    # TODO(b/325009737): consider using device puts here using the computed
    # NamedShardings.
    return self.lower(*args).compile()(*args)


def jit(
    func: Callable[..., Any],
    mpmd_config: types.MpmdConfig,
    *,
    # Possibly deprecated in the future, if also deprecated by `jax.jit`.
    in_shardings: PyTree[jax.sharding.Sharding | None] | None = None,
    out_shardings: PyTree[jax.sharding.Sharding | None] | None = None,
    donate_argnums: int | Sequence[int] | None = None,
    keep_unused: bool = False,
    override_func_name: str | None = None,
) -> MpmdWrapped:
  """Partitions a function for MPMD and SPMD parallelism.

  This uses PartIR as the MPMD partitioner and GSPMD as the SPMD partitioner.
  This aims to have the same args as jax.jit, but at the moment only
  `donate_argnums` is supported.

  Args:
    func: Jax function to partition, with one or more named computations which
      can be assigned to different meshes in the topology for MPMD partitioning.
    mpmd_config: Config specific to Mpmd with PartIR. See `types.MpmdConfig`.
    in_shardings: See `jax.jit`.
    out_shardings: See `jax.jit`.
    donate_argnums: See `jax.jit`.
    keep_unused: See `jax.jit`.
    override_func_name: If provided, the function name will be overridden to
      the provided value.

  Returns:
    An MpmdWrapped object.
  """
  return MpmdWrapped(
      func,
      mpmd_config,
      in_shardings,
      out_shardings,
      donate_argnums,
      keep_unused,
      override_func_name=override_func_name,
  )
