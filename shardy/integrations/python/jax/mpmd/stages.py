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

"""Definition of the Lowered JAX stage, which is responsible for compilation."""

from collections.abc import Callable, Sequence, Set
import dataclasses
import functools
from typing import Any, NamedTuple, cast

from absl import logging
import jax
from jax._src import core
from jax._src import stages
from jax._src.interpreters import pxla
from jax.experimental import layout
import jax.extend.backend as jax_backend
from jax.interpreters import mlir as jax_mlir
from jax.jaxlib import _sdy_mpmd as jaxlib_mpmd
import jaxtyping
import numpy as np

from shardy.integrations.python.jax.mpmd import types as mpmd_types
from shardy.integrations.python.jax.mpmd import utils

PyTree = jaxtyping.PyTree
FunctionNamedShardings = utils.FunctionNamedShardings
# Will replace this once jaxlib landed.
ifrt_mpmd_py = Any # pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True)
class MpmdJitShardingInfo:
  """Holds partitioning related information obtained by partitioning a module.

  This information can be used to initialize the inputs of a partitioned
  function (e.g., parameters and optimizer states).

  Attributes:
    named_shardings: the jax.sharding.NamedShardings of all inputs and outputs
      of the partitioned function.
    io_mesh_assignment: the mesh names of all inputs and outputs of the
      partitioned function.
    topology: the topology used to partition the function.
    name_to_mesh_map: a mapping between names from
      named_computation/named_tensor names to mesh names.
  """

  named_shardings: FunctionNamedShardings
  io_mesh_assignment: mpmd_types.FunctionIOMeshAssignment
  topology: mpmd_types.Topology
  name_to_mesh_map: mpmd_types.NameToMeshAssignment


@dataclasses.dataclass(frozen=True)
class MpmdLoweringCompileArgs:
  compile_args: dict[str, Any]
  _device_list: list[Any]


@functools.lru_cache(maxsize=1024)
def _abstract_to_concrete_mesh(
    abstract_mesh: jax.sharding.AbstractMesh,
    device_assignment: tuple[jax.Device, ...],
) -> jax.sharding.Mesh:
  # Follows the same logic as JAX for converting a device assignment to a mesh.
  np_dev = np.vectorize(lambda i: device_assignment[i], otypes=[object])(
      np.arange(len(device_assignment))
  )
  return jax.sharding.Mesh(
      np_dev.reshape(abstract_mesh.axis_sizes),
      abstract_mesh.axis_names,
      axis_types=abstract_mesh.axis_types,
  )


class MpmdExecutable(stages.Executable):
  """Executable for an MPMD program."""

  def __init__(
      self,
      executable: Any,
      *,
      module_ir: jax_mlir.ir.Module,
      func_name: str,
      flat_in_avals: Sequence[jax.core.ShapedArray],
      out_avals: Sequence[jax.core.ShapedArray],
      in_shardings: jaxtyping.PyTree[jax.sharding.Sharding],
      flat_out_shardings: Sequence[jax.sharding.Sharding],
      kept_inputs_indices: Set[int],
      donated_inputs_indices: Set[int],
      topology: mpmd_types.Topology,
  ):
    self._executable = executable
    self._module_ir = module_ir
    self._func_name = func_name
    self.in_avals = flat_in_avals
    self._in_shardings_tree = in_shardings
    self._kept_in_avals = []
    self._kept_in_shardings = []
    self._kept_in_shardings_paths = []
    self._topology = topology

    flat_in_shardings_with_paths = jax.tree.leaves_with_path(in_shardings)
    for i, (in_aval, (path, in_sharding)) in enumerate(
        zip(flat_in_avals, flat_in_shardings_with_paths, strict=True)
    ):
      if i in kept_inputs_indices:
        self._kept_in_avals.append(in_aval)
        self._kept_in_shardings.append(in_sharding)
        self._kept_in_shardings_paths.append(path)
    self._kept_in_shardings = tuple(self._kept_in_shardings)
    self._kept_in_shardings_paths = tuple(self._kept_in_shardings_paths)
    self._out_shardings = flat_out_shardings
    self._kept_var_idx = kept_inputs_indices
    self._donated_inputs_indices = donated_inputs_indices

    # Hacks for the export flow.
    unloaded_executable = NamedTuple(
        'MpmdUnloadedExecutableInfo',
        [
            ('output_avals', Sequence[jax.core.ShapedArray]),
            ('output_shardings', Sequence[jax.sharding.Sharding]),
        ],
    )
    self._unloaded_executable = unloaded_executable(
        out_avals, flat_out_shardings
    )
    # TODO(b/396086979): Carry the debug info from the lowered computation.
    self._debug_info = core.DebugInfo(
        'mpmd_executable',
        self._func_name,
        tuple(f'arg[{i}]' for i in range(len(self._kept_in_avals))),
        tuple(f'[{i}]' for i in range(len(out_avals))),
    )

  def call(self, *args) -> Sequence[Any]:
    """Calls the MPMD program. The args must be sharded as expected."""
    errors = [
        f'Got input {i} of type {type(arg)} instead of `jax.Array`'
        for i, arg in enumerate(args)
        if not isinstance(arg, jax.Array)
    ]
    if errors:
      str_errors = '\n'.join(errors)
      raise ValueError(
          'MPMD Executable called with inputs that are not a `jax.Array`. '
          f'Errors: {str_errors}'
      )
    arg_avals = []
    kept_args = {}
    for i, arg in enumerate(args):
      if i in self._kept_var_idx:
        arg_avals.append(arg.aval)
        kept_args[i] = arg
      elif i in self._donated_inputs_indices:
        # The argument is donated and not used by the program. It is safe
        # to delete the argument.
        arg.delete()
    pxla.check_arg_avals_for_call(
        self._kept_in_avals, arg_avals, self._debug_info
    )
    _verify_shardings_are_equivalent(
        self._func_name,
        self._kept_in_shardings,
        self._kept_in_shardings_paths,
        tuple((idx, arg.sharding, arg.ndim) for idx, arg in kept_args.items()),
        tuple((k, v) for k, v in self._topology.items()),
    )
    return self._executable.execute(tuple(kept_args.values()))

  def create_cpp_call(
      self, params: stages.CompiledCallParams
  ) -> Callable[..., Sequence[Any]] | None:
    del self
    return None

  def as_text(self) -> str:
    """Returns the IFRT IR module after backend-specific lowering passes."""
    return str(self._module_ir)

  @property
  def input_shardings_tree(self) -> jaxtyping.PyTree[jax.sharding.Sharding]:
    return self._in_shardings_tree

  @property
  def _in_shardings(self) -> Sequence[jax.sharding.Sharding]:
    return jax.tree.leaves(self._in_shardings_tree)

  def input_shardings(self) -> Sequence[jax.sharding.Sharding]:
    return self._in_shardings

  def output_shardings(self) -> Sequence[jax.sharding.Sharding]:
    return self._out_shardings

  def input_formats(self) -> Sequence[layout.Format]:
    return [
        layout.Format(l, s) if l is not None else layout.Format(sharding=s)
        for l, s in zip(self._xla_in_layouts, self._in_shardings, strict=True)
    ]

  def output_formats(self) -> Sequence[layout.Format]:
    return [
        layout.Format(l, s)
        for l, s in zip(self._xla_out_layouts, self._out_shardings, strict=True)
    ]

  @property
  def _xla_in_layouts(self) -> Sequence[layout.Layout | None]:
    """Returns the input layouts for used inputs only."""
    input_xla_layouts = self._executable.input_layouts()
    input_shardings = self._in_shardings
    # Remove shardings for unused inputs.
    if len(input_xla_layouts) < len(input_shardings):
      iter_layouts = iter(input_xla_layouts)
      input_xla_layouts = [
          next(iter_layouts) if i in self._kept_var_idx else None
          for i in range(len(input_shardings))
      ]
    return [
        layout.Layout.from_pjrt_layout(l) if l is not None else None
        for l in input_xla_layouts
    ]

  @property
  def _xla_out_layouts(self) -> Sequence[layout.Layout | None]:
    output_xla_layouts = self._executable.output_layouts()
    return [layout.Layout.from_pjrt_layout(l) for l in output_xla_layouts]

  def cost_analysis(self) -> dict[str, dict[str, Any]]:
    return self._executable.cost_analysis()

  def memory_analysis(self) -> dict[str, Any]:
    return self._executable.get_compiled_memory_stats()

  def runtime_executable(self) -> Any:
    """An arbitrary object representation of this executable.

    See `jax.stages.Executable.runtime_executable`.

    Returns:
      An arbitrary object representation of this executable.
    """
    return self._executable


class MpmdCompiled(jax.stages.Compiled):
  """A compiled computation for MPMD parallelism."""

  def __init__(
      self,
      executable: MpmdExecutable,
      args_info: Any,
      out_tree: jax.tree_util.PyTreeDef,
      no_kwargs: bool = False,
  ):
    # TODO(b/434270189): handle const args
    const_args = []
    super().__init__(executable, const_args, args_info, out_tree, no_kwargs)
    assert isinstance(executable, MpmdExecutable)
    self._mpmd_executable = executable

  @property
  def mpmd_executable(self) -> MpmdExecutable:
    return self._mpmd_executable


def _add_input_specs_for_removed_inputs(
    jax_fn_info: utils.JaxFunctionInfo,
    meshes_and_specs: jaxlib_mpmd.FunctionIOShardingSpecsAndMeshes,
    arbitrary_mesh_name: str,
    flat_input_mesh_assignment: Sequence[str] | None = None,
) -> jaxlib_mpmd.FunctionIOShardingSpecsAndMeshes:
  """Returns the input specs for the function."""
  # We leave these PartitionSpecs as replicated, but we should be able to
  # read them from the `jax` lowered computation.
  # TODO: b/319824326 - Read PartitionSpecs from Jax lowered computations.
  # These PSpecs do not affect this compiled function since the removed
  # unused inputs will not be put on any devices even if the mesh
  # is specified by the user, but they might be relevant for
  # param initialization.
  placeholder_spec = jaxlib_mpmd.NamedSpmdShardingSpec(
      # Pick an arbitrary mesh for the unused_args
      arbitrary_mesh_name,
      [[]],
      memory_kind=None,
  )
  input_specs = jax_fn_info.with_placeholder_for_removed_inputs(
      meshes_and_specs.input_specs,
      placeholder_spec,
  )

  removed_inputs_idx = [
      i
      for i in range(len(jax_fn_info.global_flat_input_abstract_values))
      if i not in jax_fn_info.kept_inputs_indices
  ]
  if flat_input_mesh_assignment:
    arbitrarily_assigned_input_indices = []
    for i in removed_inputs_idx:
      if input_assignment := flat_input_mesh_assignment[i]:
        mesh_name_without_memory_kind = input_assignment.split('#')[0]
        input_specs[i] = jaxlib_mpmd.NamedSpmdShardingSpec(
            mesh_name_without_memory_kind, [[]], memory_kind=None
        )
      else:
        arbitrarily_assigned_input_indices.append(i)
  else:
    arbitrarily_assigned_input_indices = removed_inputs_idx

  if arbitrarily_assigned_input_indices:
    logging.info(
        'Inputs at indices %s of the flat input array were assigned to mesh'
        " '%s'. These tensors aren't used anywhere in the computation and"
        " weren't allocated.",
        list(map(str, arbitrarily_assigned_input_indices)),
        arbitrary_mesh_name,
    )

  return jaxlib_mpmd.FunctionIOShardingSpecsAndMeshes(
      input_specs,
      meshes_and_specs.output_specs,
  )


@dataclasses.dataclass(frozen=True)
class PartitioningResult:
  """A wrapper of PartitioningResult, which stores MlirModules instead of ModuleOps."""

  module_io_sharding_specs_and_meshes: (
      jaxlib_mpmd.FunctionIOShardingSpecsAndMeshes
  )
  ifrt_ir_module: jax_mlir.ir.Module
  mpmd_module: jax_mlir.ir.Module


class MpmdLowered(stages.Lowered):
  """A lowered and partitioned computation for MPMD parallelism.

  (c.f. _Lowered and jax.stages.Lowered)
  """

  def __init__(
      self,
      stablehlo_mlir_module: jax_mlir.ir.Module,
      partitioning_result: PartitioningResult,
      *,
      jax_fn_info: utils.JaxFunctionInfo,
      args_info: jaxtyping.PyTree[stages.ArgInfo],
      lowering_metadata: dict[str, Any],
      topology: mpmd_types.Topology,
      name_to_mesh_map: mpmd_types.NameToMeshAssignment,
      flat_input_mesh_assignment: Sequence[str] | None = None,
  ):
    """Initializes an MpmdLowered object.

    See base class.

    Args:
      stablehlo_mlir_module: the result of JAX lowering to StableHLO.
      partitioning_result: the result of partitioning.
      jax_fn_info: See `utils.JaxFunctionInfo`.
      jax_fn_info: See `utils.JaxFunctionInfo`.
      args_info: A PyTree of `ArgInfo` corresponding to the input PyTree.
      lowering_metadata: lowering metrics and additional information.
      topology: the MPMD topology used.
      name_to_mesh_map: a mapping between names and meshes.
      flat_input_mesh_assignment: the mesh assignment of the inputs, including
        removed unused inputs.
    """
    self._stablehlo_mlir_module = stablehlo_mlir_module
    self.partitioning_result = partitioning_result
    self.args_info = args_info
    self.name = jax_fn_info.func_name
    self.topology = topology
    self.global_flat_input_abstract_values = (
        jax_fn_info.global_flat_input_abstract_values
    )
    self.global_flat_output_abstract_values = (
        jax_fn_info.global_flat_output_abstract_values
    )

    self.input_tree = jax_fn_info.input_tree
    self.output_tree = jax_fn_info.output_tree
    self.kept_inputs_indices = jax_fn_info.kept_inputs_indices

    meshes_and_specs = partitioning_result.module_io_sharding_specs_and_meshes

    if len(jax_fn_info.kept_inputs_indices) != len(
        jax_fn_info.global_flat_input_abstract_values
    ):
      meshes_and_specs = _add_input_specs_for_removed_inputs(
          jax_fn_info,
          meshes_and_specs,
          list(self.topology.keys())[0],
          flat_input_mesh_assignment,
      )

    self.function_mesh_assignment = mpmd_types.FunctionIOMeshAssignment(
        jax.tree_util.tree_unflatten(
            jax_fn_info.input_tree,
            [spec.mesh_name for spec in meshes_and_specs.input_specs],
        ),
        jax.tree_util.tree_unflatten(
            jax_fn_info.output_tree,
            [spec.mesh_name for spec in meshes_and_specs.output_specs],
        ),
    )

    self.function_named_shardings = (
        utils.meshes_and_sdy_specs_to_named_shardings(
            meshes_and_specs,
            jax_fn_info.input_tree,
            jax_fn_info.output_tree,
            topology,
        )
    )

    self.sharding_info = MpmdJitShardingInfo(
        self.function_named_shardings,
        self.function_mesh_assignment,
        topology,
        name_to_mesh_map,
    )

    self.no_kwargs = True

    self.mpmd_module = partitioning_result.mpmd_module

  def _compile_pathways(
      self,
      compiler_options: (
          stages.CompilerOptions | mpmd_types.MeshToCompileOptions | None
      ) = None,
      device_assignment: tuple[jax.Device, ...] | None = None,
  ) -> MpmdCompiled:
    in_avals = jax.tree.map(lambda x: x._aval, self.args_info)  # pylint: disable=protected-access
    flat_in_avals = jax.tree.leaves(in_avals)
    flat_out_avals = self.global_flat_output_abstract_values
    if device_assignment is None:
      in_shardings = self.function_named_shardings.input_specs
      flat_out_shardings = jax.tree.leaves(
          self.function_named_shardings.output_specs
      )
      topology = self.topology
      device_assignment = []
      for _, mesh in self.topology.items():
        device_assignment.extend(mesh.devices.flat)
    else:
      cur_device_idx = 0
      topology: mpmd_types.Topology = {}
      lowered_mesh_to_compiled_mesh: dict[
          jax.sharding.Mesh, jax.sharding.Mesh
      ] = {}
      for mesh_name, mesh in self.topology.items():
        abstract_mesh = mesh.abstract_mesh
        new_device_idx = cur_device_idx + mesh.size
        if new_device_idx > len(device_assignment):
          raise ValueError(
              f'`device_assignment` of length {len(device_assignment)} does not'
              f' have enough devices for topology {self.topology}'
          )
        new_mesh = _abstract_to_concrete_mesh(
            abstract_mesh,
            device_assignment[cur_device_idx:new_device_idx],
        )
        topology[mesh_name] = new_mesh
        lowered_mesh_to_compiled_mesh[mesh] = new_mesh
        cur_device_idx = new_device_idx

      flat_out_shardings = jax.tree.leaves(
          self.function_named_shardings.output_specs
      )
      flat_out_shardings = [
          jax.sharding.NamedSharding(
              lowered_mesh_to_compiled_mesh[s.mesh], s.spec, s.memory_kind  # type: ignore
          )
          for s in flat_out_shardings
      ]
      in_shardings = jax.tree.map(
          lambda s: jax.sharding.NamedSharding(
              lowered_mesh_to_compiled_mesh[s.mesh], s.spec, s.memory_kind  # type: ignore
          ),
          self.function_named_shardings.input_specs,
      )

    # Clone the IFRT IR module to avoid modifying in place the IFRT IR module.
    # This is necessary so that lowered.as_text(dialect='ifrt') returns the
    # IFRT IR module even after compilation.
    compiled_ifrt_module = jaxlib_mpmd.clone_mlir_module(
        self.partitioning_result.ifrt_ir_module
    )
    # TODO: b/424385447 - Add compilation once IFRT IR compilation is OSSed.
    program_executable = None
    executable = MpmdExecutable(
        program_executable,
        module_ir=compiled_ifrt_module,
        func_name=self.name,
        flat_in_avals=flat_in_avals,
        out_avals=flat_out_avals,
        in_shardings=in_shardings,
        flat_out_shardings=flat_out_shardings,
        kept_inputs_indices=self.kept_inputs_indices,
        donated_inputs_indices=set(self.donate_argnums),
        topology=topology,
    )
    return MpmdCompiled(
        executable,
        self.args_info,
        self.output_tree,
        self.no_kwargs,
    )

  def _get_compile_options(
      self,
      compiler_options: (
          stages.CompilerOptions | mpmd_types.MeshToCompileOptions | None
      ) = None,
  ) -> dict[str, jax.stages.CompilerOptions]:
    option_overrides = {}
    if compiler_options:
      if any(isinstance(v, dict) for v in compiler_options.values()):
        if not all(isinstance(v, dict) for v in compiler_options.values()):
          raise ValueError(
              '`compiler_options` must either be a dict of CompilerOptions or a'
              ' single CompilerOptions object.'
          )
        bad_mesh_names = set(compiler_options) - set(self.topology)
        if bad_mesh_names:
          raise ValueError(
              f'Received compiler_options for mesh: {",".join(bad_mesh_names)},'
              ' which is not defined by the topology.'
          )
      else:
        compiler_options = {k: compiler_options for k in self.topology}
      compiler_options = cast(
          mpmd_types.MeshToCompileOptions, compiler_options
      )  # for pytype.
      for mesh_name, env_option_overrides in compiler_options.items():
        option_overrides[mesh_name] = list(env_option_overrides.items())
    return ifrt_mpmd_py.get_compile_options(
        self.partitioning_result.ifrt_ir_module, option_overrides
    )

  def compile(
      self,
      compiler_options: (
          stages.CompilerOptions | mpmd_types.MeshToCompileOptions | None
      ) = None,
      device_assignment=None,
  ) -> MpmdCompiled:
    """See base class.

    Args:
      compiler_options: An optional set of compiler options. If a single
        CompilerOptions object is provided, then we use that to override for
        every mesh. If a dict is provided, then every mesh name must be part of
        the mpmd topology and if the user doesn't set compiler options for a
        given mesh, the compiler will use defaults.
      device_assignment: An optional sequence of devices to use for compilation.
        This argument can be used to compile a lowered MPMD computation for
        different devices.

    Returns:
      A `jax.stages.Compiled` object.

    Raises:
      ValueError: if the function has already been compiled.
      NotImplementedError: if the backend is not supported, or if the backend is
        MLCR and Shardy partitioner is not enabled.
    """
    available_backends = tuple(jax_backend.backends().keys())
    if 'pathways' in available_backends:
      return self._compile_pathways(compiler_options, device_assignment)
    # TODO(b/428206925): Remove sliceme once the backend is fully deprecated.
    # See go/mlcr for more details.
    if 'sliceme' in available_backends or 'mlcr' in available_backends:
      if jax.config.jax_use_shardy_partitioner:
        return self._compile_pathways(compiler_options, device_assignment)
      else:
        raise NotImplementedError(
            'MLCR backends only supports Shardy partitioning.'
            ' Please set `jax.config.jax_use_shardy_partitioner` to True.'
        )
    raise NotImplementedError(
        'MPMD functions can only be compiled through Pathways, but only the'
        f' following backends are available: {available_backends}.'
        ' Make sure you have one of the following backends available:'
        ' pathways, mlcr.'
    )

  def as_text(
      self, dialect: str | None = None, *, debug_info: bool = False
  ) -> str:
    # Return IFRT IR by default.
    if dialect is None or dialect == 'ifrt':
      return jax_mlir.module_to_string(
          self.partitioning_result.ifrt_ir_module,
          enable_debug_info=debug_info,
      )
    if dialect == 'mpmd':
      return jax_mlir.module_to_string(
          self.partitioning_result.mpmd_module,
          enable_debug_info=debug_info,
      )
    if dialect == 'stablehlo':
      return jax_mlir.module_to_string(
          self._stablehlo_mlir_module, enable_debug_info=debug_info
      )
    raise ValueError(f'Unsupported dialect: {dialect}')

  def compiler_ir(self, dialect: str | None = None) -> jax_mlir.ir.Module:
    # Return IFRT IR by default.
    if dialect is None or dialect == 'ifrt':
      return self.partitioning_result.ifrt_ir_module
    if dialect == 'mpmd':
      return self.partitioning_result.mpmd_module
    if dialect == 'stablehlo':
      return self._stablehlo_mlir_module
    raise ValueError(f'Unsupported dialect: {dialect}')


@functools.lru_cache(maxsize=1024)
def _verify_shardings_are_equivalent(
    func_name: str,
    expected_shardings: tuple[jax.sharding.Sharding, ...],
    input_pytree_paths: tuple[jax.tree_util.KeyPath, ...],
    arg_shardings_ndims: tuple[tuple[int, jax.sharding.Sharding, int], ...],
    topology: tuple[tuple[str, jax.sharding.Mesh], ...],
) -> None:
  """Raises a `ValueError` if the args do not have the expected shardings."""
  errors = []
  mesh_to_name = {mesh: name for name, mesh in topology}

  def _err_str(sharding: jax.sharding.Sharding) -> str:
    mesh = getattr(sharding, 'mesh', None)
    if mesh is None:
      return f'{sharding} on unknown mesh'
    mesh_name = mesh_to_name.get(
        mesh, f'unknown mesh {mesh} on devices {mesh.devices}'
    )
    return f'{sharding} on "{mesh_name}"'

  for idx, (expected_sharding, (orig_idx, sharding, ndim), path) in enumerate(
      zip(
          expected_shardings,
          arg_shardings_ndims,
          input_pytree_paths,
          strict=True,
      )
  ):
    if not sharding.is_equivalent_to(expected_sharding, ndim):
      errors.append(
          f'arg{idx} (idx before removing unused args: {orig_idx}) at path'
          f' {jax.tree_util.keystr(path)}:'
          f'\n\t\treceived: {_err_str(sharding)}'
          f'\n\t\texpected: {_err_str(expected_sharding)}.'
      )
  if errors:
    str_errors = '\n\n\t'.join(errors)
    raise ValueError(
        f'MPMD Executable "{func_name}" with {len(input_pytree_paths)} kept'
        ' flat inputs was called with inputs with sharding(s) that do not match'
        ' the sharding(s) it was compiled with (ignoring axes of size 1). '
        f'Errors:\n\t{str_errors}'
    )
