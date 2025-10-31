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

"""MPMD JAX ops."""

from collections.abc import Callable, Sequence
import functools
import inspect
from typing import Any, TypeVar

import jax
from jax import api_util
from jax import numpy as jnp
from jax import tree
from jax import tree_util
from jax._src import util
from jax._src.interpreters import ad as internal_ad
from jax._src.interpreters import batching as internal_batching
from jax._src.interpreters import partial_eval as internal_pe
import jax.extend as jex
from jax.extend import linear_util as lu
from jax.extend import source_info_util as siu
from jax.extend.core import primitives
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import func as func_dialect
from jax.extend.mlir.dialects import mpmd
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir as jax_mlir
from jax.interpreters import partial_eval as pe
import jaxtyping

from shardy.integrations.python.jax.mpmd import utils


PyTree = jaxtyping.PyTree
X = TypeVar('X')
Y = TypeVar('Y')


def _infer_argnums(
    fun: Callable[..., Any],
    argnames: Sequence[str] | None,
) -> tuple[int, ...]:
  """Infer static argnums from static argnames."""
  if argnames is None:
    return ()

  try:
    sig = inspect.signature(fun)
  except (ValueError, TypeError):
    return ()

  parameters = sig.parameters
  argnums = tuple(
      i
      for i, (k, param) in enumerate(parameters.items())
      if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k in argnames
  )

  return argnums


# ===----------------------------------------------------------------------=== #
# mpmd.named_computation
# ===----------------------------------------------------------------------=== #


def _named_computation(
    fn: Callable[..., X],
    *,
    name: str,
    transpose_count: int,
    static_argnames: Sequence[str],
) -> Callable[..., X]:
  """Wraps a function with a named_computation primitive. See also API docs."""

  @functools.wraps(fn)
  def wrapped_fn(*args, **kwargs):
    if static_argnames:
      static_argnums = _infer_argnums(fn, static_argnames)
      static_args_dict = {i: args[i] for i in static_argnums}
      dyn_args = tuple(
          args[i] for i in range(len(args)) if i not in static_argnums
      )
      static_kwargs = {k: v for k, v in kwargs.items() if k in static_argnames}
      dyn_kwargs = {k: v for k, v in kwargs.items() if k not in static_argnames}

      def partial_fn_with_dynamic_args(*dyn_args_inner, **dyn_kwargs_inner):
        full_args = []
        dyn_idx = 0
        for i in range(len(args)):
          if i in static_args_dict:
            full_args.append(static_args_dict[i])
          else:
            full_args.append(dyn_args_inner[dyn_idx])
            dyn_idx += 1
        if len(full_args) != len(args):
          raise ValueError(
              f'Expected {len(args)} arguments, but got {len(full_args)} after'
              f' inserting {static_args_dict}'
          )
        return fn(*full_args, **static_kwargs, **dyn_kwargs_inner)

    else:
      dyn_args = args
      dyn_kwargs = kwargs
      static_argnums = ()
      partial_fn_with_dynamic_args = fn

    fun = lu.wrap_init(
        partial_fn_with_dynamic_args,
        debug_info=api_util.debug_info(
            'mpmd.named_computation',
            partial_fn_with_dynamic_args,
            dyn_args,
            dyn_kwargs,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
        ),
    )
    flat_args, in_tree = tree_util.tree_flatten((dyn_args, dyn_kwargs))
    flat_fun, out_tree = api_util.flatten_fun(fun, in_tree)
    out_flat = named_computation_p.bind(
        flat_fun, *flat_args, name=name, transpose_count=transpose_count
    )
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped_fn


def named_computation(
    fn: Callable[..., X] | None = None,
    *,
    name: str,
    static_argnames: Sequence[str] | str | None = None,
) -> Callable[..., X]:
  """Returns a callable which has been given a name.

  This operation enables MPMD partitioning in PartIR as it breaks a JAX
  function into multiple sub-computations which can be assigned to different
  meshes.

  Example: We can define the following partitioned function:

    def f(x: Array, y: Array) -> Tuple[Array, Array]:
      f1 = partir.named_computation(lambda a: a + a, name="foo")(x)
      f2_0, f2_1 = partir.named_computation(lambda b: b, b @ y, name="bar")(f1)
      return f2_0, f2_1

  The array `x` is the input of the first named_computation (called `foo`),
  which returns the result of adding `x` to itself. The result of this op, `f1`,
  is then passed to the second named_computation (called `bar`), which
  returns the input tensor and result of a matmul between the named_computations
  input and a free variable `y` (w.r.t. the named_computation's function).

  When this function is compiled with partir.mpmd_jit, the user can assign each
  named_computation to a different SPMD mesh, meaning that each
  named_computation becomes a different MPMD program. If both named_computations
  are assigned to the same SPMD mesh, then together they will form a single
  MPMD program.

  Args:
    fn: the computation executed when the named_computation is scheduled.
    name: the name of the named_computation, which can be used to assign it to
      an SPMD mesh.
    static_argnames: the names of the arguments that are not to be traced. See
      jax.jit for more details. We'll infer static_argnums from this.

  Returns:
    A callable that returns the results of executing `fn`.

  Raises:
    ValueError: if name is a reserved name.
  """
  if name == 'main':
    raise ValueError(
        'named_computations are not allowed to have "main" as their name'
    )
  if name.startswith('partir.'):
    raise ValueError(
        'computation names that start with `partir.` are reserved for the'
        ' compiler.'
    )

  if fn is None:
    return lambda fn: named_computation(
        fn, name=name, static_argnames=static_argnames
    )

  if static_argnames is None:
    static_argnames = ()
  elif isinstance(static_argnames, str):
    static_argnames = (static_argnames,)

  return _named_computation(
      fn, name=name, transpose_count=0, static_argnames=static_argnames
  )


def named_computation_partir_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args: Sequence[ir.Value],
    name: str,
    transpose_count: int,
    call_jaxpr: jex.core.Jaxpr,
) -> Sequence[ir.Value]:
  """Lowers a named_computation op to its PartIR:MPMD MLIR counterpart."""
  if not call_jaxpr.outvars:
    return []
  mpmd.register_dialect(ctx.module_context.context)

  input_types = util.safe_map(jax_mlir.aval_to_ir_types, ctx.avals_in)
  output_types = util.safe_map(jax_mlir.aval_to_ir_types, ctx.avals_out)
  flat_input_types, _ = tree_util.tree_flatten(input_types)
  flat_output_types, _ = tree_util.tree_flatten(output_types)

  # There is no facility to construct the custom attribute other than parsing.
  origin = ir.Attribute.parse(f'#mpmd.user_origin<"{name}"({transpose_count})>')

  named_comp_op = mpmd.NamedComputationOp(
      flat_output_types, jax_mlir.flatten_ir_values(args), origin
  )

  block = named_comp_op.region.blocks.append(*flat_input_types)
  # Passing empty list as constants, as we expect call_jaxpr.constvars to be
  # empty.
  constants = []
  with ir.InsertionPoint(block):
    outs, tokens = jax_mlir.jaxpr_subcomp(
        ctx.module_context,
        call_jaxpr,
        ctx.name_stack.extend(name),
        ctx.tokens_in,
        constants,
        *block.arguments,
        dim_var_values=ctx.dim_var_values,
        const_lowering=ctx.const_lowering,
    )
    ctx.set_tokens_out(tokens)
    # The return op expects a single flat list with all outputs.
    mpmd.ReturnOp(jax_mlir.flatten_ir_values(outs))
  return named_comp_op.results


def _named_computation_default_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args: Sequence[ir.Value],
    name: str,
    transpose_count: int | None,
    call_jaxpr: jex.core.Jaxpr,
):
  del transpose_count  # Unused for the ordinary jit-based lowering.
  call_lowering = functools.partial(
      jax_mlir.core_call_lowering, name=f'partir_named_computation_call_{name}'
  )
  return call_lowering(ctx, *args, call_jaxpr=call_jaxpr)


def _register_named_computation_primitive():
  """Registers named_computation primitive and a JAX CallPrimitive."""
  primitive = jax.core.CallPrimitive('named_computation')
  # Makes it possible to execute eagerly.
  primitive.def_impl(jax.core.call_impl)

  def custom_call_transpose(params, *rest, primitive=primitive):
    new_params = dict(params)
    new_params['transpose_count'] = new_params['transpose_count'] + 1
    return internal_ad.call_transpose(primitive, new_params, *rest)

  ad.primitive_transposes[primitive] = custom_call_transpose

  # Allows JAX to remove unused_args from the primitive when
  # `keep_unused = False`.
  pe.dce_rules[primitive] = pe.dce_jaxpr_call_rule
  # Introduces the rule to lower this primitive as a call_primitive, so that
  # jax.jit users can still lower named_computations.
  jax_mlir.register_lowering(primitive, _named_computation_default_lowering)
  # Allows a jax.remat(mpmd.named_computation(...))
  pe.partial_eval_jaxpr_custom_rules[primitive] = (
      pe.partial_eval_jaxpr_custom_rules[primitives.call_p]
  )
  return primitive


named_computation_p = _register_named_computation_primitive()


# ===----------------------------------------------------------------------=== #
# mpmd.named_ tensor
# ===----------------------------------------------------------------------=== #


def named_tensor(tensor: jax.Array, name: str) -> jax.Array:
  """Names a given tensor.

  Users can then reference this tensor in the mpmd.jit assignment mapping
  to assign this named tensor a a specific mesh.

  NOTE: this function is different than partir.tag in that tags can be used
  inside/outside of a NamedComputation (while named_tensor is only outside),
  and is used for partitioning (while named_tensor is for assigning it to a
  given mesh).

  Args:
    tensor: The tensor to assign.
    name: The name the tensor.

  Returns:
    The assigned `tensor`.
  """
  return named_tensor_p.bind(tensor, name=name)


def named_tensor_partir_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    operand: ir.Value,
    *,
    name: str,
) -> Sequence[ir.Value]:
  """Jax MLIR lowering rule for named_tensor for PartIR backend."""
  mpmd.register_dialect(ctx.module_context.context)
  return mpmd.NamedTensorOp(
      operand,
      ir.StringAttr.get(name),
  ).results


def _named_tensor_default_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    operand: ir.Value,
    *,
    name: str,
) -> Sequence[ir.Value]:
  """Jax MLIR lowering rule for named_tensor."""
  del ctx, name
  return [operand]


def _named_tensor_linear(transpose, operand, *, name: str):
  del operand
  return [
      named_tensor_p.bind(
          transpose,
          name=name,
      )
  ]


def _register_named_tensor_primitive():
  """Registers the `partir_mpmd.named_tensor` op to JAX.

  Returns:
    The PartIR named_tensor op.
  """
  primitive = jex.core.Primitive('named_tensor')
  tracer = lambda arr, name: arr
  # Makes it possible to execute eagerly.
  primitive.def_impl(tracer)
  # Makes it possible to execute using eval_shape.
  primitive.def_abstract_eval(tracer)

  # Tells JAX how to lower this by default under jax.jit.
  jax_mlir.register_lowering(primitive, _named_tensor_default_lowering)
  # Makes it possible to take the transpose.
  ad.deflinear2(primitive, _named_tensor_linear)
  # Allow named_tensors to be vmapped.
  batching.defvectorized(primitive)

  return primitive


named_tensor_p = _register_named_tensor_primitive()


# ===----------------------------------------------------------------------=== #
# mpmd.call
# ===----------------------------------------------------------------------=== #


def call_mpmd_jit_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args: Sequence[ir.Value],
    call_jaxpr: jex.core.ClosedJaxpr,
    call_counter: int | None,
    orig_callable: Callable[..., Any],
) -> Sequence[ir.Value]:
  """Lowers a call primitive when mpmd.jit'ing a function."""
  mpmd.register_dialect(ctx.module_context.context)

  name = utils.get_func_name(orig_callable, prefix='shardy_mpmd')
  effects = list(ctx.tokens_in.effects())

  # TODO(b/434270189): fix for hoisted constant args.
  num_const_args = 0
  in_avals = call_jaxpr.in_avals
  input_types = util.safe_map(jax_mlir.aval_to_ir_types, ctx.avals_in)
  output_types = util.safe_map(jax_mlir.aval_to_ir_types, ctx.avals_out)

  # TODO(jupvfranco): Consider memoizing the lowering function instead of
  # caching in the context (similar to `_call_get_cached_jaxpr` below).
  # TODO(jupvfranco): Consider using effects in the key too.
  key = (call_jaxpr, tuple(input_types), tuple(output_types))
  if key in ctx.module_context.cached_primitive_lowerings:
    func_declaration = ctx.module_context.cached_primitive_lowerings[key]
  else:
    func_declaration = jax_mlir.lower_jaxpr_to_fun(
        ctx.module_context,
        name,
        call_jaxpr,
        effects,
        num_const_args=num_const_args,
        in_avals=in_avals,
    )
    ctx.module_context.cached_primitive_lowerings[key] = func_declaration

  flat_output_types, _ = tree_util.tree_flatten(output_types)
  call_op = mpmd.CallOp(
      flat_output_types,
      jax_mlir.flatten_ir_values(args),
      ir.FlatSymbolRefAttr.get(func_declaration.sym_name.value),
  )

  if call_counter is not None:
    call_op.operation.attributes['call_counter'] = ir.IntegerAttr.get(
        ir.IntegerType.get_unsigned(32), call_counter
    )
  return call_op.results


def _call_default_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args: Sequence[ir.Value],
    call_jaxpr: jex.core.ClosedJaxpr,
    call_counter: int | None,
    orig_callable: Callable[..., Any],
) -> Sequence[ir.Value]:
  """Default Lowering rule used when jit'ing a function."""

  del call_counter
  call_lowering = functools.partial(
      jax_mlir.core_call_lowering,
      name=utils.get_func_name(orig_callable, prefix='shardy_mpmd'),
  )
  return call_lowering(ctx, *args, call_jaxpr=call_jaxpr)


@util.weakref_lru_cache
def _call_get_cached_jaxpr(fn, in_avals, in_tree):
  """Returns the (potentially cached) jaxpr of a function."""
  fun = lu.wrap_init(fn)
  flat_fun, out_tree = api_util.flatten_fun(fun, in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  closed_jaxpr = jex.core.ClosedJaxpr(
      internal_pe.convert_constvars_jaxpr(jaxpr), ()
  )
  return closed_jaxpr, consts, out_tree


def _call_abstract_eval(*args, call_jaxpr, **kwargs):
  """Abstract evaluation rule for the call op."""
  del args, kwargs
  return call_jaxpr.out_avals, call_jaxpr.effects


def _call_impl(*args, call_jaxpr, **kwargs):
  """Implementation rule for the call op."""
  del kwargs
  return jex.core.jaxpr_as_fun(call_jaxpr)(*args)


def _register_call_primitive():
  """Registers the `mpmd.call` op to JAX."""
  primitive = jex.core.Primitive('mpmd_call')
  # Note: we do not set call_primitive = True, as the calling convention for
  # transposition we use is the generic one. See `backward_pass()` in ad.py.
  primitive.multiple_results = True
  primitive.def_impl(_call_impl)
  # Introduces the rule to lower this primitive as a call_primitive, so that
  # jax.jit users can still lower mpmd.calls.
  jax_mlir.register_lowering(primitive, _call_default_lowering)
  pe.dce_rules[primitive] = pe.dce_jaxpr_closed_call_rule
  primitive.def_effectful_abstract_eval(_call_abstract_eval)

  def _call_jvp_rule(
      primals,
      tangents,
      *,
      call_jaxpr,
      call_counter,
      orig_callable,
      primitive=primitive,
      **other_kwargs,
  ):
    nonzero_tangents = [not isinstance(t, ad.Zero) for t in tangents]
    tangents = [t for t in tangents if not isinstance(t, ad.Zero)]
    jvp_jaxpr, nonzero_output_tangents = internal_ad.jvp_jaxpr(
        call_jaxpr, nonzero_tangents, instantiate=False
    )
    out_flat = primitive.bind(
        *primals,
        *tangents,
        call_jaxpr=jvp_jaxpr,
        call_counter=call_counter,
        orig_callable=orig_callable,
        **other_kwargs,
    )
    # Split the primals and the tangents.
    primals_out, tangents_out = util.split_list(
        out_flat, [len(call_jaxpr.jaxpr.outvars)]
    )
    # Some tangents may be zero, so we need to add zeros for them.
    tangents_out_it = iter(tangents_out)
    return primals_out, [
        next(tangents_out_it) if nz else ad.Zero(aval)
        for nz, aval in zip(nonzero_output_tangents, call_jaxpr.out_avals)
    ]

  ad.primitive_jvps[primitive] = _call_jvp_rule

  @lu.transformation2
  def hashable_partial(f, *args):
    return f(*args)

  @lu.cache
  def _call_transpose_trace(fun, in_avals):
    transpose_jaxpr, _, consts, *_ = pe.trace_to_jaxpr_dynamic(fun, in_avals)
    transpose_jaxpr = jex.core.ClosedJaxpr(transpose_jaxpr, consts)
    return transpose_jaxpr

  def _call_transpose(
      cts_in,
      *primals_in,
      call_jaxpr,
      call_counter,
      orig_callable,
      primitive=primitive,
  ):
    body = lu.wrap_init(internal_ad.closed_backward_pass)
    body = hashable_partial(body, call_jaxpr, False)
    primals_and_nz_cts_in, in_treedef = jax.tree.flatten((primals_in, cts_in))
    body, cts_out_treedef_thunk = api_util.flatten_fun_nokwargs(
        body, in_treedef
    )

    global_cts_in_avals = tuple(
        api_util.shaped_abstractify(ct) for ct in primals_and_nz_cts_in
    )
    transpose_jaxpr = _call_transpose_trace(body, global_cts_in_avals)
    cts_out_treedef = cts_out_treedef_thunk()
    nz_cts_out = primitive.bind(
        *primals_and_nz_cts_in,
        call_jaxpr=transpose_jaxpr,
        call_counter=call_counter,
        orig_callable=orig_callable,
    )
    return jax.tree.unflatten(cts_out_treedef, nz_cts_out)

  ad.primitive_transposes[primitive] = _call_transpose

  def _call_partial_eval(
      trace,
      *in_tracers,
      call_jaxpr,
      call_counter,
      orig_callable,
      primitive=primitive,
  ):
    in_pvals = [t.pval for t in in_tracers]
    known_ins = tuple(pv.is_known() for pv in in_pvals)
    unknown_ins = tuple(not k for k in known_ins)
    known_jaxpr, unknown_jaxpr, unknown_outs, res_avals = (
        internal_pe.partial_eval_jaxpr_nounits(
            call_jaxpr, unknown_ins, instantiate=False
        )
    )
    num_residuals = len(res_avals)
    num_out_primals = len(known_jaxpr.out_avals) - num_residuals
    in_fwd = internal_pe._jaxpr_forwarding(known_jaxpr.jaxpr)  # pylint:disable=protected-access
    # Do not forward primal outputs at all, we only care about residuals.
    in_fwd = [None] * num_out_primals + in_fwd[num_out_primals:]

    # Compute which residuals are just primal outputs.
    out_vars, res_vars = util.split_list(
        known_jaxpr.jaxpr.outvars, [num_out_primals]
    )
    idx_map = {id(v): i for i, v in enumerate(out_vars)}
    out_fwd = [None] * num_out_primals + [idx_map.get(id(v)) for v in res_vars]

    # Bind known things to our primitive.
    keep = [f1 is None and f2 is None for f1, f2 in zip(in_fwd, out_fwd)]
    known_jaxpr = internal_pe.prune_closed_jaxpr_outputs(known_jaxpr, keep)
    del keep, num_out_primals

    known_params = dict(
        call_jaxpr=known_jaxpr,
        call_counter=call_counter,
        orig_callable=orig_callable,
    )

    known_inputs = [pv.get_known() for pv in in_pvals if pv.is_known()]
    all_known_outs = primitive.bind(*known_inputs, **known_params)
    all_known_outs = util.subs_list2(
        in_fwd, out_fwd, known_inputs, all_known_outs, all_known_outs
    )
    del known_inputs

    known_out_vals, residual_vals = util.split_list(
        all_known_outs, [len(all_known_outs) - num_residuals]
    )
    residual_tracers = map(trace.new_instantiated_const, residual_vals)

    # The convention of partial_eval_jaxpr_nounits is to place residual binders
    # at the front of the jaxpr produced, but here we move them to the back
    # following the residual-inputs-last convention. I do not think this is a
    # load-bearing decision, just following conventions from elsewhere (mjit).
    unknown_jaxpr = internal_pe.move_binders_to_back(
        unknown_jaxpr, [True] * num_residuals + [False] * sum(unknown_ins)
    )
    # Prepare unknown tracers
    unknown_params = dict(
        call_jaxpr=unknown_jaxpr,
        call_counter=call_counter,
        orig_callable=orig_callable,
    )
    unknown_tracers_in = [t for t in in_tracers if not t.pval.is_known()]
    unknown_out_avals = unknown_jaxpr.out_avals
    unknown_tracers_out = [
        pe.JaxprTracer(trace, pe.PartialVal.unknown(aval), None)
        for aval in unknown_out_avals
    ]
    eqn = internal_pe.new_eqn_recipe(
        trace,
        (*unknown_tracers_in, *residual_tracers),
        unknown_tracers_out,
        primitive,
        unknown_params,
        unknown_jaxpr.effects,
        siu.current(),
    )
    for t in unknown_tracers_out:
      t.recipe = eqn
    return util.merge_lists(unknown_outs, known_out_vals, unknown_tracers_out)

  pe.custom_partial_eval_rules[primitive] = _call_partial_eval

  def _call_vmap(axis_data, args, dims, *, call_jaxpr, **params):
    jaxpr_batched_, out_batched = internal_batching.batch_jaxpr_axes(
        call_jaxpr,
        axis_data,
        dims,
        [internal_batching.zero_if_mapped] * len(call_jaxpr.jaxpr.outvars),
    )
    jaxpr_batched, consts = jaxpr_batched_.jaxpr, jaxpr_batched_.consts
    if consts:
      jaxpr_batched = internal_pe.convert_constvars_jaxpr(jaxpr_batched)
    out_dims = [0 if b else None for b in out_batched]
    return (
        primitive.bind(
            *consts,
            *args,
            call_jaxpr=internal_pe.close_jaxpr(jaxpr_batched),
            **params,
        ),
        out_dims,
    )

  batching.fancy_primitive_batchers[primitive] = _call_vmap

  return primitive


call_p = _register_call_primitive()


def call(
    fn: Callable[..., X],
    *,
    call_counter: int | None = None,
) -> Callable[..., X]:
  """A custom call op.

  Args:
   fn: the Jax function being called.
   call_counter: a user defined call_counter, which can be used to annotate the
     call operation with how many times a function was called.

  Returns:
    The results of invoking `fn`.

  Raises:
    ValueError: if `call_counter` is negative.
  """

  if call_counter is not None and call_counter < 0:
    raise ValueError(
        f'call_counter must be non-negative, got {call_counter=} instead.'
    )

  @functools.wraps(fn)
  def wrapped_fn(*args, **kwargs):
    flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
    in_avals = tuple(api_util.shaped_abstractify(arg) for arg in flat_args)
    jaxpr, consts, out_tree = _call_get_cached_jaxpr(fn, in_avals, in_tree)
    out_flat = call_p.bind(
        *consts,
        *flat_args,
        call_jaxpr=jaxpr,
        call_counter=call_counter,
        orig_callable=fn,
    )
    return tree_util.tree_unflatten(out_tree(), out_flat)

  return wrapped_fn


# ===----------------------------------------------------------------------=== #
# mpmd.broadcast
# ===----------------------------------------------------------------------=== #


def broadcast(arr: jax.Array) -> jax.Array:
  """A broadcast op.

  The broadcast op indicates to PartIR:MPMD that the given array may be
  broadcasted to other meshes. This means that if `arr` is originally allocated
  in a specific mesh `m` and is used in e.g. meshes `m1` and `m2`, then the
  compiler may introduce transfers from `m` to `m1` and `m2`, or alternatively
  copy the producer of `arr` in `m1` and `m2`, if beneficial (e.g., if the array
  is a constant).

  Args:
    arr: the array that may be broadcasted to other meshes.

  Returns:
    The array that may be broadcasted to other meshes.
  """
  return broadcast_p.bind(arr)


def broadcast_mpmd_lowering(
    ctx: jax_mlir.LoweringRuleContext, operand: ir.Value
) -> Sequence[ir.Value]:
  """Jax MLIR lowering rule for broadcast for PartIR:MPMD backend."""
  mpmd.register_dialect(ctx.module_context.context)
  return mpmd.BroadcastOp(
      operand,
  ).results


def _broadcast_default_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    operand: ir.Value,
) -> Sequence[ir.Value]:
  """Jax MLIR lowering rule for broadcast."""
  del ctx
  return [operand]


def _broadcast_linear(args, _):
  """Transpose of broadcast is reduce."""
  return [reduce(args)]


def _register_broadcast_primitive():
  """Registers the `broadcast` op to JAX.

  Returns:
    The PartIR:MPMD broadcast primitive.
  """
  primitive = jex.core.Primitive('broadcast')
  tracer = lambda arr: arr
  # Makes it possible to execute eagerly.
  primitive.def_impl(tracer)
  # Makes it possible to execute using eval_shape.
  primitive.def_abstract_eval(tracer)

  # Tells JAX how to lower this by default under jax.jit.
  jax_mlir.register_lowering(primitive, _broadcast_default_lowering)
  # Makes it possible to take the transpose.
  ad.deflinear2(primitive, _broadcast_linear)
  # Allow broadcasts to be vmapped.
  batching.defvectorized(primitive)

  return primitive


broadcast_p = _register_broadcast_primitive()


# ===----------------------------------------------------------------------=== #
# mpmd.reduce
# ===----------------------------------------------------------------------=== #


def reduce(arr: jax.Array) -> jax.Array:
  """Annotates that `arr` may be the result of a cross-mesh reduction.

  Users may annotate an array with `mpmd.reduce(arr)` to denote it to be the
  result of a cross-mesh reduction, if necessary. If no extra communication is
  required, this primitive is a no-op.

  We consider to be a cross-mesh reduction a *computation* that:
    - all inputs and outputs have the same shape and dtype,
    - takes multiple arrays as input and produces a single array, and
    - the different inputs live on different meshes.

  Such computations include:
    1) A binary tree of any depth in which every node is one of the following
    binary ops: `add`, `max`, `min`, `mul`, `or`, and `and`.
    2) A chain of unary ops that consume the array produced by a computation as
    described in (1).
    3) A non-MPMD reduce operation (`add`, `max`, `min`, `mul`, `or`, and `and`)
    of a concatenate op (with matching dimensions).

  Annotating the result of such computations with an `mpmd.reduce` signals
  mpmd.jit that it may rewrite the program to introduce transfers on
  computation's operands, from their meshes to the mesh where `result` is used.

  Concretely, these computations will be rewritten as follows:
    1) mpmd.reduce(add(add(x, y), z))
            ~~> add(add(transfer(x), transfer(y)), transfer(z))
    2) mpmd.reduce(sqrt(add(x, y)))
            ~~> sqrt(add(transfer(x), transfer(y)))
    3) mpmd.reduce(hlo.reduce<add, dim=d>(concatenate<dim=d>(x, y, z)))
            ~~> add(add(transfer(x), transfer(y)), transfer(z))

  Additionally, nesting multiple mpmd.reduce ops has the same effect of using a
  single mpmd.reduce.

  An alternative to using mpmd.reduce is to enable automatic inference of
  cross-mesh reduction, using the absl flag:
    `--partir_mpmd_infer_cross_mesh_reductions`

  Args:
    arr: the array that may result from a reduction.

  Returns:
    The original array.
  """
  return reduce_p.bind(arr)


def reduce_lowering(
    ctx: jax_mlir.LoweringRuleContext, tensor: ir.Value
) -> Sequence[ir.Value]:
  """Jax MLIR lowering rule for reduce for PartIR:MPMD backend."""
  mpmd.register_dialect(ctx.module_context.context)
  return mpmd.ReduceOp([tensor]).results


def _reduce_default_lowering(
    ctx: jax_mlir.LoweringRuleContext, tensor: ir.Value
) -> Sequence[ir.Value]:
  """Jax MLIR lowering rule for reduce."""
  del ctx
  return [tensor]


def _reduce_linear(args, _):
  """Transpose of reduce is broadcast."""
  return [broadcast(args)]


def _register_reduce_primitive():
  """Registers the `reduce` op to JAX.

  Returns:
    The PartIR:MPMD reduce primitive.
  """
  primitive = jex.core.Primitive('reduce')
  tracer = lambda arg: arg
  # Makes it possible to execute eagerly.
  primitive.def_impl(tracer)
  # Makes it possible to execute using eval_shape.
  primitive.def_abstract_eval(tracer)

  # Tells JAX how to lower this by default under jax.jit.
  jax_mlir.register_lowering(primitive, _reduce_default_lowering)
  # Makes it possible to take the transpose.
  ad.deflinear2(primitive, _reduce_linear)
  # Allow reduce to be vmapped.
  batching.defvectorized(primitive)

  return primitive


reduce_p = _register_reduce_primitive()


# ===----------------------------------------------------------------------=== #
# mpmd.for_i_loop
# ===----------------------------------------------------------------------=== #


def _for_i_impl(
    *args,
    call_jaxpr,
    num_iterations: int,
    carried_arguments_start: int,
    **kwargs,
):
  """Defines implementation rule for the call op."""
  del kwargs
  consts = args[0:carried_arguments_start]
  xs = args
  for i in range(num_iterations):
    results = jex.core.jaxpr_as_fun(call_jaxpr)(*xs, jnp.uint32(i))
    xs = (*consts, *results)
  return xs[carried_arguments_start:]


def fori_loop_mpmd_jit_lowering(
    ctx: jax_mlir.LoweringRuleContext,
    *args: Sequence[ir.Value],
    call_jaxpr: jex.core.ClosedJaxpr,
    num_iterations: int,
    carried_arguments_start: int,
    func_name: str,
    wrap_with_mpmd_call: bool,
) -> Sequence[ir.Value]:
  """Lowers a fori_loop primitive when mpmd.jit'ing a function.

  `args` include [const_0, ..., const_n, carried_arg_0, ..., carried_arg_m],
  such that `n + 1 == carried_arguments_start` and because the `partir_mpmd.for`
  primitive requires the number of operands to match the number of results, we
  make the loop return all the constants too, even though the respective results
  will be unused. I.e., we lower the `fori` loop to:

   ```mlir
   %consts = const_0, ..., const_n
   %carried_args = carried_arg_0, ..., carried_arg_m
   partir_mpmd.for (%consts, %carried_args) {iterations = num_iterations : ui32}
    (%consts_block_args, %carried_block_args, %index: tensor<ui32>) {
      // %call will return m+1 values which are the carried values of the loop.
      %call:{m+1} = mpmd.call @funcame(%consts_block_args,
                                       %carried_block_args,
                                       %index)
      return %consts_block_args, %call#0, ..., %call#m
    }
  ```

  Args:
    ctx: the lowering context.
    *args: the arguments of the fori_loop primitive.
    call_jaxpr: the jaxpr of the body of the loop.
    num_iterations: the number of iterations of the loop.
    carried_arguments_start: the index of the first argument that is
      loop-carried.
    func_name: the name of the function to be called in the loop body.
    wrap_with_mpmd_call: whether to wrap the body of the loop with an mpmd.call.
      This is necessary for pipeline scheduling which operates on mpmd.calls and
      for mesh inference. When set to false, we use a function call from the
      `func` dialect. TODO: b/357831621: for now, we lower the body of the loop
      as an mpmd.call. This is so that we can unroll the loop and make sure that
      mesh inference and pipeline scheduling will still work. Though, once we
      fully support the for-loop in our passes, we can remove this code/option.

  Returns:
    The results of the fori_loop ignoring any result that is not part of the
    carried values, i.e., constants.
  """
  mpmd.register_dialect(ctx.module_context.context)

  # TODO(b/434270189): fix for hoisted constant args.
  num_const_args = 0
  in_avals = call_jaxpr.in_avals
  effects = list(ctx.tokens_in.effects())
  func_declaration = jax_mlir.lower_jaxpr_to_fun(
      ctx.module_context,
      func_name,
      call_jaxpr,
      effects,
      num_const_args=num_const_args,
      in_avals=in_avals,
  )

  input_types = util.safe_map(jax_mlir.aval_to_ir_types, ctx.avals_in)
  flat_input_types, _ = tree_util.tree_flatten(input_types)
  const_types = flat_input_types[0:carried_arguments_start]
  output_types = util.safe_map(jax_mlir.aval_to_ir_types, ctx.avals_out)
  flat_output_types, _ = tree_util.tree_flatten(output_types)
  for_loop = mpmd.ForOp(
      const_types + flat_output_types,
      jax_mlir.flatten_ir_values(args),
      num_iterations,
      unroll_factor=num_iterations,
  )
  # `args` does not include the index argument, so we explicitly add it as a
  # block argument of the loop (and operand of the call).
  # Creates a block with one more argument than the op has operands to account
  # for the index argument, which appears last.
  index_type = ir.RankedTensorType.get((), ir.IntegerType.get_unsigned(32))
  block = for_loop.region.blocks.append(*flat_input_types, index_type)
  # Return the constants so that the number of results matches the number of
  # operands.
  constants_to_return = block.arguments[0:carried_arguments_start]
  with ir.InsertionPoint(block):
    if wrap_with_mpmd_call:
      call_op = mpmd.CallOp(
          flat_output_types,
          block.arguments,
          ir.FlatSymbolRefAttr.get(func_declaration.sym_name.value),
      )
    else:
      call_op = func_dialect.CallOp(
          flat_output_types,
          ir.FlatSymbolRefAttr.get(func_declaration.sym_name.value),
          block.arguments,
      )
    mpmd.ReturnOp(list(constants_to_return) + list(call_op.results))
  return for_loop.results[carried_arguments_start:]


def _add_leaf_to_inputs_tree(args_tree) -> jax.tree_util.PyTreeDef:
  """Returns the arguments tree extended with kwargs and leaf trees."""
  kwargs_tree = tree.structure({})
  single_leaf_tree = tree.structure(object())
  return tree_util.treedef_tuple([
      tree_util.treedef_tuple([args_tree, single_leaf_tree]),
      kwargs_tree,
  ])


# TODO: b/357831621 - remove this indirection once we fully support the
# for-loop in our passes. When that happens, the enumerated-for should be a
# simple wrapper around the fori-loop without call ops.
def _fori_loop_implementation(
    num_iterations: int,
    body_fun: Callable[[int, X], X],
    init_val: X,
    *,
    wrap_body_with_mpmd_call: bool,
) -> X:
  """A fori_loop primitive with optional wrapping of the body with an mpmd.call."""

  flat_args, in_tree = tree_util.tree_flatten(init_val)
  # List of abstract vals *without* the index argument.
  in_avals = [api_util.shaped_abstractify(arg) for arg in flat_args]
  # Tuple of abstract vals *with* the index argument.
  in_avals_w_index = tuple(in_avals) + (
      jax.core.ShapedArray((), dtype=jnp.uint32),
  )
  # Add a leaf for the index argument, which is the second positional argument
  # of the traced function, to the tree structure.
  in_tree_w_index = _add_leaf_to_inputs_tree(in_tree)

  jaxpr, consts, out_tree = _call_get_cached_jaxpr(
      lambda v, i: body_fun(i, v), in_avals_w_index, in_tree_w_index
  )

  if in_tree != out_tree():
    raise ValueError(
        'Tree structs of init_val and the result of the fori_loop must match.'
        f' Got {in_tree} and {out_tree()}'
    )
  for i, o in zip(in_avals, jaxpr.out_avals):
    if (i.shape, i.dtype) != (o.shape, o.dtype):
      raise ValueError(
          f'Input and output avals must match. Got {i} and {o} respectively.'
      )

  out_flat = fori_loop_p.bind(
      *consts,
      *flat_args,  # Does not include the index argument.
      call_jaxpr=jaxpr,
      num_iterations=num_iterations,
      carried_arguments_start=len(consts),
      func_name=utils.get_func_name(body_fun, prefix='shardy_mpmd_'),
      wrap_with_mpmd_call=wrap_body_with_mpmd_call,
  )
  return tree_util.tree_unflatten(in_tree, out_flat)


def fori_loop(
    num_iterations: int,
    body_fun: Callable[[int, X], X],
    init_val: X,
) -> X:
  """Loops for `num_iterations`, applying `body_fun` in each iteration.

  It behaves similarly to a jax.lax.fori_loop, except it takes a fixed number of
  iterations as input, instead of a lower and upper bound. Another difference is
  that it lowers to a partir_mpmd.for primitive, instead of lowering as a scan
  primitive.

  NOTE: This primitive cannot be DCE'd when jit(..., keep_unused=False) because
  we need to guarantee that the loop as as many operands as results. Thus, we do
  not register a rule for DCE.

  TODO(jupvfranco): consider transpose and batching rules.
  TODO(jupvfranco): support nested loops. ATM this is not possible because we
  wrap the body loop with mpmd.call+function.

  Args:
    num_iterations: the number of iterations of the loop.
    body_fun: the body of the loop.
    init_val: the initial value of the loop.

  Returns:
    the result of the last iteration of the loop.

  Raises:
    ValueError: if the input and output structures, shapes and data types are
    different.
  """
  return _fori_loop_implementation(
      num_iterations, body_fun, init_val, wrap_body_with_mpmd_call=True
  )


def _register_fori_loop_primitive():
  """Registers the `fori_loop` op to JAX."""
  primitive = jex.core.Primitive('fori_loop')
  primitive.multiple_results = True
  primitive.def_impl(_for_i_impl)
  primitive.def_effectful_abstract_eval(_call_abstract_eval)
  # Introduces the rule to lower this primitive as a call_primitive, so that
  # jax.jit users can still lower for_i loops.
  # TODO(jupvfranco): ideally, this should be lowered the same way a scan is
  # lowered. However, we lower the loop as if fully unrolled for the sake of
  # simplicity. jax.jitting a fori_loop should be used for testing only (jax
  # users should use a jax.lax.fori_loop instead) so it shouldn't affect any
  # performance sensitive code. We should revisit soon though.
  jax_mlir.register_lowering(
      primitive, jax_mlir.lower_fun(_for_i_impl, multiple_results=True)
  )
  return primitive


fori_loop_p = _register_fori_loop_primitive()

# ===----------------------------------------------------------------------=== #
# mpmd.enumerated_for
# ===----------------------------------------------------------------------=== #


def _is_sequence_of_arrays(leaf):
  return isinstance(leaf, Sequence) and all(
      isinstance(x, jax.typing.ArrayLike) for x in leaf
  )


def _validate_inputs(itree: PyTree[Sequence[Y]]) -> None:
  """Checks that all `tree` leaves are sequences of arrays of the same length."""

  def _check_leaf(path, leaf) -> None:
    # While these checks may seem redundant, they are necessary as any object
    # that is not a sequence of arrays will still be visited in the tree.map.
    if not isinstance(leaf, Sequence):
      raise ValueError(
          'Expected all leaves of the `inputs` tree to be Sequences. Got leaf'
          f' at path {path} of type {type(leaf)}'
      )

    if not all(isinstance(x, jax.typing.ArrayLike) for x in leaf):
      raise ValueError(
          'Expected all leaves of the `inputs` tree to be sequences of arrays.'
          f' Got leaf at path {path} of type {type(leaf)}'
      )

  jax.tree_util.tree_map_with_path(
      _check_leaf,
      itree,
      is_leaf=_is_sequence_of_arrays,
  )

  all_sequences = jax.tree.leaves(itree, is_leaf=_is_sequence_of_arrays)
  if not all(len(seq) == len(all_sequences[0]) for seq in all_sequences):
    raise ValueError(
        'Expected all leaves of the `inputs` tree to be lists of the same'
        f' length. Got list lengths: {set(len(seq) for seq in all_sequences)}'
    )


def enumerated_for(
    body_fun: Callable[[X, int, Y], X],
    initial_carry: X,
    inputs: PyTree[list[Y]],
    *,
    unroll: bool = False,
) -> X:
  """A for-loop that iterates over the enumerated elements of a list.

  In python code:
  ```python
  carry = initial_carry
  for i, x in enumerate(inputs):
    carry = body_fun(carry, i, x)
  return carry
  ```

  Args:
    body_fun: the body of the loop.
    initial_carry: the initial value of the loop.
    inputs: a tree of sequences of arrays. The loop iterates over the enumerated
      elements of each sequences. All sequences must hold array like objects
      only and have the same length.
    unroll: whether to unroll the loop.

  Returns:
    The result of the last iteration of the loop.
  """

  _validate_inputs(inputs)

  all_sequences = jax.tree.leaves(inputs, is_leaf=_is_sequence_of_arrays)
  num_iterations = len(all_sequences[0])

  def _get_element_tree(inputs: PyTree[list[Y]], index) -> PyTree[Y]:
    # When the loop is rolled, we need to stack the arrays before the lookup.
    # PartIR will unroll the loop and canonicalize away the stacking+slicing.
    return jax.tree.map(
        lambda ys: ys[index] if unroll else jax.numpy.stack(ys)[index],
        inputs,
        is_leaf=_is_sequence_of_arrays,
    )

  if unroll:
    carry = initial_carry
    for i in range(num_iterations):
      carry = call(body_fun, call_counter=i)(
          carry, i, _get_element_tree(inputs, i)
      )
    return carry

  # We wrap `body_fun` with an mpmd.call so that we can apply the pipeline
  # scheduling pass on the unrolled loop.
  # But note we don't wrap the entire body of the for-loop. This is because
  # we want to canonicalize away the stacking and slicing used in
  # `_get_element_tree`, which would not happen when unrolling the for-loop if
  # the entire body was wrapped in an mpmd.call.
  # TODO: b/357831621 - This won't be an issue once we fully support the
  # for-loop in our passes and don't rely on having mpmd.calls in the body.
  body_fun = call(body_fun)

  def _for_i_body(index, x):
    carry, all_inputs = x
    new_carry = body_fun(carry, index, _get_element_tree(all_inputs, index))
    return new_carry, all_inputs

  # The for_i loop also returns the inputs, which are discarded here.
  resulting_carry, _ = _fori_loop_implementation(
      num_iterations,
      _for_i_body,
      (initial_carry, inputs),
      wrap_body_with_mpmd_call=False,
  )
  return resulting_carry


# ===----------------------------------------------------------------------=== #
# Lowerings
# ===----------------------------------------------------------------------=== #


_ops_to_lowering: dict[jex.core.Primitive, jax_mlir.LoweringRule] = {
    named_computation_p: named_computation_partir_lowering,
    named_tensor_p: named_tensor_partir_lowering,
    call_p: call_mpmd_jit_lowering,
    broadcast_p: broadcast_mpmd_lowering,
    reduce_p: reduce_lowering,
    fori_loop_p: fori_loop_mpmd_jit_lowering,
}

jit_lowerings = tuple(_ops_to_lowering.items())
