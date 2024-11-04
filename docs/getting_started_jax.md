# Shardy guide for JAX users

Shardy is a new propagation system being introduced into the XLA stack, and below we want to introduce any JAX users to:

1. What has changed in JAX
2. Why Shardy?
3. Future plans

This is meant for JAX users who use `jax.jit` for running training/inference models across more than 1 GPU or TPU (batch parallelism, megatron, ZeRO, etc). They would be using things like `PartitionSpec`s and `NamedSharding`s.

## 1. What has changed in JAX?

### State of JAX before: GSPMD

Prior to Shardy, JAX users who partitioned their models across models across multiple devices used [GSPMD](http://go/arxiv/2105.04663) behind the scenes.

GSPMD is the propagation+partitioning system that lives in the middle of the XLA pipeline. It operates on HLO - the IR that comes after StableHLO (the program you get after running `jax.jit.lower`).

JAX doesn't run GSPMD directly, but encodes instructions into the StableHLO IR.

But before we go any further, let's introduce our working example.

Note that this example should be executed in a TPU environment with 8 devices.

```python
import os
# make sure our program runs on 8 devices
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax._src.sharding_impls import UNSPECIFIED
from jax.experimental.shard_map import shard_map
```

First let's create our mesh.

```python
mesh = Mesh(
    np.reshape(np.array(jax.devices()), (4, 2)),
    ('data', 'model'))

print(mesh.shape)
```

### In/Out shardings

Let's look what changed the most: how sharding attributes are encoded in the JAX program for the compiler to read.

Let's look at it through an example. It's going to be an MLP-like model consisting of no bias tensors, and 2 layers (two matmuls).

```python
def predict(x, w1, w2):
  x = jnp.tanh(x)
  z1 = jnp.einsum('ij,jk->ik', x, w1)
  z2 = jnp.einsum('ij,jk->ik', z1, w2)
  return jnp.sin(z2)
```

What we will want to do here sharding wise is:

1. `data` parallelism on x
2. `model` parallelism on w1 and w2 through the [megatron](http://go/arxiv/1909.08053) sharding strategy.

Now let's prepare the model for GSPMD sharding. Note that we will explicitly shard `w1`, but let GSPMD propagation shard `w2`.

```python
def run_in_out_shardings():
  samples = jax.ShapeDtypeStruct((16, 128), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec('data', None)))
  samples_sharding = NamedSharding(mesh, PartitionSpec('data', None))
  w1 = jax.ShapeDtypeStruct((128, 256), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec(None, 'model')))
  w1_sharding = NamedSharding(mesh, PartitionSpec(None, 'model'))
  w2 = jax.ShapeDtypeStruct((256, 10), jnp.float32)
  w2_sharding = UNSPECIFIED

  print(jax.jit(predict, in_shardings=(samples_sharding, w1_sharding, w2_sharding)).lower(samples, w1, w2).as_text())

run_in_out_shardings()
```

GSPMD's sharding annotations are defined as [HloShardingV2](TBA). So we have the following correspondance:

| JAX sharding    | GSPMD sharding |
| -------- | ------- |
| `NamedSharding(mesh, PartitionSpec('data', None))`  | `{devices=[4,1,2]<=[8] last_tile_dim_replicate}`    |
| `NamedSharding(mesh, PartitionSpec(None, 'model'))` | `{devices=[1,2,4]<=[4,2]T(1,0) last_tile_dim_replicate}`     |
| `UNSPECIFIED`    | nothing    |

`UNSPECIFIED` is no sharding as expected since GSPMD will populate this during sharding propagation.

Notice how all the axis names go away? While there is a 1:1 correspondance between `NamedSharding` and `HloShardingV2`, as a reader, it can be difficult to read. It is only more difficult once you introduce various axis names.

Let's look at Shardy for comparison. To enable Shardy in JAX, simplify enable the flag:

```python
jax.config.update("jax_use_shardy_partitioner", True)
run_in_out_shardings()
jax.config.update("jax_use_shardy_partitioner", False)
```

Now we have

| JAX sharding    | Shardy sharding |
| -------- | ------- |
| `NamedSharding(mesh, PartitionSpec('data', None))`  | `#sdy.sharding<@mesh, [{"data"}, {}]>`    |
| `NamedSharding(mesh, PartitionSpec(None, 'model'))` | `#sdy.sharding<@mesh, [{}, {"model"}]>`     |
| `UNSPECIFIED`    | nothing    |

Shardy's representation is a lot closer to what JAX `NamedSharding`s are like. So when looking at a file dump of your program after propagation, it will be a lot easier to understand what is going on since the correspondance is a lot closer to JAX.

Note that instead of the total devices/axes living on the sharding, they live on a top level `@mesh` value.

### `jax.lax.with_sharding_constraint`

GSPMD currently lowers it to a custom call:

```python
def run_with_sharding_constraint():
  x = jax.ShapeDtypeStruct((32, 64), jnp.float32)

  def f(x):
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, PartitionSpec('data', PartitionSpec.UNCONSTRAINED)))

  print(jax.jit(f).lower(x).as_text())

run_with_sharding_constraint()
```

But under Shardy, it's an explicit op:

```python
jax.config.update("jax_use_shardy_partitioner", True)
run_with_sharding_constraint()
jax.config.update("jax_use_shardy_partitioner", False)
```

Note that `UNCONSTRAINED` under GSPMD has the custom call have an op attribute `backend_config = "unspecified_dims=[1]"`. But under Shardy, it makes dim 1 be `{?}`. In Shardy, dimension shardings without a `?` are closed, meaning that dimension can't be further sharded, but when it has a trailing `?`, it can be further sharded. Refer to [Sharding representation](sharding_representation.md) for more info on the sharding representation.

### `jax.experimental.shard_map`

Under GSPMD this is a few different custom calls with various `shard_map` specific attributes on the `HloShardingV2`. Let's look where the `model` axis is `auto`, meaning it's free to be used inside the body of the shard_map by sharding constraints.

```python
def run_shard_map():
  x = jax.ShapeDtypeStruct((32, 64), jnp.float32)

  def body(x):
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, PartitionSpec('model', PartitionSpec.UNCONSTRAINED)))
    return jax.lax.all_gather(x, 'data', tiled=True)

  shmaped_f = shard_map(
        body,
        mesh=mesh,
        in_specs=(jax.sharding.PartitionSpec('data',),),
        out_specs=jax.sharding.PartitionSpec(),
        check_rep=False,
        auto = frozenset(['model']))

  print(jax.jit(shmaped_f).lower(x).as_text())

print(run_shard_map())
```

With the custom calls and `HloShardingV2`s, it's getting pretty confusing. Let's look at what Shardy gives:

```python
jax.config.update("jax_use_shardy_partitioner", True)
run_shard_map()
jax.config.update("jax_use_shardy_partitioner", False)
```

We now:

- Have a single op called `sdy.manual_computation` which holds:
  - the `in_specs`
  - the `out_specs`
  - the body of the shard_map
  - the inverse of the `auto` axes which we call `manual_axes`

A lot easier to read!

Note that `manual_axes` is always equal to the axes in the `mesh` but not in the `auto = frozenset([...])`.

Also, this will be relaxed in the JAX side soon, but you will be able to use `UNSPECIFIED` on in/out specs and have `auto` axes propagate from inside to outside the shard_map body (and vice-versa).

### Auto partitioners

In progress.

### XLA_DUMP_TO

When specifying the `XLA_DUMP_TO`, you will see an additional `shardy/` directory containing various dumps of the StableHLO program. A lot of them are currently only relevant to the Shardy team to debug issues. The one you should focus on when debugging is `sdy_module_after_sdy_export.mlir` which is the module after propagation finishes and the StableHLO program.

## 2. Why Shardy?

### Readability

As seen above, it's much easier to read the shardings and `shard_maps` and understand how they match what is happening in the JAX code. Similarly GSPMD propagation will give back HLO code - not MLIR which both Shardy and `jax.jit.lower` return.

### Interpretability

We are planning on exposing a feature we call "user priorities" (not in JAX yet!). It allows you to attach a value telling Shardy how important a tensor's dimension sharding is over other constraints in the program.

Higher priorities are defines as lower values (lowest being 0).

```python
PartitionSpec(None, 'x', 'y', priorities=(None, 0, 1))
```

Here the sharding of dim 1 on `x` has a higher priority than dim 2 on `y`, meaning dim 1 will be propagated through the program first and then dim 2, meaning any potential sharding conflicts will be explicitly resolved by having `x` propagated first.

This can be helpful for debugging models as well by having you break down your sharding strategies to separate rounds of propagation in Shardy. For example:

* Priority 0: data parallelism
* Priority 1: megatron
* Priority 2: ZeRO sharding

## FAQS

Below is a list of questions you may have on various JAX features and capabilities.

### JAX Sharding types

#### What about GSPMDSharding?

`GSPMDSharding` is closely tied to `HloShardingV1`/`HloShardingV2`. As such the type itself won't be supported.

#### How about different logical device ID orderings?

`GSPMDSharding`s were the answer for this, but instead (TBD) JAX will add a `device_ids` field to `NamedSharding`. And then a different `@mesh` value will be added to the `StableHLO` which saves the different device orderings. You'd now just need to give names to this type of sharding while you didn't before.

TODO add example

#### What about PositionalSharding?

This won't be supported. Instead use a `NamedSharding` with `device_ids`.

#### PmapSharding

This won't be supported. Shardy is meant for `jax.jit`, not `jax.pmap`.

### Propagation Questions

Section for questions about what you may see during propagation. It's probably best to refer to our existing content under go/shardy.

TODO

#### What are split Axes in Shardy, aka "x":(2)2?

Refer to [Axis splitting and sub-axes](sharding_representation.md/#axis-splitting-and-sub-axes).
