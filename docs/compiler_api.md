# Compiler API

## Background

We assume readers are familiar with at least the basics of [sharding representation](sharding_representation.md), which describes how the sharding of a tensor can be expressed in Shardy. This document shows how the sharding representations can be used in a program, e.g. to attach a sharding to a specific tensor of the program.

Sharding propagation is the process of deciding on a sharding for every tensor in a program given sharding constraints for a subset of the tensors. Shardy’s compiler API exposes several ways to influence/control sharding propagation. Additionally it allows users to insert manually sharded computations into their programs.

## Objective

This doc describes the design of such API components in Shardy and explains their behavior and invariants. Note that while this API is used to control sharding propagation, this doc is **NOT** going to discuss anything about the behavior of propagation nor how it’s designed.

## Overview

* [**Input/output shardings**](#inputoutput-shardings) - attach a sharding to an input or output of the main function, to indicate that this is how the input/output tensor should be sharded when given-to/returned-from the function.

* [**Sharding Constraint**](#sharding-constraint) - attach a sharding to an intermediate tensor (e.g. the result of a matmul) to indicate that this is how that tensor, or a subset of its uses, should be sharded.

* [**Shard As/Like**](#shard-as) - group multiple tensors by an ID to indicate that they should be sharded in the same way.

* [**Manual Computation**](#manual-computation) - encloses a sub-computation that is manually partitioned using a subset of mesh axes, where the shardings along those manual axes are specified for all inputs and outputs, and inside the sub-computation the tensor types are local w.r.t those shardings.

## Detailed Design

### Input/output shardings

Allows users to specify a sharding for the inputs and outputs of the main function.

In MLIR, attributes can be attached to function arguments and results, and therefore users can attach sharding attributes to the function this way.

For example:

```c
@mesh_xy = <"x"=2, "y"=2>

// The 1st input has a sharding specified, but the 2nd input doesn't.
// The output has a sharding specified.
func @main(%arg0: tensor<8x8xf32> 
            {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}]>},
                       %arg1: tensor<8x16xf32>)
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>}) {
  ...
}
```

### Sharding Constraint

Allows users to attach a sharding to an intermediate tensor in their program, which tells the partitioner that this is how that tensor, or a subset of its uses, should be sharded.

This is an MLIR operation that takes the tensor as input, and has a sharding attribute attached to it. The operation can either:

* Have no uses (dangling) - which means the attached sharding is how the tensor itself should be sharded.
* Have uses - which means the attached sharding is how the uses of the sharding constraint op should be sharded, while other uses of the input tensor might have a different sharding (if the input tensor has no other uses then the behavior is the same as the no uses case). Propagation will determine the sharding of the tensor itself and reshard it if necessary.

It can have open dimension shardings, which mean the operand can be further sharded along available axes.

```c
@mesh_xy = <"x"=2, "y"=2>

%0 = ... : tensor<8x8xf32>
%1 = sdy.sharding_constraint %0 <@mesh_xy, [{"x"}, {?}]> : tensor<8x8xf32>
```

### Shard As

In cases where there are no data dependencies or no strong data dependencies between two or more tensors, while users have the knowledge that those tensors should be partitioned in the same or in a similar ways, the Shardy API offers a way to specify this relation. This gives users the freedom to explicitly specify that tensors should be partitioned as each other.

To achieve this, we introduce a notion of shard groups, where each group contains any number of instructions which are associated with the same shard group id. Sharding groups enforce shardings within the same group to be the same.

For instance, in a hypothetical user program such as shown below, we want to shard the output of the program exactly the same as the input of the program while there are no data dependencies between the two. 

If we run this program, sharding propagation will not be able to infer on the sharding of tensors `%1` and `%2`, and they will end up being replicated. However, by attaching a `shard_group` attribute which says that the input `%0` and the output `%2` are within the same `shard_group`, we allow the sharding `@mesh_xy,` `[{"x"},{"y"}]>` to be propagated from the input `%0` to the output `%2`, and in turn to the rest of the graph, which is broadcasted constant `%1` here.

```c
@mesh_xy = <"x"=2, "y"=2>

module @"jit_zeros_like" {
  func.func @main(%arg0: tensor<8x2xi64> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"},{"y"}]>}}) -> (tensor<8x2xi64>) {
    %0 = sdy.sharding_group %arg0, id=0 : tensor<8x2xi64>
    %1 = stablehlo.constant dense<0> : tensor<8x2xi64>
    %2 = sdy.sharding_group %1, id=0 : tensor<8x2xi64>
    return %2 : tensor<8x2xi64>
  }
}
```

In this simple example above, alternatively we could’ve explicitly specified the same sharding on the output as the input, which would achieve the same effect, since we’ve already known what shard we want to assign to the input ahead of time but in more realistic cases, we use shard as to keep the sharding of multiple tensors in sync without necessarily knowing the sharding for any of them, while Shardy will take care of the rest and find the best sharding to assign to them.

### Manual Computation

Users might want explicit control of how parts of their computation are partitioned, and what collectives are used. For example, some users want to apply collective matmul manually (from the frontend API) rather than deferring to the compiler. We provide a Manual Computation API that allows them to do that.

This is the MLIR operation with a single region for the manual sub-computation. Users would specify input/output shardings to this sub-computation using a subset (including possibly all) of the mesh axes. The sub-computation would be local/manual w.r.t. the specified mesh axes (aka manual axes), and global/unpartitioned w.r.t. unspecified ones (aka free axes). The sub-computation can be further sharded along the free axes during propagation in the same way that computation outside of this operation can be.

For example:

```c
@mesh_name = <"data"=2, "model"=2>

%0 = ... : tensor<16x32xf32>
%1 = sdy.manual_computation(%0)     in_shardings=[<@mesh_name, [{"data"}, {"model",?}]>]      out_shardings=[<@mesh_name, [{"data"}, {?}]>]      manual_axes={"data"}      (%arg1: tensor<8x32xf32>) {
  // body
  return %42 : tensor<8x32xf32>
} : (tensor<16x32xf32>) -> tensor<16x32xf32>
```

**Note** that the shape of the input and output tensors inside the body are the local shapes of the corresponding operands and results of the op, i.e., the shape on a single device if all non-manual axes are replicated. Therefore, the local shape can be derived from the corresponding in/out sharding and the manual axes. If an input/output dimension of size `d` is sharded on axis `"x"` and that axis is in `manual_axes`, then that corresponding input/output dimension inside the body is `d/size("x")`.

#### Invariants

1. All `in_shardings`, `out_shardings` and `manual_axes` must refer to the same mesh. `manual_axes` is sorted w.r.t. the mesh.

2. The `manual_axes` must be explicitly used in **all** in/out shardings, i.e., for each sharding, all manual axes must either shard a dimension or be explicitly replicated.

3. If a free axis (any mesh axis not in `manual_axes`) exists in one of the in/out shardings, it must be minor to any manual axis in the same dimension sharding (in the above example, a dimension sharding `{"model", "data"}` would be invalid).

4. The region/body of the computation is the local computation (e.g., including user specified collectives). It must be local w.r.t. the in/out sharding along manual axes (see note above).

#### Nesting manual computations

You can nest multiple manual computations within one another as long as each one operates on their own unique set of manual axes.

#### Grammar

```c
sdy.manual_computation ::=      (operand_1,...,operand_n)    in_shardings=[in_sharding_1,...,in_sharding_n]   out_shardings=[out_sharding_1,...,out_sharding_m]   manual_axes={<manual_axis_name_1>,...,<manual_axis_name_r>}   region
: (operand_type_1,...,operand_type_n)
  -> (result_type_1,...,result_type_m) 

operand ::= MLIR::Value

in_sharding ::= sharding // see go/sdy-sharding-representation 
out_sharding ::= sharding // see go/sdy-sharding-representation   

manual_axis_name ::= str

region ::= body of the computation (MLIR::Region)

operand_type := mlir::RankedTensorType
result_type := mlir::RankedTensorType
```
