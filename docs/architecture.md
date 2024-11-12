# Shardy architecture

## Overview

The [GSPMD](https://arxiv.org/abs/2105.04663) and [PartIR](https://arxiv.org/abs/2401.11202) teams have joined forces to build a new partitioning system called Shardy, which incorporates the best of both systems, and the shared experience of both teams and users.

### Benefits

* More control and predictability for users by combining GSPMD's propagation with PartIR's incremental partitioning.
* New features driven by shared experience, e.g. novel support for reshapes which notoriously generate extra communication unless users know how to work around them.
* Better usability and debuggability to increase end-user velocity, e.g. by using an axis-based sharding representation.
* A simple, open source codebase using MLIR, with a broader set of active contributors (internal, external, and across various time zones) to support users.

### Components

* Sharding Representation: An axis-based sharding representation that is bound to a specific logical mesh (out of potentially multiple meshes), and supports constraining dimension shardings and axes, splitting axes for operations like reshape, priorities for incremental partitioning, and more.
* Compiler APIs: A set of compiler components that can be used alongside the sharding representation to influence sharding propagation.
    * Input/output shardings - attach a sharding to an input or output of the main function, to indicate that this is how the input/output tensor should be sharded when given-to/returned-from the function.
    * Sharding Constraint - attach a sharding to an intermediate tensor (e.g. the result of a matmul) to indicate that this is how that tensor, or a subset of its uses, should be sharded.
    * Shard As/Like - group multiple tensors by an ID to indicate that they should be sharded in the same way.
    * Manual Computation - Encloses a sub-computation that is manually partitioned using a subset of mesh axes, where the shardings along those manual axes are specified for all inputs and outputs, and inside the sub-computation the tensor types are local w.r.t those shardings.
* Sharding Propagation: A propagation algorithm which combines user priorities and sharding constraints, with compiler cost-models and heuristics:
    * User defined priorities, e.g. do batch parallelism then ZeRO
    * Op-based priorities, e.g. element-wise ops first then matmuls, etc.
    * More fine grained heuristics, e.g. prefer batch dimensions.
* SPMD Partitioner: A component which lowers sharding propagation decisions by partitioning the program into a SPMD program, adding the necessary data movement/formatting and collective operations in the process.
    * Short term, the initial implementation will use the current GSPMD SPMD partitioner.
    * Long term, we plan to create a new MLIR-based SPMD partitioner.

### Status

Shardy is being developed as an MLIR-based partitioning system. The Shardy project is in active development, and we seek feedback from the open source community.
