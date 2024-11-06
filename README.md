# Shardy

Shardy is an MLIR-based tensor partitioning system for all dialects. Built from
the collaboration of both the [GSPMD](https://arxiv.org/abs/2105.04663) and
[PartIR](https://arxiv.org/abs/2401.11202) teams, it incorporates the best of
both systems, and the shared experience of both teams and users.

Shardy is meant to be dialect
agnostic and provide extensive user control and debuggability features. It
includes an axis-based sharding representation, a set of compiler APIs,
functionality for sharding propagation, and plans for an SPMD partitioner.

## Status

Shardy is a work in progress. Currently the core dialect and c bindings are
fully functional. The python bindings are under development.

## Contacts

*   For questions, contact the maintainers - maintainers at openxla.org
