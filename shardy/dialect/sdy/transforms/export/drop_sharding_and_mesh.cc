/* Copyright 2026 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
...
==============================================================================*/

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace sdy {

#define GEN_PASS_DEF_DROPSHARDINGANDMESHPASS
#include "shardy/dialect/sdy/transforms/export/passes.h.inc"

namespace {

struct DropShardingAndMeshPass
    : public impl::DropShardingAndMeshPassBase<DropShardingAndMeshPass> {
  using DropShardingAndMeshPassBase::DropShardingAndMeshPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module = getOperation();

    module.walk([](Operation* op) {
      // kShardingAttr is "sdy.sharding"
      op->removeAttr(kShardingAttr);

      // Function arguments and results require special handling as their
      // attributes are stored on the FuncOp itself.
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        for (int i = 0; i < funcOp.getNumArguments(); ++i) {
          funcOp.removeArgAttr(i, kShardingAttr);
        }
        for (int i = 0; i < funcOp.getNumResults(); ++i) {
          funcOp.removeResultAttr(i, kShardingAttr);
        }
      }
    });

    module.walk([](MeshOp meshOp) { meshOp.erase(); });
  }
};

}  // namespace

}  // namespace sdy
}  // namespace mlir
