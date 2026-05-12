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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"
#include "shardy/dialect/mpmd/ir/utils.h"
#include "shardy/dialect/mpmd/transforms/import/passes.h"  // IWYU pragma: keep

using ::mlir::func::FuncOp;

namespace mlir::mpmd {

#define GEN_PASS_DEF_COPYTOPOLOGYFROMMAINPASS
#include "shardy/dialect/mpmd/transforms/import/passes.h.inc"

namespace {

class CopyTopologyFromMainPass
    : public impl::CopyTopologyFromMainPassBase<CopyTopologyFromMainPass> {
  using CopyTopologyFromMainPassBase::CopyTopologyFromMainPassBase;

 protected:
  void runOnOperation() final {
    ModuleOp module_op = getOperation();

    for (FuncOp func_op : module_op.getOps<FuncOp>()) {
      auto main_topology_attr =
          func_op->getAttrOfType<TopologyAttr>(kTopologyAttr);
      if (!main_topology_attr) continue;

      func_op.walk([main_topology_attr](CallOp call_op) {
        FuncOp callee = GetCalleeFunc(call_op);
        callee->setAttr(kTopologyAttr, main_topology_attr);
        // Set the callee to private to mark that it's not an entry point
        // function.
        callee.setPrivate();
      });
    }
  }
};

}  // namespace
}  // namespace mlir::mpmd
