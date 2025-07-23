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

#include "shardy/dialect/mpmd/transforms/export/utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::mpmd {

using ::mlir::func::FuncOp;

using OperandLastUseMap = DenseMap<Operation*, SmallVector<unsigned int>>;

DenseSet<BlockArgument> GetAliasedBlockArguments(FuncOp main_func) {
  DenseSet<BlockArgument> aliased_block_args;
  for (unsigned i = 0; i < main_func.getNumArguments(); ++i) {
    if (main_func.getArgAttrOfType<IntegerAttr>(i, kAliasingAttrName)) {
      aliased_block_args.insert(main_func.getArgument(i));
    }
  }
  return aliased_block_args;
}

DenseSet<BlockArgument> GetDonatedBlockArguments(FuncOp main_func) {
  DenseSet<BlockArgument> donated_block_args;
  for (unsigned i = 0; i < main_func.getNumArguments(); ++i) {
    auto donated_attr =
        main_func.getArgAttrOfType<BoolAttr>(i, kBufferDonationAttrName);
    if (donated_attr && donated_attr.getValue()) {
      donated_block_args.insert(main_func.getArgument(i));
    }
  }
  return donated_block_args;
}

OperandLastUseMap OperandsForDeletionMapping(FuncOp main_func) {
  OperandLastUseMap last_use_map;

  DenseMap<Value, OpOperand*> value_to_last_use;
  for (Operation& op : main_func.getOps()) {
    for (OpOperand& use : op.getOpOperands()) {
      value_to_last_use[use.get()] = &use;
    }
  }

  for (auto [_, use] : value_to_last_use) {
    last_use_map[use->getOwner()].push_back(use->getOperandNumber());
  }
  return last_use_map;
}

}  // namespace mlir::mpmd
