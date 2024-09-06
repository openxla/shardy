/* Copyright 2024 The Shardy Authors.

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

#ifndef SDY_OP_INTERFACE
#define SDY_OP_INTERFACE

include "mlir/IR/OpBase.td"
include "shardy/dialect/sdy/ir/attrs.td"

//===----------------------------------------------------------------------===//
// ShardableDataFlowOpInterface
//===----------------------------------------------------------------------===//

def ShardableDataFlowOpInterface : OpInterface<"ShardableDataFlowOpInterface"> {
  let description = [{
    An op interface that allows shardy to add data flow edges to propagate
    shardings for:
      1. Between ops that extend this op interface and other ops and
      2. Within the region (if any) of ops that extend this op interface.

    Each data flow edge has an owner, a set of targets and a set of sources.
    If the sharding of the targets change, the sharding of the sources should
    be updated accordingly and vice versa.
    An owner is a user specified target of the data flow edge used by shardy's
    propagation. The user can choose it arbitrarily but it needs to be static.
  }];
  let cppNamespace = "::mlir::sdy";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the shardings of all block argument data flow edge owners.
      }],
      /*retType=*/"mlir::ArrayRef<mlir::sdy::TensorShardingAttr>",
      /*methodName=*/"getBlockArgumentEdgeOwnerShardings"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the shardings of the block argument edge owner with the given index.
      }],
      /*retType=*/"void",
      /*methodName=*/"setBlockArgumentEdgeOwnerSharding",
      /*args=*/(ins "unsigned":$index, "mlir::sdy::TensorShardingAttr":$sharding)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets shardings of all block argument edge owners.
      }],
      /*retType=*/"void",
      /*methodName=*/"setBlockArgumentEdgeOwnerShardings",
      /*args=*/(ins "mlir::ArrayRef<mlir::sdy::TensorShardingAttr>":$shardings)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets all block argument edge owners.
      }],
      /*retType=*/"mlir::ArrayRef<mlir::BlockArgument>",
      /*methodName=*/"getBlockArgumentEdgeOwners"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets all op result edge owners.
      }],
      /*retType=*/"mlir::ResultRange",
      /*methodName=*/"getOpResultEdgeOwners"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the data flow edge sources given a target value.
      }],
      /*retType=*/"mlir::SmallVector<mlir::Value>",
      /*methodName=*/"getSources",
      /*args=*/(ins "mlir::Value":$target)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the owner target of a data flow edge given a target that may or may not be the owner.
      }],
      /*retType=*/"mlir::Value",
      /*methodName=*/"getEdgeOwner",
      /*args=*/(ins "mlir::Value":$target)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the owner target of a data flow edge given a source.
      }],
      /*retType=*/"mlir::Value",
      /*methodName=*/"getEdgeOwnerFromSource",
      /*args=*/(ins "mlir::OpOperand&":$source)
    >,
  ];

  let extraClassDeclaration = [{
    // Gets the sharding of the block argument edge owner with the given index.
    mlir::sdy::TensorShardingAttr getBlockArgumentEdgeOwnerSharding(unsigned index){
      mlir::ArrayRef<mlir::sdy::TensorShardingAttr> argSharding = getBlockArgumentEdgeOwnerShardings();
      if(!argSharding.empty()) {
        return argSharding[index];
      }
      return nullptr;
    }
  }];
}

#endif  // SDY_OP_INTERFACE
