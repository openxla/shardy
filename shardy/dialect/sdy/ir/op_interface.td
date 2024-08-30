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
    An op interface that allows shardy to get and set the op argument shardings.
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
