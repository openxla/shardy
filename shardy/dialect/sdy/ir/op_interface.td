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
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// ShardableDataFlowOpInterface
//===----------------------------------------------------------------------===//

def Sdy_ShardableDataFlowOpInterface : OpInterface<"ShardableDataFlowOpInterface"> {
  let description = [{
    An op interface that allows shardy to propagate shardings through data flow
    edges of ops that extend this interface.

    A data flow edge of some op X defines a bridge between a set of sources
    (each is either an operand of X or an operand of X's block terminator) and
    a set of targets (each is either a result of X or a block argument of X),
    such that all sources and targets should be sharded in the same way.
    An op can have multiple data flow edges that are orthogonal to one another.

    An owner is a user specified target of the data flow edge used by shardy's
    propagation. The user can choose it arbitrarily but it needs to be static.

    For example:

    ```mlir
      y_1, ..., y_n = custom_op (x_1, ..., x_n)
                      ((body_arg_1,..., body_arg_n) {
                        ...
                        return return_value_1, ..., return_value_n
                      })
    ```
    This custom_op has two types for data flow edges, n edges each between
    `return_value_i` (sources) and `y_i` (targets) and n edges between
    `x_i`(sources) and `body_arg_i`(targets). In this case the edge owners are
    the same as the targets.

    Here is an example of an op with multiple targets:
    ```mlir
      y_0, ..., y_n = while (x_0, ..., x_n)
                      ((pred_arg_0,... , pred_arg_n) { ... })
                      ((body_arg_0,..., body_arg_n) {
                        ...
                        return return_value_0, ..., return_value_n
                      })
    ```

    This while op has n data flow edges, the i-th data flow edges is between
    sources `x_i`, `return_value_i` and targets `y_i`, `pred_arg_i`,
    `body_arg_i`.
  }];
  let cppNamespace = "::mlir::sdy";

  let methods = [
    // TODO(b/380924978): Delete this method in favor of a new method returning
    // a single sharding.
    InterfaceMethod<
      /*desc=*/[{
        Returns the shardings of all block argument data flow edge owners.
      }],
      /*retType=*/"mlir::SmallVector<mlir::sdy::TensorShardingAttr>",
      /*methodName=*/"getBlockArgumentEdgeOwnerShardings",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{return {};}]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets `shardings` of all block argument edge owners.
      }],
      /*retType=*/"void",
      /*methodName=*/"setBlockArgumentEdgeOwnerShardings",
      /*args=*/(ins "mlir::ArrayRef<mlir::sdy::TensorShardingAttr>":$shardings),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{return;}]
    >,
    // TODO(b/380924978): Delete this method in favor of a new method returning
    // a single sharding.
    InterfaceMethod<
      /*desc=*/[{
        Returns the shardings of all op result data flow edge owners.
      }],
      /*retType=*/"mlir::SmallVector<mlir::sdy::TensorShardingAttr>",
      /*methodName=*/"getOpResultEdgeOwnerShardings",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return mlir::sdy::details::getOpResultEdgeOwnerShardingsImpl($_op);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets `shardings` of all op result edge owners.
      }],
      /*retType=*/"void",
      /*methodName=*/"setOpResultEdgeOwnerShardings",
      /*args=*/(ins "mlir::ArrayRef<mlir::sdy::TensorShardingAttr>":$shardings),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        mlir::sdy::details::setOpResultEdgeOwnerShardingsImpl($_op, shardings);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Transforms the `sharding` of the target depending on `transformType`

        See `DataFlowShardingTransformType` for more information.
      }],
      /*retType=*/"mlir::sdy::TensorShardingAttr",
      /*methodName=*/"transformTargetSharding",
      /*args=*/(ins "mlir::Value":$target,
                    "mlir::sdy::TensorShardingAttr":$sharding,
                    "mlir::sdy::DataFlowShardingTransformType":$transformType),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return sharding;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets all block argument edge owners.
      }],
      /*retType=*/"mlir::ArrayRef<mlir::BlockArgument>",
      /*methodName=*/"getBlockArgumentEdgeOwners",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{return {};}]
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
        Gets the data flow edge sources given the edge `owner`.
      }],
      /*retType=*/"mlir::SmallVector<mlir::Value>",
      /*methodName=*/"getEdgeSources",
      /*args=*/(ins "mlir::Value":$owner)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the owner `target` of a data flow edge given a `target` that may or
        may not be the owner.
      }],
      /*retType=*/"mlir::Value",
      /*methodName=*/"getEdgeOwnerFromTarget",
      /*args=*/(ins "mlir::Value":$target)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the owner target of a data flow edge given a `source`.
      }],
      /*retType=*/"mlir::Value",
      /*methodName=*/"getEdgeOwnerFromSource",
      /*args=*/(ins "mlir::OpOperand&":$source)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Gets the non-owner targets of a data flow edge given the edge `owner`.
      }],
      /*retType=*/"mlir::SmallVector<mlir::Value>",
      /*methodName=*/"getNonEdgeOwnerTargets",
      /*args=*/(ins "mlir::Value":$owner),
      /*methodBody=*/"",
      /*defaultImplementation=*/"return {};"
    >,
  ];

  let extraClassDeclaration = [{
    // Gets the sharding of the block argument edge owner with the given
    // `index`.
    TensorShardingAttr getBlockArgumentEdgeOwnerSharding(
        unsigned index);

    // Gets the sharding of the op result edge owner with the given `index`.
    TensorShardingAttr getOpResultEdgeOwnerSharding(unsigned index);

    // Gets the sharding of the edge owner of the given `value`.
    TensorShardingAttr getEdgeOwnerSharding(Value value);

    // Sets the `sharding` of the block argument edge owner with the given
    // `index`.
    void setBlockArgumentEdgeOwnerSharding(
      unsigned index, TensorShardingAttr sharding);

    //  Sets the `sharding` of the op result edge owner with the given `index`.
    void setOpResultEdgeOwnerSharding(
      unsigned index, TensorShardingAttr sharding);

    // Sets the `sharding` of the edge owner of the given `value`.
    void setEdgeOwnerSharding(
        Value value, TensorShardingAttr sharding);
  }];
}

//===----------------------------------------------------------------------===//
// ShardingRuleOpInterface
//===----------------------------------------------------------------------===//

def Sdy_ShardingRuleOpInterface : OpInterface<"ShardingRuleOpInterface"> {
  let description = [{
    An op interface that allows the op to define its own sharding rule.
    A sharding rule specifies how an operation can be partitioned according to
    various properties on the op - any attributes, the shape of operands,
    the shape of the results, etc. See `OpShardingRuleAttr` for more
    details.
  }];
  let cppNamespace = "::mlir::sdy";
    let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the sharding rule of the op.
      }],
      /*retType=*/"mlir::sdy::OpShardingRuleAttr",
      /*methodName=*/"getShardingRule"
    >,
  ];
}

//===----------------------------------------------------------------------===//
// CollectiveOpInterface
//===----------------------------------------------------------------------===//

def Sdy_CollectiveOpInterface : OpInterface<"CollectiveOpInterface"> {
  let description = [{
    Interface for all collective ops. Encapsulates common get/set for
    outSharding attribute.

    **Constraints:**
    - Operand must have a sharding.
    - `out_sharding` is valid w.r.t the corresponding type.
    - MeshAttr of result and operand is the same.
    - Same rank for the operand and result sharding.
  }];
  let cppNamespace = "::mlir::sdy";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Returns the output tensor sharding of the collective op.
      }],
      /*retType=*/"::mlir::sdy::TensorShardingAttr",
      /*methodName=*/"getOutSharding"
    >,
    InterfaceMethod<
      /*desc=*/[{
        Sets the output tensor sharding of the collective op.
      }],
      /*retType=*/"void",
      /*methodName=*/"setOutShardingAttr",
      /*args=*/(ins "::mlir::sdy::TensorShardingAttr":$sharding)
    >,
    InterfaceMethod<
      /*desc=*/[{ Get the tensor operand of the collective op. }],
      /*retType=*/"::mlir::TypedValue<::mlir::TensorType>",
      /*methodName=*/"getTensor"
    >,
    InterfaceMethod<
      /*desc=*/[{ Get the type of the collective op result. }],
      /*retType=*/"::mlir::Type",
      /*methodName=*/"getType"
    >
  ];
  let verify = [{
    return ::mlir::sdy::verifyCollectiveOp($_op);
  }];
}

#endif  // SDY_OP_INTERFACE
