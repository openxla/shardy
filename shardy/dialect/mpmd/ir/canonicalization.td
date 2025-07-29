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

#ifndef MPMD_CANONICALIZATION
#define MPMD_CANONICALIZATION

include "shardy/dialect/mpmd/ir/ops.td"
include "shardy/dialect/mpmd/ir/types.td"
include "mlir/IR/PatternBase.td"

def HaveSameType : Constraint<CPred<"$0.getType() == $1.getType()">, "have same type">;

def HaveSameMeshName : Constraint<
    CPred<[{mlir::cast<mlir::mpmd::MeshTensorType>($0.getType()).getMeshName() ==
            mlir::cast<mlir::mpmd::MeshTensorType>($1.getType()).getMeshName()}]>,
    "have same mesh name">;

def IdentityTransferPattern : Pat<(TransferOp:$result $tensor),
                                (replaceWithValue $tensor),
                                [(HaveSameType $result, $tensor)]>;

// Replaces a chain of intra-mesh transfers with a single transfer.
//
// In symbols:
//
//  y = transfer x : (mesh_tensor<M, <D1>>) -> mesh_tensor<M, <D2>>
//  z = transfer y : (mesh_tensor<M, <D2>>) -> mesh_tensor<M, <D3>>
//  ~>
//  z = transfer x : (mesh_tensor<M, <D1>>) -> mesh_tensor<M, <D3>>
//
//  where y has only one use, M is a specific mesh name, and each di denotes a
//  different distributed type.
// TODO(jupvfranco): Consider removing. These transfers could have been created
// by the user.
def IntraMeshTransferOfTransferPattern : Pat<(TransferOp:$outer_res (TransferOp:$inner_res $tensor)),
                                             (TransferOp $tensor),
                                             [(HaveSameMeshName $tensor, $inner_res),
                                              (HaveSameMeshName $inner_res, $outer_res)]>;

// Replaces a chain of broadcasts with a single broadcast.
//
// In symbols:
//
//  y = broadcast x
//  z = broadcast y
//  ~>
//  z = broadcast x
def BroadcastOfBroadcastPattern : Pat<(BroadcastOp (BroadcastOp AnyTensor: $operand)),
                                        (BroadcastOp $operand)>;

#endif
