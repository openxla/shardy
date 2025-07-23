// RUN: mpmd_opt %s -canonicalize 2>&1 | FileCheck %s

!mesh_1_replicated = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_distributed_0 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
!mesh_1_distributed_1 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{?}, {"x"}]>>

!mesh_2_replicated = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!mesh_2_distributed = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {?}]>>

// CHECK-LABEL: func @identity_transfer
func.func @identity_transfer(%arg0: !mesh_1_replicated) -> !mesh_1_distributed_0 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[TRANSFER:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
// CHECK-NEXT:  return %[[TRANSFER]]
  %0 = mpmd.transfer %arg0 : (!mesh_1_replicated) -> !mesh_1_replicated
  %1= mpmd.transfer %0 : (!mesh_1_replicated) -> !mesh_1_distributed_0
  func.return %1 : !mesh_1_distributed_0
}

// CHECK-LABEL: func @intra_mesh_transfer_of_transfer_one_use
func.func @intra_mesh_transfer_of_transfer_one_use(%arg0: !mesh_1_replicated) -> !mesh_1_distributed_1 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[TRANSFER:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{?}, {"x"}]>>
// CHECK-NEXT:  return %[[TRANSFER]]
  %0 = mpmd.transfer %arg0 : (!mesh_1_replicated) -> !mesh_1_distributed_0
  %1= mpmd.transfer %0 : (!mesh_1_distributed_0) -> !mesh_1_distributed_1
  func.return %1 : !mesh_1_distributed_1
}

// CHECK-LABEL: func @intra_mesh_transfer_of_transfer_multi_use
func.func @intra_mesh_transfer_of_transfer_multi_use(%arg0: !mesh_1_replicated) -> (!mesh_1_distributed_0, !mesh_1_distributed_1) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[TRANSFER_0:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
// CHECK-NEXT:  %[[TRANSFER_1:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{?}, {"x"}]>>
// CHECK-NEXT:  return %[[TRANSFER_0]], %[[TRANSFER_1]]
  %0 = mpmd.transfer %arg0 : (!mesh_1_replicated) -> !mesh_1_distributed_0
  %1= mpmd.transfer %0 : (!mesh_1_distributed_0) -> !mesh_1_distributed_1
  func.return %0, %1 : !mesh_1_distributed_0, !mesh_1_distributed_1
}

// CHECK-LABEL: func @inter_mesh_transfer_of_transfer
func.func @inter_mesh_transfer_of_transfer(%arg0: !mesh_1_replicated) -> !mesh_2_distributed attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
// CHECK-NEXT:  %[[TRANSFER_0:.*]] = mpmd.transfer %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
// CHECK-NEXT:  %[[TRANSFER_1:.*]] = mpmd.transfer %[[TRANSFER_0]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, sharding=<@mesh, [{"y"}, {?}]>>
// CHECK-NEXT:  return %[[TRANSFER_1]]
  %0 = mpmd.transfer %arg0 : (!mesh_1_replicated) -> !mesh_1_distributed_0
  %1= mpmd.transfer %0 : (!mesh_1_distributed_0) -> !mesh_2_distributed
  func.return %1 : !mesh_2_distributed
}
