// RUN: mpmd_opt %s -mpmd-merge-transfers 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<f32>>
!mesh_1_tensor_2 = !mpmd.mesh_tensor<"m1", tensor<1xf32>>
!mesh_1_tensor_3 = !mpmd.mesh_tensor<"m1", tensor<bf16>>

sdy.mesh @mesh = <["x"=1]>
!mesh_1_tensor_sharded = !mpmd.mesh_tensor<"m1", tensor<1xf32>, sharding=<@mesh, [{"x"}]>>
!mesh_1_tensor_over_threshold = !mpmd.mesh_tensor<"m1", tensor<2xf32>>


!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<f32>>
!mesh_2_tensor_2 = !mpmd.mesh_tensor<"m2", tensor<1xf32>>
!mesh_2_tensor_3 = !mpmd.mesh_tensor<"m2", tensor<bf16>>
!mesh_2_tensor_sharded = !mpmd.mesh_tensor<"m2", tensor<1xf32>, sharding=<@mesh, [{"x"}]>>
!mesh_2_tensor_over_threshold = !mpmd.mesh_tensor<"m2", tensor<2xf32>>


!mesh_3_tensor = !mpmd.mesh_tensor<"m3", tensor<f32>>

// CHECK-LABEL: func @one_producer_many_consumers
func.func @one_producer_many_consumers(%arg0: !mesh_1_tensor)
  -> (!mesh_2_tensor, !mesh_3_tensor, !mesh_2_tensor) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=4]>>, <"m2": <["x"=4]>>, <"m3": <["x"=4]>>>
  }
{

// The producer fragment.
// CHECK-NEXT: %[[PROD:.*]]:6 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%[[ARG1:.*]]:
// Reshapes + Concat for one of the consumers.
// CHECK-NEXT:   %[[RESHAPE1:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE2:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[RESHAPE1]], %[[RESHAPE2]], dim = 0
// CHECKS-SAME:     : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// Reshapes + Concat for another consumer.
// CHECK-NEXT:   %[[RESHAPE3:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE4:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[RESHAPE3]], %[[RESHAPE4]], dim = 0
// CHECK-SAME:      : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// Reshapes + Concat for another consumer.
// CHECK-NEXT:   %[[RESHAPE5:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE6:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE7:.*]] = stablehlo.reshape %[[ARG1]] : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[CONCAT3:.*]] = stablehlo.concatenate %[[RESHAPE5]], %[[RESHAPE6]], %[[RESHAPE7]], dim = 0
// CHECK-SAME:      : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3xf32>
// The fragment returns the same results as before the rewrite and all the
// concat ops.
// CHECK-NEXT:   return %[[ARG1]], %[[ARG1]], %[[ARG1]], %[[CONCAT1]], %[[CONCAT2]], %[[CONCAT3]]

  %0:3 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<f32>) {
    mpmd.return %arg2, %arg2, %arg2 : tensor<f32>, tensor<f32>, tensor<f32>
  } : (!mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)

// The concatenated transfers. One per consumer fragment.
// CHECK-DAG: %[[CONCAT_TRANSFER1:.*]] = mpmd.transfer {concat_transfer} %[[PROD]]#5
// CHECK-SAME:   (!mpmd.mesh_tensor<"m1", tensor<3xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<3xf32>>
// CHECK-DAG: %[[CONCAT_TRANSFER2:.*]] = mpmd.transfer {concat_transfer} %[[PROD]]#4
// CHECK-SAME:   (!mpmd.mesh_tensor<"m1", tensor<2xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<2xf32>>
// CHECK-DAG: %[[CONCAT_TRANSFER3:.*]] = mpmd.transfer {concat_transfer} %[[PROD]]#3
// CHECK-SAME:   (!mpmd.mesh_tensor<"m1", tensor<2xf32>>) -> !mpmd.mesh_tensor<"m3", tensor<2xf32>>

// Existing transfers are not removed/rewriten.
// CHECK-DAG: %[[TRANSFER1:.*]] = mpmd.transfer %[[PROD]]#0
// CHECK-DAG: %[[TRANSFER2:.*]] = mpmd.transfer %[[PROD]]#1
// CHECK-DAG: %[[TRANSFER3:.*]] = mpmd.transfer %[[PROD]]#2
  %t1 = mpmd.transfer %0#0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %t2 = mpmd.transfer %0#1 : (!mesh_1_tensor) -> !mesh_2_tensor
  %t3 = mpmd.transfer %0#2 : (!mesh_1_tensor) -> !mesh_2_tensor

// A consumer fragment.
// CHECK-DAG: fragment<mesh="m2", origin=["f2"]> (%[[TRANSFER1]], %[[TRANSFER2]], %[[TRANSFER3]], %[[CONCAT_TRANSFER1]]) ({{.*}}, {{.*}}, {{.*}}, %[[ARG4:.*]]:
// The argument respective to concat transfer (%[[ARG4]]) is split into three tensors.
// CHECK-NEXT:   %[[SLICE1:.*]] = stablehlo.slice %[[ARG4]] [0:1] : (tensor<3xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE1:.*]] = stablehlo.reshape %[[SLICE1]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:   %[[SLICE2:.*]] = stablehlo.slice %[[ARG4]] [1:2] : (tensor<3xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE2:.*]] = stablehlo.reshape %[[SLICE2]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:   %[[SLICE3:.*]] = stablehlo.slice %[[ARG4]] [2:3] : (tensor<3xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE3:.*]] = stablehlo.reshape %[[SLICE3]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[RESHAPE1]], %[[RESHAPE2]]
// CHECK-NEXT:   stablehlo.add %[[ADD]], %[[RESHAPE3]]
  %1 = mpmd.fragment<mesh="m2", origin=["f2"]> (%t1, %t2, %t3)
    (%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
    %5 = stablehlo.add %4, %arg4 : tensor<f32>
    mpmd.return %5 : tensor<f32>
  } : (!mesh_2_tensor, !mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor

// Another consumer in a different mesh from the first consumer.
// CHECK-DAG: %[[TRANSFER4:.*]] = mpmd.transfer %[[PROD]]#0
// CHECK-SAME:   (!mpmd.mesh_tensor<"m1", tensor<f32>>) -> !mpmd.mesh_tensor<"m3", tensor<f32>>
// CHECK-DAG: %[[TRANSFER5:.*]] = mpmd.transfer %[[PROD]]#1
// CHECK-SAME:   (!mpmd.mesh_tensor<"m1", tensor<f32>>) -> !mpmd.mesh_tensor<"m3", tensor<f32>>
  %t4 = mpmd.transfer %0#0 : (!mesh_1_tensor) -> !mesh_3_tensor
  %t5 = mpmd.transfer %0#1 : (!mesh_1_tensor) -> !mesh_3_tensor
// CHECK-DAG: fragment<mesh="m3", origin=["f3"]> (%[[TRANSFER4]], %[[TRANSFER5]], %[[CONCAT_TRANSFER3]]) ({{.*}}, {{.*}}, %[[ARG3:.*]]:
// CHECK-NEXT:   %[[SLICE4:.*]] = stablehlo.slice %[[ARG3]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE4:.*]] = stablehlo.reshape %[[SLICE4]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:   %[[SLICE5:.*]] = stablehlo.slice %[[ARG3]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE5:.*]] = stablehlo.reshape %[[SLICE5]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:   stablehlo.add %[[RESHAPE4]], %[[RESHAPE5]]
  %2 = mpmd.fragment<mesh="m3", origin=["f3"]> (%t4, %t5)
    (%arg2: tensor<f32>, %arg3: tensor<f32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
    mpmd.return %4 : tensor<f32>
  } : (!mesh_3_tensor, !mesh_3_tensor) -> !mesh_3_tensor

// Another consumer in the same mesh from the first consumer.
// CHECK-DAG:  mpmd.fragment<mesh="m2", origin=["f4"]> (%[[TRANSFER1]], %[[TRANSFER2]], %[[CONCAT_TRANSFER2]]) ({{.*}}, {{.*}}, %[[ARG3:.*]]:
// CHECK-NEXT:   %[[SLICE6:.*]] = stablehlo.slice %[[ARG3]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE6:.*]] = stablehlo.reshape %[[SLICE6]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:   %[[SLICE7:.*]] = stablehlo.slice %[[ARG3]] [1:2] : (tensor<2xf32>) -> tensor<1xf32>
// CHECK-NEXT:   %[[RESHAPE7:.*]] = stablehlo.reshape %[[SLICE7]] : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:   stablehlo.add %[[RESHAPE6]], %[[RESHAPE7]]

  %3 = mpmd.fragment<mesh="m2", origin=["f4"]> (%t1, %t2)
    (%arg2: tensor<f32>, %arg3: tensor<f32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
    mpmd.return %4 : tensor<f32>
  } : (!mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor

  func.return %1, %2, %3 : !mesh_2_tensor, !mesh_3_tensor, !mesh_2_tensor
}

// CHECK-LABEL: func @different_shapes_do_not_merge
func.func @different_shapes_do_not_merge(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor_2)
  -> (!mesh_2_tensor, !mesh_2_tensor_2) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=4]>>, <"m2": <["x"=4]>>, <"m3": <["x"=4]>>>
  }
{
// CHECK-NOT: concat_transfer
// CHECK-NEXT: %[[PROD:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
// CHECK-NEXT:   mpmd.return %arg2, %arg3 : tensor<f32>, tensor<1xf32>
// CHECK-NEXT: }
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    (%arg2: tensor<f32>, %arg3: tensor<1xf32>) {
    mpmd.return %arg2, %arg3 : tensor<f32>, tensor<1xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor_2) -> (!mesh_1_tensor, !mesh_1_tensor_2)
// CHECK-NEXT: %[[TRANSFER1:.*]] = mpmd.transfer %[[PROD]]#0
// CHECK-NEXT: %[[TRANSFER2:.*]] = mpmd.transfer %[[PROD]]#1
  %t1 = mpmd.transfer %0#0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %t2 = mpmd.transfer %0#1 : (!mesh_1_tensor_2) -> !mesh_2_tensor_2
// CHECK-NEXT: fragment<mesh="m2", origin=["f2"]> (%[[TRANSFER1]], %[[TRANSFER2]])
// CHECK-NEXT:   mpmd.return %arg2, %arg3
  %1:2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%t1, %t2)
    (%arg2: tensor<f32>, %arg3: tensor<1xf32>) {
    mpmd.return %arg2, %arg3 : tensor<f32>, tensor<1xf32>
  } : (!mesh_2_tensor, !mesh_2_tensor_2) -> (!mesh_2_tensor, !mesh_2_tensor_2)

  func.return %1#0, %1#1 : !mesh_2_tensor, !mesh_2_tensor_2
}

// CHECK-LABEL: func @different_dtypes_do_not_merge
func.func @different_dtypes_do_not_merge(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor_3)
  -> (!mesh_2_tensor, !mesh_2_tensor_3) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=4]>>, <"m2": <["x"=4]>>, <"m3": <["x"=4]>>>
  }
{
// CHECK-NEXT: %[[PROD:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
// CHECK-NEXT:   mpmd.return %arg2, %arg3 : tensor<f32>, tensor<bf16>
// CHECK-NEXT: }
// CHECK-NOT: concat_transfer
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    (%arg2: tensor<f32>, %arg3: tensor<bf16>) {
    mpmd.return %arg2, %arg3 : tensor<f32>, tensor<bf16>
  } : (!mesh_1_tensor, !mesh_1_tensor_3) -> (!mesh_1_tensor, !mesh_1_tensor_3)
// CHECK-NEXT: %[[TRANSFER1:.*]] = mpmd.transfer %[[PROD]]#0
// CHECK-NEXT: %[[TRANSFER2:.*]] = mpmd.transfer %[[PROD]]#1
  %t1 = mpmd.transfer %0#0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %t2 = mpmd.transfer %0#1 : (!mesh_1_tensor_3) -> !mesh_2_tensor_3
// CHECK-NEXT: fragment<mesh="m2", origin=["f2"]> (%[[TRANSFER1]], %[[TRANSFER2]])
// CHECK-NEXT:   mpmd.return %arg2, %arg3
  %1:2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%t1, %t2)
    (%arg2: tensor<f32>, %arg3: tensor<bf16>) {
    mpmd.return %arg2, %arg3 : tensor<f32>, tensor<bf16>
  } : (!mesh_2_tensor, !mesh_2_tensor_3) -> (!mesh_2_tensor, !mesh_2_tensor_3)

  func.return %1#0, %1#1 : !mesh_2_tensor, !mesh_2_tensor_3
}


// CHECK-LABEL: func @sharded_tensors_do_not_merge
func.func @sharded_tensors_do_not_merge(%arg0: !mesh_1_tensor_sharded, %arg1: !mesh_1_tensor_sharded)
  -> (!mesh_2_tensor_sharded, !mesh_2_tensor_sharded) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=1]>>, <"m2": <["x"=1]>>>
  }
{
// CHECK-NOT: concat_transfer
// CHECK-NEXT: %[[PROD:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
// CHECK-NEXT:   mpmd.return %arg2, %arg3 : tensor<1xf32>, tensor<1xf32>
// CHECK-NEXT: }
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    (%arg2: tensor<1xf32>, %arg3: tensor<1xf32>) {
    mpmd.return %arg2, %arg3 : tensor<1xf32>, tensor<1xf32>
  } : (!mesh_1_tensor_sharded, !mesh_1_tensor_sharded) -> (!mesh_1_tensor_sharded, !mesh_1_tensor_sharded)
// CHECK-NEXT: %[[TRANSFER1:.*]] = mpmd.transfer %[[PROD]]#0
// CHECK-NEXT: %[[TRANSFER2:.*]] = mpmd.transfer %[[PROD]]#1
  %t1 = mpmd.transfer %0#0 : (!mesh_1_tensor_sharded) -> !mesh_2_tensor_sharded
  %t2 = mpmd.transfer %0#1 : (!mesh_1_tensor_sharded) -> !mesh_2_tensor_sharded
// CHECK-NEXT: fragment<mesh="m2", origin=["f2"]> (%[[TRANSFER1]], %[[TRANSFER2]])
// CHECK-NEXT:   mpmd.return %arg2, %arg3
  %1:2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%t1, %t2)
    (%arg2: tensor<1xf32>, %arg3: tensor<1xf32>) {
    mpmd.return %arg2, %arg3 : tensor<1xf32>, tensor<1xf32>
  } : (!mesh_2_tensor_sharded, !mesh_2_tensor_sharded) -> (!mesh_2_tensor_sharded, !mesh_2_tensor_sharded)

  func.return %1#0, %1#1 : !mesh_2_tensor_sharded, !mesh_2_tensor_sharded
}


// We're using the default threshold of 1 element.
// CHECK-LABEL: func @too_many_elements_do_not_merge
func.func @too_many_elements_do_not_merge(%arg0: !mesh_1_tensor_over_threshold, %arg1: !mesh_1_tensor_over_threshold)
  -> (!mesh_2_tensor_over_threshold, !mesh_2_tensor_over_threshold) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=1]>>, <"m2": <["x"=1]>>>
  }
{
// CHECK-NOT: concat_transfer
// CHECK-NEXT: %[[PROD:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
// CHECK-NEXT:   mpmd.return %arg2, %arg3 : tensor<2xf32>, tensor<2xf32>
// CHECK-NEXT: }
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1)
    (%arg2: tensor<2xf32>, %arg3: tensor<2xf32>) {
    mpmd.return %arg2, %arg3 : tensor<2xf32>, tensor<2xf32>
  } : (!mesh_1_tensor_over_threshold, !mesh_1_tensor_over_threshold) -> (!mesh_1_tensor_over_threshold, !mesh_1_tensor_over_threshold)
// CHECK-NEXT: %[[TRANSFER1:.*]] = mpmd.transfer %[[PROD]]#0
// CHECK-NEXT: %[[TRANSFER2:.*]] = mpmd.transfer %[[PROD]]#1
  %t1 = mpmd.transfer %0#0 : (!mesh_1_tensor_over_threshold) -> !mesh_2_tensor_over_threshold
  %t2 = mpmd.transfer %0#1 : (!mesh_1_tensor_over_threshold) -> !mesh_2_tensor_over_threshold
// CHECK-NEXT: fragment<mesh="m2", origin=["f2"]> (%[[TRANSFER1]], %[[TRANSFER2]])
// CHECK-NEXT:   mpmd.return %arg2, %arg3
  %1:2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%t1, %t2)
    (%arg2: tensor<2xf32>, %arg3: tensor<2xf32>) {
    mpmd.return %arg2, %arg3 : tensor<2xf32>, tensor<2xf32>
  } : (!mesh_2_tensor_over_threshold, !mesh_2_tensor_over_threshold) -> (!mesh_2_tensor_over_threshold, !mesh_2_tensor_over_threshold)

  func.return %1#0, %1#1 : !mesh_2_tensor_over_threshold, !mesh_2_tensor_over_threshold
}

// CHECK-LABEL: func @duplicate_transfer_use
func.func @duplicate_transfer_use(%arg0: !mesh_1_tensor)
  -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=4]>>, <"m2": <["x"=4]>>, <"m3": <["x"=4]>>>
  }
{

  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<f32>) {
    mpmd.return %arg2, %arg2 : tensor<f32>, tensor<f32>
  } : (!mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)

// CHECK:      %[[CONCAT_TRANSFER:.*]] = mpmd.transfer {concat_transfer} %0#2 : (!mpmd.mesh_tensor<"m1", tensor<2xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<2xf32>>
// CHECK:      mpmd.fragment<mesh="m2", origin=["f2"]> (%{{.*}}, %{{.*}}, %{{.*}}, %[[CONCAT_TRANSFER]])
// CHECK-SAME:   (%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<2xf32>) {
// CHECK-DAG:      stablehlo.slice %arg4 [0:1]
// CHECK-DAG:      stablehlo.slice %arg4 [0:1]
// CHECK-DAG:      stablehlo.slice %arg4 [1:2]

  %t1 = mpmd.transfer %0#0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %t2 = mpmd.transfer %0#1 : (!mesh_1_tensor) -> !mesh_2_tensor

  %1 = mpmd.fragment<mesh="m2", origin=["f2"]> (%t2, %t1, %t1)
    (%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
    mpmd.return %4 : tensor<f32>
  } : (!mesh_2_tensor, !mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor

  func.return %1 : !mesh_2_tensor
}
