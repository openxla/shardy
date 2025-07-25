// RUN: mpmd_opt %s -mpmd-unroll-for-loops 2>&1 | FileCheck %s

// CHECK-LABEL: func @unroll_for_loop_of_three_iterations
func.func @unroll_for_loop_of_three_iterations(%arg0: tensor<10xui32>, %arg1: tensor<10xui32>) -> (tensor<10xui32>, tensor<10xui32>)
  attributes {mesh_shape = #sdy.mesh<["x"=4]>}
{
  // CHECK-NEXT: %[[I0:.*]] = stablehlo.constant {unroll_counter = 0 : ui32} dense<0>
  // CHECK-NEXT: %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %[[I0]], {{.*}} {unroll_counter = 0 : ui32
  // CHECK-NEXT: %[[ADD0_0:.*]] = stablehlo.add %arg0, %[[BCAST0]] {unroll_counter = 0 : ui32
  // CHECK-NEXT: %[[ADD0_1:.*]] = stablehlo.add %arg0, %arg1 {unroll_counter = 0 : ui32
  // CHECK-NEXT: %[[I1:.*]] = stablehlo.constant {unroll_counter = 1 : ui32} dense<1>
  // CHECK-NEXT: %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %[[I1]], {{.*}} {unroll_counter = 1 : ui32
  // CHECK-NEXT: %[[ADD1_0:.*]] = stablehlo.add %[[ADD0_0]], %[[BCAST1]] {unroll_counter = 1 : ui32
  // CHECK-NEXT: %[[ADD1_1:.*]] = stablehlo.add %[[ADD0_0]], %[[ADD0_1]]
  // CHECK-NEXT: %[[I2:.*]] = stablehlo.constant {unroll_counter = 2 : ui32} dense<2>
  // CHECK-NEXT: %[[BCAST2:.*]] = stablehlo.broadcast_in_dim %[[I2]], {{.*}} {unroll_counter = 2 : ui32
  // CHECK-NEXT: %[[ADD2_0:.*]] = stablehlo.add %[[ADD1_0]], %[[BCAST2]] {unroll_counter = 2 : ui32
  // CHECK-NEXT: %[[ADD2_1:.*]] = stablehlo.add %[[ADD1_0]], %[[ADD1_1]] {unroll_counter = 2 : ui32
  // CHECK-NEXT: return %[[ADD2_0]], %[[ADD2_1]]

  // CHECK-NOT: mpmd.for
  %0:2 = mpmd.for (%arg0, %arg1) {iterations = 3 : ui32, unroll_factor = 3 : ui32}
  (%arg2: tensor<10xui32>, %arg3: tensor<10xui32>, %index: tensor<ui32>) {
    %1 = stablehlo.broadcast_in_dim %index, dims = [] : (tensor<ui32>) -> tensor<10xui32>
    %2 = stablehlo.add %arg2, %1 : tensor<10xui32>
    %3 = stablehlo.add %arg2, %arg3 : tensor<10xui32>
    mpmd.return %2, %3 : tensor<10xui32>, tensor<10xui32>
  } : tensor<10xui32>, tensor<10xui32>
  func.return %0#0, %0#1 : tensor<10xui32>, tensor<10xui32>
}
