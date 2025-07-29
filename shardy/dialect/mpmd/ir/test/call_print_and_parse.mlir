// RUN: mpmd_opt %s 2>&1 -split-input-file | FileCheck %s

// CHECK-LABEL: module
// CHECK: func.func public @main
func.func public @main(%arg0: tensor<3xi32>) -> (tensor<i32>) {
  %0 = stablehlo.slice %arg0 [0:1] : (tensor<3xi32>) -> tensor<1xi32>
  %1 = stablehlo.reshape %0 : (tensor<1xi32>) -> tensor<i32>
  %2 = stablehlo.slice %arg0 [1:2] : (tensor<3xi32>) -> tensor<1xi32>
  %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
  %4 = stablehlo.slice %arg0 [2:3] : (tensor<3xi32>) -> tensor<1xi32>
  %5 = stablehlo.reshape %4 : (tensor<1xi32>) -> tensor<i32>
  %6 = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[i7:.+]] = mpmd.call @fn(%[[i6:.+]], %[[i1:.+]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %7 = mpmd.call @fn(%6, %1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %[[i8:.+]] = mpmd.call @fn(%[[i7]], %[[i3:.+]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %8 = mpmd.call @fn(%7, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: %{{.*}} = mpmd.call @fn(%[[i8]], %[[i5:.+]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %9 = mpmd.call @fn(%8, %5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %9 : tensor<i32>
}

func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

!m3elements = !mpmd.mesh_tensor<"mesh1", tensor<3xi32>>
!m_scalar = !mpmd.mesh_tensor<"mesh1", tensor<i32>>

// CHECK-LABEL: module
// CHECK: func.func public @main
func.func public @main(%arg0: !m3elements) -> (!m_scalar) attributes {
    "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK: %[[SPLIT_INPUTS:.*]]:3 = mpmd.fragment<mesh="mesh1", origin=["split"]> (%arg0)
  %split_inputs:3 = mpmd.fragment<mesh="mesh1", origin=["split"]> (%arg0) (%arg1: tensor<3xi32>) {
    %0 = stablehlo.slice %arg1 [0:1] : (tensor<3xi32>) -> tensor<1xi32>
    %1 = stablehlo.reshape %0 : (tensor<1xi32>) -> tensor<i32>
    %2 = stablehlo.slice %arg1 [1:2] : (tensor<3xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = stablehlo.slice %arg1 [2:3] : (tensor<3xi32>) -> tensor<1xi32>
    %5 = stablehlo.reshape %4 : (tensor<1xi32>) -> tensor<i32>
    mpmd.return %1, %3, %5 : tensor<i32>, tensor<i32>, tensor<i32>
  } : (!m3elements) -> (!m_scalar, !m_scalar, !m_scalar)
  // CHECK-DAG: %[[INIT_COUNTER:.*]] = mpmd.fragment<mesh="mesh1", origin=["split"]> ()
  %init_counter = mpmd.fragment<mesh="mesh1", origin=["split"]> () () {
    %6 = stablehlo.constant dense<0> : tensor<i32>
    mpmd.return %6 : tensor<i32>
  } : () -> !m_scalar
  // CHECK-DAG: %[[I1:.*]] = mpmd.call @fn(%[[INIT_COUNTER]], %[[SPLIT_INPUTS]]#0)
  %7 = mpmd.call @fn(%init_counter, %split_inputs#0) : (!m_scalar, !m_scalar) -> !m_scalar
  // CHECK-NEXT: %[[I2:.*]] = mpmd.call @fn(%[[I1]], %[[SPLIT_INPUTS]]#1)
  %8 = mpmd.call @fn(%7, %split_inputs#1) : (!m_scalar, !m_scalar) -> !m_scalar
  // CHECK-NEXT: mpmd.call @fn(%[[I2]], %[[SPLIT_INPUTS]]#2)
  %9 = mpmd.call @fn(%8, %split_inputs#2) : (!m_scalar, !m_scalar) -> !m_scalar
  return %9 : !m_scalar
}

func.func private @fn(%arg0: !m_scalar, %arg1: !m_scalar) -> !m_scalar attributes {
    "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  %add = mpmd.fragment<mesh="mesh1", origin=["add"]> (%arg0, %arg1) (%arg2: tensor<i32>, %arg3: tensor<i32>) {
    %0 = stablehlo.add %arg2, %arg3 : tensor<i32>
    mpmd.return %0 : tensor<i32>
  } : (!m_scalar, !m_scalar) -> !m_scalar
  return %add : !m_scalar
}
