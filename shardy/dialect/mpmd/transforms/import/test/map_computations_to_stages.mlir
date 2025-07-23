// RUN: mpmd_opt %s -mpmd-map-named-ops-to-mpmd-ops='assignment=c1@m1/0,c2@m2/0,c3@m1/1' -mpmd-introduce-transfers 2>&1 | FileCheck %s

// CHECK-LABEL: func @simple_assignment
func.func @simple_assignment(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["c1"], stage=0>
  // CHECK: mpmd.transfer
  // CHECK: mpmd.fragment<mesh="m2", origin=["c2"], stage=0>
  // CHECK: mpmd.transfer
  // CHECK: mpmd.fragment<mesh="m1", origin=["c3"], stage=1>
  %1 = mpmd.named_computation<"c1"> (%arg0) (%arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %2 = mpmd.named_computation<"c2"> (%1) (%arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %3 = mpmd.named_computation<"c3"> (%2) (%arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %3 : tensor<4x8xf32>
}
