// RUN: mpmd_opt %s -mpmd-insert-nameless-clone-of-negligible-ops 2>&1 | FileCheck %s

// CHECK-LABEL: func @main
func.func @main() -> (tensor<4x8xf32>, tensor<4x8xf32>)
  attributes {mesh_shape = #sdy.mesh<["x"=1]>}
{
  // The name computation remains unmodified.
  // CHECK-NEXT: %[[NC:.*]]:3 = mpmd.named_computation<"nc"> () () {
  // CHECK-NEXT:   stablehlo.constant dense<0.000000e+00>
  // CHECK-NEXT:   stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT:   stablehlo.add
  // CHECK-NEXT:   stablehlo.constant dense<2.000000e+00>
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[PUSHED1:.*]] = stablehlo.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[PUSHED2:.*]] = stablehlo.constant dense<2.000000e+00>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[PUSHED1]], %[[NC]]#1
  // CHECK-NEXT: return %[[ADD]], %[[NC]]#2
  %1:3 = mpmd.named_computation<"nc"> () () {
    // Used only within the named_computation. Should not be pushed out.
    %2 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
    // Used within the named_computation and by the return op. Should be pushed
    // out.
    %3 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    %4 = stablehlo.add %2, %3 : tensor<4x8xf32>
    // Used only by the return op. Should be pushed out.
    %5 = stablehlo.constant dense<2.0> : tensor<4x8xf32>
    mpmd.return %3, %4, %5 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : () -> (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>)
  %5 = stablehlo.add %1#0, %1#1 : tensor<4x8xf32>
  // The %1#2 should not be replaced.
  func.return %5, %1#2 : tensor<4x8xf32>, tensor<4x8xf32>
}
