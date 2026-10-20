// RUN: sdy_opt %s -sdy-remove-propagation-barriers | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @propagation_barrier_ops
func.func @propagation_barrier_ops(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %[[ADD0]], %arg0
  // CHECK-NEXT: return %[[ADD1]]
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : tensor<8x8xf32>
  %2 = sdy.propagation_barrier %1 allowed_direction=FORWARD : tensor<8x8xf32>
  %3 = stablehlo.add %2, %arg0 : tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}
