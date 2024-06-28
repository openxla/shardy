// RUN: sdy_opt %s -- 2>&1 | FileCheck %s

sdy.mesh @mesh = <"a"=2,"b"=2>

// CHECK-LABEL: func @backward
func.func @backward(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @backward_with_sharding
func.func @backward_with_sharding(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK sdy.propagation_barrier %arg0 allowed_direction=BACKWARD {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a",?}]>]>} : tensor<8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=BACKWARD {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{"a",?}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @forward
func.func @forward(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8xf32>
  return %0 : tensor<8xf32>
}


// CHECK-LABEL: func @none
func.func @none(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK sdy.propagation_barrier %arg0 allowed_direction=NONE : tensor<8xf32>
  %0 = sdy.propagation_barrier %arg0 allowed_direction=NONE : tensor<8xf32>
  return %0 : tensor<8xf32>
}
