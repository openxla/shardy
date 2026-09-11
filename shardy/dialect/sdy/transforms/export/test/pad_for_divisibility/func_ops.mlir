// RUN: sdy_opt %s -sdy-pad-for-divisibility -split-input-file -verify-diagnostics | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// expected-error @+2  {{argument #0 has a non-divisible sharding}}
// expected-error @+1  {{failed to legalize operation 'func.func'}}
func.func @indivisible_input(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> tensor<7x8xf32> {
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// expected-error @+2 {{result #0 has a non-divisible sharding}}
// expected-error @+1 {{failed to legalize operation 'func.func'}}
func.func @indivisible_output(
  %arg0: tensor<7x8xf32>)
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Tests subroutine argument padding.
// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK:         %[[CALL:.*]] = call @subroutine_indivisible_arg(%[[ALL_SLICE]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:         %[[GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[CALL]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK:         %[[TRIM:.*]] = stablehlo.slice %[[GATHER]] [0:7, 0:8]
// CHECK:         return %[[TRIM]] : tensor<7x8xf32>
// CHECK:       }

// CHECK-LABEL: func.func private @subroutine_indivisible_arg(
// CHECK-SAME:                                                %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
// CHECK-SAME:                                                -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
// CHECK:         %[[GATHER:.*]] = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK:         %[[SLICE:.*]] = stablehlo.slice %[[GATHER]] [0:7, 0:8]
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK:         return %[[ALL_SLICE]] : tensor<8x8xf32>
// CHECK:       }
func.func @main(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  %1 = func.call @subroutine_indivisible_arg(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x8xf32>) -> tensor<7x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %2 : tensor<7x8xf32>
}

func.func private @subroutine_indivisible_arg(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  %1 = sdy.all_slice [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %1 : tensor<7x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Tests subroutine result padding.
// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK:         %[[CALL:.*]] = call @subroutine_indivisible_output(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x8xf32>) -> tensor<8x8xf32>
// CHECK:         %[[GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[CALL]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK:         %[[TRIM:.*]] = stablehlo.slice %[[GATHER]] [0:7, 0:8]
// CHECK:         return %[[TRIM]] : tensor<7x8xf32>
// CHECK:       }

// CHECK-LABEL: func.func private @subroutine_indivisible_output(
// CHECK-SAME:                                                   %arg0: tensor<7x8xf32>)
// CHECK-SAME:                                                   -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK:         return %[[ALL_SLICE]] : tensor<8x8xf32>
// CHECK:       }
func.func @main(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  %0 = func.call @subroutine_indivisible_output(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x8xf32>) -> tensor<7x8xf32>
  %1 = sdy.all_gather [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %1 : tensor<7x8xf32>
}

func.func private @subroutine_indivisible_output(
  %arg0: tensor<7x8xf32>)
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Tests nested subroutine calls with indivisible sharding.
// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK:         %[[CALL:.*]] = call @callee_outer(%[[ALL_SLICE]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:         %[[GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[CALL]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK:         %[[TRIM:.*]] = stablehlo.slice %[[GATHER]] [0:7, 0:8]
// CHECK:         return %[[TRIM]] : tensor<7x8xf32>
// CHECK:       }

// CHECK-LABEL: func.func private @callee_outer(
// CHECK-SAME:                                  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
// CHECK-SAME:                                  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
// CHECK:         %[[CALL:.*]] = call @callee_inner(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:         return %[[CALL]] : tensor<8x8xf32>
// CHECK:       }

// CHECK-LABEL: func.func private @callee_inner(
// CHECK-SAME:                                  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
// CHECK-SAME:                                  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
// CHECK:         %[[GATHER:.*]] = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x8xf32>
// CHECK:         %[[SLICE:.*]] = stablehlo.slice %[[GATHER]] [0:7, 0:8]
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK:         return %[[ALL_SLICE]] : tensor<8x8xf32>
// CHECK:       }
func.func @main(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  %1 = func.call @callee_outer(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x8xf32>) -> tensor<7x8xf32>
  %2 = sdy.all_gather [{"x"}, {}] %1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  return %2 : tensor<7x8xf32>
}

func.func private @callee_outer(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = func.call @callee_inner(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<7x8xf32>) -> tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

func.func private @callee_inner(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  %1 = sdy.all_slice [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %1 : tensor<7x8xf32>
}
