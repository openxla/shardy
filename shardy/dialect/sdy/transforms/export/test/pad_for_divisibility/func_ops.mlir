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

// Tests subroutine argument padding. Unlike @main, subroutines are permitted to
// have indivisible sharded arguments and results:
// 1. The call site in @main pads the argument tensor<7x8xf32> -> tensor<8x8xf32>
//    via target materialization (stablehlo.pad).
// 2. The subroutine signature is converted to accept tensor<8x8xf32>.
// 3. Inside the subroutine body, a source materialization (stablehlo.slice) slices
//    the argument back to tensor<7x8xf32> for the unpadded operations.
// 4. The call result is sliced back to tensor<7x8xf32> at the call site.

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[CALL:.*]] = call @subroutine_indivisible_arg(%[[PAD]])
// CHECK:         %[[SLICE:.*]] = stablehlo.slice %[[CALL]] [0:7, 0:8]
// CHECK:         return %[[SLICE]] : tensor<7x8xf32>
// CHECK:       }

// CHECK-LABEL: func.func private @subroutine_indivisible_arg(
// CHECK-SAME:                                                %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
// CHECK-SAME:                                                -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
// CHECK:         %[[SLICE:.*]] = stablehlo.slice %arg0 [0:7, 0:8]
// CHECK:         %[[GATHER:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %[[GATHER]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[ALL_SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK:         return %[[ALL_SLICE]] : tensor<8x8xf32>
// CHECK:       }
func.func @main(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  %0 = func.call @subroutine_indivisible_arg(%arg0) : (tensor<7x8xf32>) -> tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
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

// Tests subroutine result padding. When a subroutine returns a tensor with
// indivisible sharding:
// 1. The subroutine signature is converted to return divisible shape tensor<8x8xf32>.
// 2. The return operand inside the subroutine is padded to tensor<8x8xf32>.
// 3. At the call site in @main, the result is sliced back to tensor<7x8xf32> via
//    source materialization (stablehlo.slice).

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK:         %[[CALL:.*]] = call @subroutine_indivisible_output(%arg0)
// CHECK:         %[[SLICE:.*]] = stablehlo.slice %[[CALL]] [0:7, 0:8]
// CHECK:         return %[[SLICE]] : tensor<7x8xf32>
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
  %0 = func.call @subroutine_indivisible_output(%arg0) : (tensor<7x8xf32>) -> tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

func.func private @subroutine_indivisible_output(
  %arg0: tensor<7x8xf32>)
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Tests nested subroutine calls with indivisible sharding:
// @main calls @callee_outer, which in turn calls @callee_inner. Both callees have
// indivisible sharded arguments and results.
// Verifies that:
// 1. Pad and slice materializations are placed properly at each call and return boundary.
// 2. Both subroutine signatures are updated to divisible shapes (tensor<8x8xf32>).
// 3. Types and shardings reconcile consistently across multiple call levels.

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
// CHECK-DAG:     %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[CALL:.*]] = call @callee_outer(%[[PAD]])
// CHECK:         %[[SLICE:.*]] = stablehlo.slice %[[CALL]] [0:7, 0:8]
// CHECK:         return %[[SLICE]] : tensor<7x8xf32>
// CHECK:       }

// CHECK-LABEL: func.func private @callee_outer(
// CHECK-SAME:                                  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
// CHECK-SAME:                                  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
// CHECK:         %[[SLICE1:.*]] = stablehlo.slice %arg0 [0:7, 0:8]
// CHECK-DAG:     %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD1:.*]] = stablehlo.pad %[[SLICE1]], %[[CST1]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[CALL1:.*]] = call @callee_inner(%[[PAD1]])
// CHECK:         %[[SLICE2:.*]] = stablehlo.slice %[[CALL1]] [0:7, 0:8]
// CHECK-DAG:     %[[CST2:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD2:.*]] = stablehlo.pad %[[SLICE2]], %[[CST2]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         return %[[PAD2]] : tensor<8x8xf32>
// CHECK:       }

// CHECK-LABEL: func.func private @callee_inner(
// CHECK-SAME:                                  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
// CHECK-SAME:                                  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
// CHECK:         %[[SLICE3:.*]] = stablehlo.slice %arg0 [0:7, 0:8]
// CHECK:         %[[GATHER3:.*]] = sdy.all_gather [{"x"}, {}] %[[SLICE3]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
// CHECK-DAG:     %[[CST3:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:     %[[PAD3:.*]] = stablehlo.pad %[[GATHER3]], %[[CST3]], low = [0, 0], high = [1, 0], interior = [0, 0]
// CHECK:         %[[ALL_SLICE3:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD3]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<8x8xf32>
// CHECK:         return %[[ALL_SLICE3]] : tensor<8x8xf32>
// CHECK:       }
func.func @main(%arg0: tensor<7x8xf32>) -> tensor<7x8xf32> {
  %0 = func.call @callee_outer(%arg0) : (tensor<7x8xf32>) -> tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

func.func private @callee_outer(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = func.call @callee_inner(%arg0) : (tensor<7x8xf32>) -> tensor<7x8xf32>
  return %0 : tensor<7x8xf32>
}

func.func private @callee_inner(
  %arg0: tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>})
  -> (tensor<7x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<7x8xf32>
  %1 = sdy.all_slice [{"x"}, {}] %0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<7x8xf32>
  return %1 : tensor<7x8xf32>
}
