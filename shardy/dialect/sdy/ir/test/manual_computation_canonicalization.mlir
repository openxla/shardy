// RUN: sdy_opt %s -canonicalize | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=1, "c"=1]>

// CHECK-LABEL: func @unused_args
func.func @unused_args(%arg0: tensor<8xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<16xf32>) -> tensor<32x32xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg1)
  // CHECK-SAME:     in_shardings=[<@mesh, [{"a"}, {}]>]
  // CHECK-SAME:     out_shardings=[<@mesh, [{"a"}, {}]>]
  // CHECK-SAME:     manual_axes={"a"} (%arg3: tensor<16x32xf32>) {
  // CHECK-NEXT:   sdy.return %arg3
  // CHECK-NEXT: }
  %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"a"}]>, <@mesh, [{"a"}, {}]>, <@mesh, [{"a"}]>] out_shardings=[<@mesh, [{"a"}, {}]>]
      manual_axes={"a"} (%arg3: tensor<4xf32>, %arg4: tensor<16x32xf32>, %arg5: tensor<8xf32>) {
    sdy.return %arg4 : tensor<16x32xf32>
  } : (tensor<8xf32>, tensor<32x32xf32>, tensor<16xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func @inline_no_inputs_outputs
func.func @inline_no_inputs_outputs(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: stablehlo.custom_call @foo() {has_side_effect = true} : () -> ()
  // CHECK-NEXT: return %arg0 : tensor<8xf32>
  sdy.manual_computation() in_shardings=[] out_shardings=[]
      manual_axes={} () {
    stablehlo.custom_call @foo() {has_side_effect = true} : () -> ()
    sdy.return
  } : () -> ()
  return %arg0: tensor<8xf32>
}

// CHECK-LABEL: func @redundant_manual_axes
func.func @redundant_manual_axes(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %0 = stablehlo.custom_call @foo(%arg0) {has_side_effect = true} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %0 : tensor<8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", "c", "a"}]>] out_shardings=[<@mesh, [{"b", "c", "a"}]>]
      manual_axes={"b", "c"}  (%arg1: tensor<8xf32>) {
    %1 = stablehlo.custom_call @foo(%arg1) {has_side_effect = true} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> (tensor<8xf32>)
  return %0: tensor<8xf32>
}

// CHECK-LABEL: func @redundant_manual_axes_inlined_mesh
func.func @redundant_manual_axes_inlined_mesh(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %0 = stablehlo.custom_call @foo(%arg0) {has_side_effect = true} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %0 : tensor<8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<mesh<["a"=2, "b"=1, "c"=1]>, [{"b", "c", "a"}]>] out_shardings=[<mesh<["a"=2, "b"=1, "c"=1]>, [{"b", "c", "a"}]>]
      manual_axes={"b", "c"}  (%arg1: tensor<8xf32>) {
    %1 = stablehlo.custom_call @foo(%arg1) {has_side_effect = true} : (tensor<8xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> (tensor<8xf32>)
  return %0: tensor<8xf32>
}

// Top manual_computation is inlinable, middle one isn't, most inner is.
// CHECK-LABEL: func @nested_two_inlinable
func.func @nested_two_inlinable(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"a"}]>] out_shardings=[<@mesh, [{}, {"a"}]>] manual_axes={"a"} (%arg1: tensor<4x4xf32>) {
  // CHECK-NEXT:   %[[WSC:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{}, {"c"}]> : tensor<4x4xf32>
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %[[WSC]], %[[WSC]] : tensor<4x4xf32>
  // CHECK-NEXT:   sdy.return %[[MULT]] : tensor<4x4xf32>
  // CHECK-NEXT: } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[MAN_COMP]], %[[MAN_COMP]] : tensor<4x8xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<4x8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b"}, {}]>] out_shardings=[<@mesh, [{"b"}, {}]>] manual_axes={"b"} (%arg1: tensor<4x8xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{}, {"a"}]>] out_shardings=[<@mesh, [{}, {"a"}]>] manual_axes={"a"} (%arg2: tensor<4x4xf32>) {
      %3 = sdy.sharding_constraint %arg2 <@mesh, [{}, {"c"}]> : tensor<4x4xf32>
      %4 = sdy.manual_computation(%3) in_shardings=[<@mesh, [{}, {"c"}]>] out_shardings=[<@mesh, [{}, {"c"}]>] manual_axes={"c"} (%arg3: tensor<4x4xf32>) {
        %5 = stablehlo.multiply %arg3, %arg3 : tensor<4x4xf32>
        sdy.return %5 : tensor<4x4xf32>
      } : (tensor<4x4xf32>) -> tensor<4x4xf32>
      sdy.return %4 : tensor<4x4xf32>
    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %2 = stablehlo.add %1, %1 : tensor<4x8xf32>
    sdy.return %2 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
