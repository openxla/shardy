// RUN: sdy_opt %s -canonicalize | FileCheck %s

sdy.mesh @mesh = <"a"=2>

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
