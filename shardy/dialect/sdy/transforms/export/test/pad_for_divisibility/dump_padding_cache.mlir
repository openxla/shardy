// RUN: sdy_opt %s -sdy-pad-for-divisibility="dump-padding-cache=true"

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @expand_then_trim
func.func @expand_then_trim(%arg0: tensor<4x7xi32> ) -> tensor<4x7xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] {sdy.padding_kinds = ["kZero"]} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> {sdy.padding_kinds = ["kZero"]} : tensor<4x8xi32>
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{}, {"y"}] %[[ALL_SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> {sdy.padding_kinds = ["kZero"]} : tensor<4x8xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:4, 0:7] {sdy.padding_kinds = ["kZero"]} : (tensor<4x8xi32>) -> tensor<4x7xi32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4x7xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %1 = sdy.all_gather [{}, {"y"}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x7xi32>
  return %1 : tensor<4x7xi32>
}

// CHECK-LABEL: func @generic_op_unknown
func.func @generic_op_unknown(%arg0: tensor<4x7xi32>) -> tensor<4x7xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] {sdy.padding_kinds = ["kZero"]} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[PAD]], %[[PAD]] {sdy.padding_kinds = ["kUnknown"]} : tensor<4x8xi32>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[ADD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> {sdy.padding_kinds = ["kUnknown"]} : tensor<4x8xi32>
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{}, {"y"}] %[[ALL_SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> {sdy.padding_kinds = ["kUnknown"]} : tensor<4x8xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:4, 0:7] {sdy.padding_kinds = ["kUnknown"]} : (tensor<4x8xi32>) -> tensor<4x7xi32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4x7xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x7xi32>
  %1 = sdy.all_slice [{}, {"y"}] %0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %2 = sdy.all_gather [{}, {"y"}] %1 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x7xi32>
  return %2 : tensor<4x7xi32>
}

