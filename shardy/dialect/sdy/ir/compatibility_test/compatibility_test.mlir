// Smoke test:
// RUN: sdy_opt %s.bc | FileCheck %s
// RUN: sdy_opt %s.bc | sdy_translate --serialize | sdy_opt | FileCheck %s
// RUN: sdy_opt %s.bc | sdy_translate --serialize --strip-debuginfo | sdy_opt | FileCheck %s
// RUN: sdy_translate --deserialize %s.bc | sdy_opt | FileCheck %s
//
// Backward compatibility test:
// RUN: sdy_translate --serialize %s | sdy_opt > %t.0
// RUN: sdy_opt %s > %t.1
// RUN: diff %t.0 %t.1
//
// Forward compatibility test:
// RUN: sdy_translate %s --serialize -strip-debuginfo > %t.2
// RUN: diff %s.bc %t.2

// CHECK: sdy.mesh @empty_mesh = <[]>
sdy.mesh @empty_mesh = <[]>

// CHECK: sdy.mesh @maximal_mesh_1 = <[], device_ids=[0]>
sdy.mesh @maximal_mesh_1 = <[], device_ids=[0]>

// CHECK: sdy.mesh @maximal_mesh_2 = <[], device_ids=[3]>
sdy.mesh @maximal_mesh_2 = <[], device_ids=[3]>

// CHECK: sdy.mesh @mesh_xy = <["x"=2, "y"=4]>
sdy.mesh @mesh_xy = <["x"=2, "y"=4]>

// CHECK: sdy.mesh @mesh_x_non_iota_device_ids = <["x"=4], device_ids=[0, 3, 2, 1]>
sdy.mesh @mesh_x_non_iota_device_ids = <["x"=4], device_ids=[0, 3, 2, 1]>

// CHECK: sdy.mesh @mesh_xyz = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_xyz = <["x"=2, "y"=2, "z"=2]>

// CHECK-LABEL: func @sharding_constraint
func.func @sharding_constraint(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.sharding_constraint %arg0 <@mesh_xy, [{}, {"x"}], replicated={"y"}>
  %0 = sdy.sharding_constraint %arg0 <@mesh_xy, [{}, {"x"}], replicated={"y"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @reshard
func.func @reshard(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK-NEXT: sdy.reshard %arg0 <@mesh_xy, [{}, {"y"}], replicated={"x"}>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{}, {"y"}], replicated={"x"}> : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@mesh_xy, [{"x", ?}, {?}]>] out_shardings=[<@mesh_xy, [{"x", ?}, {?}]>] manual_axes={"x"} (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:       sdy.return %arg1 : tensor<8x32xf32>
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_xy, [{"x", ?}, {?}]>] out_shardings=[<@mesh_xy, [{"x", ?}, {?}]>] manual_axes={"x"} (%arg1: tensor<8x32xf32>) {
    sdy.return %arg1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// CHECK-LABEL: func @sharding_group
func.func @sharding_group(%arg0: tensor<8xf32>) {
  // CHECK sdy.sharding_group %arg0 group_id=21 type=AS  : tensor<8xf32>
  sdy.sharding_group %arg0 group_id=21 : tensor<8xf32>
  func.return
}

// CHECK-LABEL: func @constant
func.func @constant() {
  // CHECK-NEXT: sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  func.return
}

// CHECK-LABEL: func @data_flow_edge
func.func @data_flow_edge(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: sdy.data_flow_edge %arg0
  // CHECK-NEXT: sdy.data_flow_edge %arg1 sharding=<@mesh_x_non_iota_device_ids, [{"x"}, {?}]>
  %1 = sdy.data_flow_edge %arg0 : tensor<32x96xf32>
  %2 = sdy.data_flow_edge %arg1 sharding=<@mesh_x_non_iota_device_ids, [{"x"}, {?}]> : tensor<32x96xf32>
  return %1, %2 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @propagation_barrier
func.func @propagation_barrier(%arg0 : tensor<8xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<8x16xf32>)
    -> (tensor<8xf32>, tensor<16x8xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: sdy.propagation_barrier %arg0 allowed_direction=NONE
  // CHECK-NEXT: sdy.propagation_barrier %arg1 allowed_direction=FORWARD
  // CHECK-NEXT: sdy.propagation_barrier %arg2 allowed_direction=BACKWARD
  %0 = sdy.propagation_barrier %arg0 allowed_direction=NONE : tensor<8xf32>
  %1 = sdy.propagation_barrier %arg1 allowed_direction=FORWARD : tensor<16x8xf32>
  %2 = sdy.propagation_barrier %arg2 allowed_direction=BACKWARD : tensor<8x16xf32>
  return %0, %1, %2 : tensor<8xf32>, tensor<16x8xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @named_computation
func.func @named_computation(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
  // CHECK-NEXT:   sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @tensor_sharding
func.func @tensor_sharding(%arg0 : tensor<8x8xf32>, %arg1 : tensor<8x8xf32>) -> (tensor<64xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.custom_call @bar(%arg0, %arg1)
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@mesh_xy, [{"x", "y"}]>, <@mesh_xy, [{"x"}p0, {"y":(1)2}p123]>]>
  %0:2 = stablehlo.custom_call @bar(%arg0, %arg1)
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x", "y"}]>, <@mesh_xy, [{"x"}p0, {"y":(1)2}p123]>]>}
    : (tensor<8x8xf32>, tensor<8x8xf32>) -> (tensor<64xf32>, tensor<8x8xf32>)
  return %0#0, %0#1 : tensor<64xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @tensor_sharding_on_parameter_result
// CHECK-SAME{LITERAL}: (%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}p2]>}) -> (tensor<64xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x", "y"}]>})
func.func @tensor_sharding_on_parameter_result(%arg0 : tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}p2]>})
  -> (tensor<64xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x", "y"}]>}) {
  %0 = stablehlo.custom_call @foo(%arg0) : (tensor<8x8xf32>) -> (tensor<64xf32>)
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func @tensor_sharding_scalar
// CHECK-SAME{LITERAL}: (%arg0: tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_xy, []>}) -> (tensor<64xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x", "y"}]>})
func.func @tensor_sharding_scalar(%arg0 : tensor<f32> {sdy.sharding = #sdy.sharding<@mesh_xy, []>})
  -> (tensor<64xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x", "y"}]>}) {
  %0 = stablehlo.custom_call @foo(%arg0) : (tensor<f32>) -> (tensor<64xf32>)
  return %0 : tensor<64xf32>
}

// CHECK-LABEL: func @tensor_sharding_dynamic_shape
func.func @tensor_sharding_dynamic_shape(%arg0 : tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  // CHECK-NEXT: stablehlo.custom_call @bar(%arg0)
  // CHECK-SAME{LITERAL}: #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {}], replicated={"z"}>]>
  %0 = stablehlo.custom_call @bar(%arg0)
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {}], replicated={"z"}>]>}
    : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @sharding_rule_scalar
func.func @sharding_rule_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([], [])->([]), custom>}
  %0 = stablehlo.custom_call @foo(%arg0, %arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([], [])->([]), custom>} :
    (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @sharding_rule_tensor
func.func @sharding_rule_tensor(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @sharding_rule_tensor_with_many_dimensions
func.func @sharding_rule_tensor_with_many_dimensions(%arg0: tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x8xf32> {
  // CHECK:      #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10])
  // CHECK-SAME: ->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8z_9z_10])
  // CHECK-SAME: {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2, z_3=2, z_4=2, z_5=2, z_6=2, z_7=2, z_8=2, z_9=2, z_10=2}>} :
  %0 = stablehlo.custom_call @foo(%arg0)
    {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10])->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8z_9z_10]) {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2, z_3=2, z_4=2, z_5=2, z_6=2, z_7=2, z_8=2, z_9=2, z_10=2}>}
    : (tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x8xf32>
  return %0 : tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x8xf32>
}

// CHECK-LABEL: func @custom_sharding_rule_custom_call
func.func @custom_sharding_rule_custom_call(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom>}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}
