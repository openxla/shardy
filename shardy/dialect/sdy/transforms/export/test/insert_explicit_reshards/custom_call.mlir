// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xpq = <["x"=4, "p"=3, "q"=5]>
sdy.mesh @mesh_iota = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_non_iota = <["x"=2, "y"=2, "z"=2], device_ids=[7, 6, 5, 4, 3, 2, 1, 0]>
sdy.mesh @mesh_diff_axes_non_iota = <["a"=4, "b"=2], device_ids=[7, 6, 5, 4, 3, 2, 1, 0]>

// CHECK-LABEL: func @custom_call_inspect_sharding
func.func @custom_call_inspect_sharding(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @InspectSharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @InspectSharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @custom_call_xla_megascale_provide_metadata
func.func @custom_call_xla_megascale_provide_metadata(%arg0: tensor<8x2xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @xla.megascale.provide_metadata(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xbf16>) -> tensor<8x2xbf16>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xbf16>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xbf16>
  %0 = stablehlo.custom_call @xla.megascale.provide_metadata(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>} : (tensor<8x2xbf16>) -> tensor<8x2xbf16>
  return %0 : tensor<8x2xbf16>
}

// CHECK-LABEL: func @custom_call_move_to_device
func.func @custom_call_move_to_device(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @MoveToDevice(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @MoveToDevice(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_move_to_host
func.func @custom_call_move_to_host(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @MoveToHost(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @MoveToHost(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_pin_to_device
func.func @custom_call_pin_to_device(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @PinToDevice(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @PinToDevice(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_pin_to_device_sram
func.func @custom_call_pin_to_device_sram(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @PinToDeviceSram(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @PinToDeviceSram(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_layout_constraint
func.func @custom_call_layout_constraint(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @LayoutConstraint(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @LayoutConstraint(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_eigh
func.func @custom_call_eigh(%arg0: tensor<8x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<8x4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}, tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, k]) {i=8, j=4, k=4} need_replication={j, k}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}, {}]> : tensor<8x4x4xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @Eigh(%[[RESHARD1]]) {backend_config = "1, 1, 100, 1e-6", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}]>, <@mesh, [{"x", "y"}, {}]>]>} : (tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x"}, {}, {"y"}]> : tensor<8x4x4xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{}, {"y"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<8x4x4xf32>, tensor<8x4xf32>
  %0:2 = stablehlo.custom_call @Eigh(%arg0) {backend_config = "1, 1, 100, 1e-6", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>)
  return %0#0, %0#1 : tensor<8x4x4xf32>, tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_qr
// TODO(enver): Actually the factors that need replication can be moved to batching dim.
func.func @custom_call_qr(%arg0: tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}) -> (tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}, tensor<8x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"p"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @Qr(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {}, {}]>, <@mesh_xpq, [{"x"}, {}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh_xpq, [{"x"}, {"q"}, {"p"}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh_xpq, [{"x"}, {"p"}]> : tensor<8x3xf32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<8x5x3xf32>, tensor<8x3xf32>
  %0:2 = stablehlo.custom_call @Qr(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>, <@mesh_xpq, [{"x"}, {"p"}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_householder_product
// TODO(enver): Could it instead be better to reshard:
//   sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]>
//   sdy.reshard %arg1 <@mesh_xpq, [{"x"}, {}]>
func.func @custom_call_householder_product(%arg0: tensor<8x12x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>}, %arg1: tensor<8x5xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}]>}) -> (tensor<8x12x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, j, k]) {i=8, j=12, k=16, l=5} need_replication={j, k, l}>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x":(1)2}, {}, {}]> : tensor<8x12x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xpq, [{"x":(1)2}, {}]> : tensor<8x5xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%[[RESHARD1]], %[[RESHARD2]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x":(1)2}, {}, {}]>]>} : (tensor<8x12x16xf32>, tensor<8x5xf32>) -> tensor<8x12x16xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]> : tensor<8x12x16xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x12x16xf32>
  %0 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>]>} : (tensor<8x12x16xf32>, tensor<8x5xf32>) -> tensor<8x12x16xf32>
  return %0 : tensor<8x12x16xf32>
}

// CHECK-LABEL: func @custom_call_erf
func.func @custom_call_erf(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @mhlo.erf(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4xf32>
  %0 = stablehlo.custom_call @mhlo.erf (%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func @custom_call_topk_of_1d
func.func @custom_call_topk_of_1d(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<1xf32>, tensor<1xi32>) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i])->([i], [i]) {i=16} need_replication={i} blocked_propagation={i}>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<16xf32>
  // CHECK: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%[[RESHARD]])
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: return %[[CUSTOM_CALL]]#0, %[[CUSTOM_CALL]]#1 : tensor<1xf32>, tensor<1xi32>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 1 : i64,
        largest = true},
    mhlo.version = 1 : i64}
    : (tensor<16xf32>) -> (tensor<1xf32>, tensor<1xi32>)
  return %0#0, %0#1 : tensor<1xf32>, tensor<1xi32>
}

// CHECK-LABEL: func @custom_call_topk_of_2d
func.func @custom_call_topk_of_2d(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<16x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, tensor<16x1xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} need_replication={j} blocked_propagation={j}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%[[RESHARD1]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>, <@mesh, [{"x", "y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x"}, {}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"y"}, {}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<16x1xf32>, tensor<16x1xi32>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 1 : i64,
        largest = true},
    mhlo.version = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{"y"}, {}]>]>}
    : (tensor<16x8xf32>) -> (tensor<16x1xf32>, tensor<16x1xi32>)
  return %0#0, %0#1 : tensor<16x1xf32>, tensor<16x1xi32>
}

// CHECK-LABEL: func @custom_call_top2_of_2d
func.func @custom_call_top2_of_2d(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<16x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} need_replication={j} blocked_propagation={j}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%[[RESHARD1]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>, <@mesh, [{"x", "y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x"}, {"y"}]> : tensor<16x2xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"y"}, {"x":(1)2}]> : tensor<16x2xi32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<16x2xf32>, tensor<16x2xi32>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 2 : i64,
        largest = true},
    mhlo.version = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>}
    : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xi32>
}

// CHECK-LABEL: func @custom_call_top2_of_2d_all_same_sharding
func.func @custom_call_top2_of_2d_all_same_sharding(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<16x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j], [i, j]) {i=16, j=8} need_replication={j} blocked_propagation={j}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%[[RESHARD1]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>, <@mesh, [{"x", "y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x"}, {"y"}]> : tensor<16x2xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"x"}, {"y"}]> : tensor<16x2xi32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<16x2xf32>, tensor<16x2xi32>
  %0:2 = stablehlo.custom_call @mhlo.topk(%arg0) {
    mhlo.attributes = {
        k = 2 : i64,
        largest = true},
    mhlo.version = 1 : i64, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>]>}
    : (tensor<16x8xf32>) -> (tensor<16x2xf32>, tensor<16x2xi32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xi32>
}

// CHECK-LABEL: func @custom_call_approx_topk
func.func @custom_call_approx_topk(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(2)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {"x":(2)2}]>
  // CHECK-NEXT: %[[APPROX_TOPK:.*]]:2 = stablehlo.custom_call @ApproxTopK(%[[RESHARD1]], %arg1, %arg2, %arg3)
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[APPROX_TOPK]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[APPROX_TOPK]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @custom_call_approx_topk_majority_does_not_fit_all_factors
func.func @custom_call_approx_topk_majority_does_not_fit_all_factors(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(1)2}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[APPROX_TOPK:.*]]:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg1, %arg2, %arg3)
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[APPROX_TOPK]]#0 <@mesh, [{}, {"x":(1)2}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[APPROX_TOPK]]#1 <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @ApproxTopK(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x":(1)2}]>, <@mesh, [{}, {"y"}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @unregistered_custom_call_with_existing_rule
func.func @unregistered_custom_call_with_existing_rule(%arg0: tensor<4x2xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x4xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}){
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @unregistered_custom_call_without_existing_rule
func.func @unregistered_custom_call_without_existing_rule(%arg0: tensor<4x2xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x4xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}){
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @custom_call_without_rule_normalizes_replicated_operand
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @custom_call_without_rule_normalizes_replicated_operand(
    %arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{}, {}]>},
    %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<4x2xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[R]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: return %[[CC]] : tensor<4x2xf32>
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<4x2xf32>, tensor<4x2xf32>) -> tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func @custom_call_with_custom_rule_normalizes_replicated_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @custom_call_with_custom_rule_normalizes_replicated_result(
    %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{}, {}]>}) {
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[ARG0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}, custom>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[CC]] <@mesh_xpq, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[R]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}, custom>, sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{}, {}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @custom_call_conflicting_non_replicated_meshes_untouched
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {}]>})
func.func @custom_call_conflicting_non_replicated_meshes_untouched(
    %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {}]>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[ARG0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[CC]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// Non-custom sharding rule where all tensor dims map to different non-overlapping
// factors and @mesh_non_iota vs @mesh_iota have the same axes with disjoint
// mesh axes across factors: %arg0 keeps its "x" sharding while moving onto
// @mesh_iota.
// CHECK-LABEL: func @custom_call_distinct_factors_non_custom_rule_disjoint_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
func.func @custom_call_distinct_factors_non_custom_rule_disjoint_axes(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"z"}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_iota, [{"x"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[RESHARD0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8}>} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[CC]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>,
    sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8}>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Non-custom sharding rule where all tensor dims map to different non-overlapping
// factors, and %arg0 (@mesh_non_iota) and %arg1 (@mesh_iota) both shard along
// "x": findCommonAxesHeuristic keeps mesh axes mutually disjoint across
// factors, so the smaller %arg0 drops "x" to replicated on @mesh_iota.
// CHECK-LABEL: func @custom_call_distinct_factors_non_custom_rule_conflicting_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>})
func.func @custom_call_distinct_factors_non_custom_rule_conflicting_axes(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"z"}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_iota, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[RESHARD0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=16, m=8, n=8}>} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[CC]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>,
    sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=16, m=8, n=8}>
  } : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Non-custom sharding rule where all tensor dims map to different non-overlapping
// factors, but the non-iota mesh has a different axis structure than
// @mesh_iota: prePivotForeignMeshes wipes %arg0 via a 2-step replicated pivot.
// CHECK-LABEL: func @custom_call_distinct_factors_non_custom_rule_different_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_diff_axes_non_iota, [{"a"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
func.func @custom_call_distinct_factors_non_custom_rule_different_axes(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_diff_axes_non_iota, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"z"}]>}) {
  // CHECK-NEXT: %[[PIVOT0:.*]] = sdy.reshard %[[ARG0]] <@mesh_diff_axes_non_iota, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[PIVOT1:.*]] = sdy.reshard %[[PIVOT0]] <@mesh_iota, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[PIVOT1]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8}>} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[CC]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>,
    sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8}>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Non-custom sharding rule where all non-overlapping factors are marked
// need_replication: distributeAxisRefsToBatchingFactors clears every factor
// because no batching factors exist, so all operands/results reshard to
// replicated on @mesh_iota.
// CHECK-LABEL: func @custom_call_distinct_need_replication_factors_non_custom_rule
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
func.func @custom_call_distinct_need_replication_factors_non_custom_rule(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"z"}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_iota, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG1]] <@mesh_iota, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[RESHARD0]], %[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8} need_replication={i, j, k, l, m, n}>} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[CC]] <@mesh_iota, [{}, {"z"}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[RESHARD_OUT]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>,
    sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8} need_replication={i, j, k, l, m, n}>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// Same distinct-factor sharding rule marked `custom`: InsertExplicitReshardsPass
// skips ShardingProjection and only normalizes replicated tensors, leaving the
// mixed non-replicated @mesh_non_iota and @mesh_iota shardings untouched.
// CHECK-LABEL: func @custom_call_distinct_factors_custom_rule_untouched
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
func.func @custom_call_distinct_factors_custom_rule_untouched(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"z"}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @foo(%[[ARG0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8}, custom>} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: return %[[CC]] : tensor<8x8xf32>
  %0 = stablehlo.custom_call @foo(%arg0, %arg1) {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"z"}]>]>,
    sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([m, n]) {i=8, j=8, k=8, l=8, m=8, n=8}, custom>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @custom_call_multi_result_mismatched_unreduced
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}], unreduced={"z"}>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>})
func.func @custom_call_multi_result_mismatched_unreduced(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}], unreduced={"z"}>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}], unreduced={"z"}>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[IN_AR:.*]] = sdy.all_reduce {"z"} %[[ARG0]] out_sharding=<@mesh_iota, [{"x"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[CC:.*]]:2 = stablehlo.custom_call @foo(%[[IN_AR]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}]>, <@mesh_iota, [{"x"}, {}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([i, j], [k, l]) {i=8, j=8, k=8, l=8}, custom>} : (tensor<8x8xf32>, tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>)
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"z"} %[[CC]]#0 out_sharding=<@mesh_iota, [{"x"}, {}], unreduced={"z"}> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[R2U]], %[[CC]]#1 : tensor<8x8xf32>, tensor<8x8xf32>
  %0:2 = stablehlo.custom_call @foo(%arg0, %arg1) {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}], unreduced={"z"}>, <@mesh_iota, [{"x"}, {}]>]>,
    sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k, l])->([i, j], [k, l]) {i=8, j=8, k=8, l=8}, custom>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>)
  return %0#0, %0#1 : tensor<8x8xf32>, tensor<8x8xf32>
}
