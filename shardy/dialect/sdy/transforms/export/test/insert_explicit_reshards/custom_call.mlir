// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xpq = <["x"=4, "p"=3, "q"=5]>

// CHECK-LABEL: func @custom_call_compact_wy_helper
func.func @custom_call_compact_wy_helper(%arg0: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<128x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<128x128xf32>
  %0 = stablehlo.custom_call @CompactWyHelper(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @custom_call_inspect_sharding
func.func @custom_call_inspect_sharding(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=4, j=8}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @InspectSharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x8xf32>
  %0 = stablehlo.custom_call @InspectSharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @custom_call_x64_combine
func.func @custom_call_x64_combine(%arg0: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) -> (tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {"y"}]> : tensor<8x2xui32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64Combine(%arg0, %[[RESHARD]]) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  // CHECK-NEXT: return %[[CUSTOM_CALL]] : tensor<8x2xui64>
  %0 = stablehlo.custom_call @X64Combine(%arg0, %arg1) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui32>, tensor<8x2xui32>) -> tensor<8x2xui64>
  return %0 : tensor<8x2xui64>
}

// CHECK-LABEL: func @custom_call_x64_split_high
func.func @custom_call_x64_split_high(%arg0: tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xui32>
  %0 = stablehlo.custom_call @X64SplitHigh(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>}  : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
}

// CHECK-LABEL: func @custom_call_x64_split_low
func.func @custom_call_x64_split_low(%arg0: tensor<8x2xui64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=2}>
  // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CUSTOM_CALL]] <@mesh, [{"y"}, {"x":(1)2}]> : tensor<8x2xui32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xui32>
  %0 = stablehlo.custom_call @X64SplitLow(%arg0) {backend_config = "", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x":(1)2}]>]>} : (tensor<8x2xui64>) -> tensor<8x2xui32>
  return %0 : tensor<8x2xui32>
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
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, k]) {i=8, j=4, k=4}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {"y"}]> : tensor<8x4x4xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @Eigh(%[[RESHARD1]]) {backend_config = "1,1,100,1e-6", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{}, {"y"}]> : tensor<8x4xf32>
  // CHECK-NEXT: return %[[CUSTOM_CALL]]#0, %[[RESHARD2]] : tensor<8x4x4xf32>, tensor<8x4xf32>
  %0:2 = stablehlo.custom_call @Eigh(%arg0) {backend_config = "1,1,100,1e-6", sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x4x4xf32>) -> (tensor<8x4x4xf32>, tensor<8x4xf32>)
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

// CHECK-LABEL: func @custom_call_qr_decomposition_block
func.func @custom_call_qr_decomposition_block(%arg0: tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}) -> (tensor<8x5x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>}, tensor<8x3xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"p"}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k], [i, l]) {i=8, j=5, k=3, l=3} need_replication={j, k, l}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @QrDecompositionBlock(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {}, {}]>, <@mesh_xpq, [{"x"}, {}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh_xpq, [{"x"}, {"q"}, {"p"}]> : tensor<8x5x3xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh_xpq, [{"x"}, {"p"}]> : tensor<8x3xf32>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]] : tensor<8x5x3xf32>, tensor<8x3xf32>
  %0:2 = stablehlo.custom_call @QrDecompositionBlock(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {"q"}, {"p"}]>, <@mesh_xpq, [{"x"}, {"p"}]>]>} : (tensor<8x5x3xf32>) -> (tensor<8x5x3xf32>, tensor<8x3xf32>)
  return %0#0, %0#1 : tensor<8x5x3xf32>, tensor<8x3xf32>
}

// CHECK-LABEL: func @custom_call_householder_product
func.func @custom_call_householder_product(%arg0: tensor<8x12x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>}, %arg1: tensor<8x5xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x"}, {"q"}]>}) -> (tensor<8x12x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xpq, [{"x":(1)2}, {"p"}, {"x":(2)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, l])->([i, j, k]) {i=8, j=12, k=16, l=5} need_replication={j, k, l}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xpq, [{"x"}, {}, {}]> : tensor<8x12x16xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xpq, [{"x"}, {}]> : tensor<8x5xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%[[RESHARD1]], %[[RESHARD2]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xpq, [{"x"}, {}, {}]>]>} : (tensor<8x12x16xf32>, tensor<8x5xf32>) -> tensor<8x12x16xf32>
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
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<16x8xf32>
  // CHECK: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%[[RESHARD1]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"y"}, {}]> : tensor<16x1xi32>
  // CHECK-NEXT: return %[[CUSTOM_CALL]]#0, %[[RESHARD2]] : tensor<16x1xf32>, tensor<16x1xi32>
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
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @mhlo.topk(%[[RESHARD1]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]>}
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

// CHECK-LABEL: func @custom_call_partial_reduce
func.func @custom_call_partial_reduce(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(2)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, k], [i, k]) {i=16, j=4, k=2} need_replication={k} blocked_propagation={k}>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {"x":(2)2}]>
  // CHECK-NEXT: %[[PARTIAL_REDUCE:.*]]:2 = stablehlo.custom_call @PartialReduce(%[[RESHARD1]], %arg1, %arg2, %arg3)
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[PARTIAL_REDUCE]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[PARTIAL_REDUCE]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
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

// CHECK-LABEL: func @custom_call_partial_reduce_string_backend_config
func.func @custom_call_partial_reduce_string_backend_config(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}, %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}, tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x":(1)2}]>}) {
  // NOTE: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j], [], [])->([i, j], [i, j]) {i=16, j=4} blocked_propagation={j}>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"y"}, {"x"}]>
  // CHECK-NEXT: %[[CUSTOM_CALL:.*]]:2 = stablehlo.custom_call @PartialReduce(%arg0, %[[RESHARD1]], %arg2, %arg3)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CUSTOM_CALL]]#0 <@mesh, [{"x":(1)2}, {"y"}]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CUSTOM_CALL]]#1 <@mesh, [{"y"}, {"x":(1)2}]>
  // CHECK-NEXT: return %[[RESHARD2]], %[[RESHARD3]]
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    backend_config = "{\22log2_reduction\22: 5, \22reduction_dim\22: 1, \22to_apply_type\22: \22comparator\22, \22top_k\22: 2, \22recall_target\22: 0.950000}",
    called_computations = [@top_k_gt_f32_comparator], sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>, <@mesh, [{"y"}, {"x":(1)2}]>]>} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @unregisterd_custom_call_with_existing_rule
func.func @unregisterd_custom_call_with_existing_rule(%arg0: tensor<4x2xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x4xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}){
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([j, i]) {i=4, j=2}, custom>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func @unregisterd_custom_call_without_existing_rule
func.func @unregisterd_custom_call_without_existing_rule(%arg0: tensor<4x2xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<2x4xf32>  {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"y"}]>}){
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
