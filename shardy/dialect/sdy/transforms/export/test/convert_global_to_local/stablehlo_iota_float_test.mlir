// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>
sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @iota_bf16_exceeds_significand
// CHECK-SAME:   -> (tensor<257xbf16> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>})
func.func @iota_bf16_exceeds_significand() -> (tensor<1028xbf16> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  // CHECK-NEXT: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<257xi32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 257, 514, 771]> : tensor<4xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET_CONV:.*]] = stablehlo.convert %[[OFFSET]] : tensor<i32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_CONV]], dims = [] : (tensor<i32>) -> tensor<257xi32>
  // CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<257xi32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.convert %[[SUM]] : (tensor<257xi32>) -> tensor<257xbf16>
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<1028xbf16>
  // CHECK-NEXT: return %[[RES]] : tensor<257xbf16>
  return %0 : tensor<1028xbf16>
}

// CHECK-LABEL: func @iota_bf16_within_significand
// CHECK-SAME:   -> (tensor<64xbf16> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>})
func.func @iota_bf16_within_significand() -> (tensor<256xbf16> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  // CHECK-NEXT: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<64xbf16>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 64, 128, 192]> : tensor<4xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET_CONV:.*]] = stablehlo.convert %[[OFFSET]] : (tensor<i32>) -> tensor<bf16>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_CONV]], dims = [] : (tensor<bf16>) -> tensor<64xbf16>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<64xbf16>
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<256xbf16>
  // CHECK-NEXT: return %[[RES]] : tensor<64xbf16>
  return %0 : tensor<256xbf16>
}

// CHECK-LABEL: func @iota_c64_exceeds_f32_significand
// CHECK-SAME:   -> (tensor<8388609xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>})
func.func @iota_c64_exceeds_f32_significand() -> (tensor<33554436xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  // CHECK-NEXT: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<8388609xi32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 8388609, 16777218, 25165827]> : tensor<4xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET_CONV:.*]] = stablehlo.convert %[[OFFSET]] : tensor<i32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_CONV]], dims = [] : (tensor<i32>) -> tensor<8388609xi32>
  // CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<8388609xi32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.convert %[[SUM]] : (tensor<8388609xi32>) -> tensor<8388609xcomplex<f32>>
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<33554436xcomplex<f32>>
  // CHECK-NEXT: return %[[RES]] : tensor<8388609xcomplex<f32>>
  return %0 : tensor<33554436xcomplex<f32>>
}

// CHECK-LABEL: func @iota_f32_exceeds_i32_uses_i64
// CHECK-SAME:   -> (tensor<1610612736xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}]>})
func.func @iota_f32_exceeds_i32_uses_i64() -> (tensor<3221225472xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}]>}) {
  // CHECK-NEXT: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<1610612736xi64>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 1610612736, 0, 1610612736, 0, 1610612736, 0, 1610612736]> : tensor<8xi64>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<8xi64>, tensor<ui32>) -> tensor<1xi64>
  // CHECK-NEXT: %[[OFFSET:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: %[[OFFSET_CONV:.*]] = stablehlo.convert %[[OFFSET]] : tensor<i64>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_CONV]], dims = [] : (tensor<i64>) -> tensor<1610612736xi64>
  // CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<1610612736xi64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.convert %[[SUM]] : (tensor<1610612736xi64>) -> tensor<1610612736xf32>
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}]>]>} : tensor<3221225472xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<1610612736xf32>
  return %0 : tensor<3221225472xf32>
}

// CHECK-LABEL: func @iota_f8e8m0fnu_lacks_zero
// CHECK-SAME:   -> (tensor<2x1xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>})
func.func @iota_f8e8m0fnu_lacks_zero() -> (tensor<2x4xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
  // CHECK-NEXT: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 1 : tensor<2x1xi32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[OFFSET:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET_CONV:.*]] = stablehlo.convert %[[OFFSET]] : tensor<i32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_CONV]], dims = [] : (tensor<i32>) -> tensor<2x1xi32>
  // CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<2x1xi32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.convert %[[SUM]] : (tensor<2x1xi32>) -> tensor<2x1xf8E8M0FNU>
  %0 = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : tensor<2x4xf8E8M0FNU>
  // CHECK-NEXT: return %[[RES]] : tensor<2x1xf8E8M0FNU>
  return %0 : tensor<2x4xf8E8M0FNU>
}

// CHECK-LABEL: func @iota_f64_large_dim_within_significand
// CHECK-SAME:   -> (tensor<1073741824xf64> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>})
func.func @iota_f64_large_dim_within_significand() -> (tensor<4294967296xf64> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}]>}) {
  // CHECK-NEXT: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<1073741824xf64>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[0, 1073741824, 2147483648, 3221225472]> : tensor<4xi64>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<4xi64>, tensor<ui32>) -> tensor<1xi64>
  // CHECK-NEXT: %[[OFFSET_I64:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: %[[OFFSET_CONV:.*]] = stablehlo.convert %[[OFFSET_I64]] : (tensor<i64>) -> tensor<f64>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_CONV]], dims = [] : (tensor<f64>) -> tensor<1073741824xf64>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<1073741824xf64>
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{"x"}]>]>} : tensor<4294967296xf64>
  // CHECK-NEXT: return %[[RES]] : tensor<1073741824xf64>
  return %0 : tensor<4294967296xf64>
}
