// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh_xyzp = <["x"=4, "y"=2, "z"=4, "p"=3]>

// CHECK-LABEL: func @fft
func.func @fft(%arg0: tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y", "z"}, {}, {}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y", "z"}, {}, {}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  return %0 : tensor<128x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_inverse
func.func @fft_inverse(%arg0: tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) ->(tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y", "z"}, {}, {}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  IFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y", "z"}, {}, {}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = IFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  return %0 : tensor<128x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_real_truncated_result
func.func @fft_real_truncated_result(%arg0: tensor<128x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x33xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y", "z"}, {}, {}]> : tensor<128x32x64xf32>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  RFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y", "z"}, {}, {}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"p"}]> : tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x33xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = RFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  return %0 : tensor<128x32x33xcomplex<f32>>
}

// CHECK-LABEL: func @fft_inverse_real_expanded_result
func.func @fft_inverse_real_expanded_result(%arg0: tensor<128x32x33xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>}) -> (tensor<128x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y", "z"}, {}, {}]> : tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  IRFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y", "z"}, {}, {}]>]>} : (tensor<128x32x33xcomplex<f32>>) -> tensor<128x32x64xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xf32>
  %0  = stablehlo.fft %arg0, type = IRFFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x33xcomplex<f32>>) -> tensor<128x32x64xf32>
  return %0 : tensor<128x32x64xf32>
}

// CHECK-LABEL: func @fft_small_batch_dimension
// TODO(enver): Subaxes of "z" should be distributed to the batching dimension.
func.func @fft_small_batch_dimension(%arg0: tensor<16x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<16x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "y"}, {}, {}]> : tensor<16x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "y"}, {}, {}]>]>} : (tensor<16x32x64xcomplex<f32>>) -> tensor<16x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<16x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = FFT, length = [32, 64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<16x32x64xcomplex<f32>>) -> tensor<16x32x64xcomplex<f32>>
  return %0 : tensor<16x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_single_fft_dimension
func.func @fft_single_fft_dimension(%arg0: tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x64xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "z"}, {"y"}, {}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  FFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "z"}, {"y"}, {}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"z"}]> : tensor<128x32x64xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x64xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = FFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>]>} : (tensor<128x32x64xcomplex<f32>>) -> tensor<128x32x64xcomplex<f32>>
  return %0 : tensor<128x32x64xcomplex<f32>>
}

// CHECK-LABEL: func @fft_single_fft_dimension_real_truncated_result
func.func @fft_single_fft_dimension_real_truncated_result(%arg0: tensor<128x32x64xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"z"}]>}) -> (tensor<128x32x33xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"x", "z"}, {"y"}, {}]> : tensor<128x32x64xf32>
  // CHECK-NEXT: %[[FFT:.*]] = stablehlo.fft %[[RESHARD1]], type =  RFFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x", "z"}, {"y"}, {}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[FFT]] <@mesh_xyzp, [{"x"}, {"y"}, {"p"}]> : tensor<128x32x33xcomplex<f32>>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x32x33xcomplex<f32>>
  %0  = stablehlo.fft %arg0, type = RFFT, length = [64] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {"y"}, {"p"}]>]>} : (tensor<128x32x64xf32>) -> tensor<128x32x33xcomplex<f32>>
  return %0 : tensor<128x32x33xcomplex<f32>>
}
