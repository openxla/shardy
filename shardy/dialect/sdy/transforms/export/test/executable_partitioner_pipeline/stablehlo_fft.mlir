// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// Test 1: FFT with sharded FFT dimension on 2D mesh. Expects explicit reshard
// before FFT and reshard back after FFT.
func.func @parallel_fft_sharded_fft_dim(
    %arg0: tensor<4x6xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>})
    -> (tensor<4x6xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}) {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type FFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>
  } : (tensor<4x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
  return %0 : tensor<4x6xcomplex<f32>>
}

func.func @sequential_fft_sharded_fft_dim(
    %arg0: tensor<4x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type FFT>
  } : (tensor<4x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
  return %0 : tensor<4x6xcomplex<f32>>
}

// Test 2: IFFT batch-sharded on 1D mesh (no communication required).
func.func @parallel_ifft_batch_sharded(
    %arg0: tensor<2x4xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<2x4xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 4>,
    fft_type = #stablehlo<fft_type IFFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xcomplex<f32>>
  return %0 : tensor<2x4xcomplex<f32>>
}

func.func @sequential_ifft_batch_sharded(
    %arg0: tensor<2x4xcomplex<f32>>) -> tensor<2x4xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 4>,
    fft_type = #stablehlo<fft_type IFFT>
  } : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xcomplex<f32>>
  return %0 : tensor<2x4xcomplex<f32>>
}

// Test 3: RFFT with sharded FFT dimension on 2D mesh. Input is f32 (4x6),
// output is complex<f32> (4x4).
func.func @parallel_rfft_sharded_fft_dim(
    %arg0: tensor<4x6xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>})
    -> (tensor<4x4xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type RFFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {}]>]>
  } : (tensor<4x6xf32>) -> tensor<4x4xcomplex<f32>>
  return %0 : tensor<4x4xcomplex<f32>>
}

func.func @sequential_rfft_sharded_fft_dim(
    %arg0: tensor<4x6xf32>) -> tensor<4x4xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type RFFT>
  } : (tensor<4x6xf32>) -> tensor<4x4xcomplex<f32>>
  return %0 : tensor<4x4xcomplex<f32>>
}

// Test 4: IRFFT with even fft_length and sharded FFT dimension on 2D mesh.
// Input is complex<f32> (4x4), output is f32 (4x6).
func.func @parallel_irfft_sharded_fft_dim(
    %arg0: tensor<4x4xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>})
    -> (tensor<4x6xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type IRFFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {}]>]>
  } : (tensor<4x4xcomplex<f32>>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}

func.func @sequential_irfft_sharded_fft_dim(
    %arg0: tensor<4x4xcomplex<f32>>) -> tensor<4x6xf32> {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type IRFFT>
  } : (tensor<4x4xcomplex<f32>>) -> tensor<4x6xf32>
  return %0 : tensor<4x6xf32>
}

// Test 5a: Indivisible batch padding on complex FFT (tests ComplexType zero
// padding in pad-for-divisibility via reshard from replicated 3x4 to sharded).
func.func @parallel_fft_indivisible_batch(
    %arg0: tensor<3x4xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>})
    -> (tensor<3x4xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh_2, [{"x"}, {}]> : tensor<3x4xcomplex<f32>>
  %1 = "stablehlo.fft"(%0) {
    fft_length = array<i64: 4>,
    fft_type = #stablehlo<fft_type FFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
  return %1 : tensor<3x4xcomplex<f32>>
}

func.func @sequential_fft_indivisible_batch(
    %arg0: tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 4>,
    fft_type = #stablehlo<fft_type FFT>
  } : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>
  return %0 : tensor<3x4xcomplex<f32>>
}

// Test 5b: Indivisible batch padding on RFFT (tests inferReturnTypes shape
// reconciliation in padGenericOp when input and output element types differ).
func.func @parallel_rfft_indivisible_batch(
    %arg0: tensor<4x6xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
    -> (tensor<3x4xcomplex<f32>> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:3, 0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<4x6xf32>) -> tensor<3x6xf32>
  %1 = "stablehlo.fft"(%0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type RFFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<3x6xf32>) -> tensor<3x4xcomplex<f32>>
  return %1 : tensor<3x4xcomplex<f32>>
}

func.func @sequential_rfft_indivisible_batch(
    %arg0: tensor<4x6xf32>) -> tensor<3x4xcomplex<f32>> {
  %0 = stablehlo.slice %arg0 [0:3, 0:6] : (tensor<4x6xf32>) -> tensor<3x6xf32>
  %1 = "stablehlo.fft"(%0) {
    fft_length = array<i64: 6>,
    fft_type = #stablehlo<fft_type RFFT>
  } : (tensor<3x6xf32>) -> tensor<3x4xcomplex<f32>>
  return %1 : tensor<3x4xcomplex<f32>>
}

//--- part2.mlir

func.func @main() {
  // Test 1: FFT with sharded FFT dim on 2D mesh.
  %in_fft = stablehlo.constant dense<[
    [(1.0, 0.5), (2.0, -1.0), (3.0, 0.25), (4.0, -0.5), (5.0, 1.5), (6.0, -2.0)],
    [(7.0, -0.25), (8.0, 1.0), (9.0, -1.5), (10.0, 0.5), (11.0, -0.75), (12.0, 2.5)],
    [(-1.0, 2.0), (-2.0, 0.5), (-3.0, -1.0), (-4.0, 1.25), (-5.0, -0.5), (-6.0, 3.0)],
    [(0.5, -1.5), (1.5, 2.0), (-2.5, 0.75), (3.5, -2.25), (-4.5, 1.0), (5.5, -0.25)]
  ]> : tensor<4x6xcomplex<f32>>

  %seq_fft = func.call @sequential_fft_sharded_fft_dim(%in_fft)
    : (tensor<4x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>

  %f00 = stablehlo.slice %in_fft [0:2, 0:3] : (tensor<4x6xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
  %f01 = stablehlo.slice %in_fft [0:2, 3:6] : (tensor<4x6xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
  %f10 = stablehlo.slice %in_fft [2:4, 0:3] : (tensor<4x6xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
  %f11 = stablehlo.slice %in_fft [2:4, 3:6] : (tensor<4x6xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>

  %par_fft:4 = "interpreter.run_parallel"(%f00, %f01, %f10, %f11) {
    programs = [[@parallel_fft_sharded_fft_dim, @parallel_fft_sharded_fft_dim,
                 @parallel_fft_sharded_fft_dim, @parallel_fft_sharded_fft_dim]]
  } : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>,
       tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    -> (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>,
        tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
  %fft_r0 = "stablehlo.concatenate"(%par_fft#0, %par_fft#1) {dimension = 1 : i64}
    : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x6xcomplex<f32>>
  %fft_r1 = "stablehlo.concatenate"(%par_fft#2, %par_fft#3) {dimension = 1 : i64}
    : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<2x6xcomplex<f32>>
  %out_fft = "stablehlo.concatenate"(%fft_r0, %fft_r1) {dimension = 0 : i64}
    : (tensor<2x6xcomplex<f32>>, tensor<2x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
  "check.expect_eq"(%seq_fft, %out_fft) : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()

  // Test 2: IFFT batch-sharded on 1D mesh.
  %in_ifft = stablehlo.constant dense<[
    [(4.0, 0.0), (-2.0, 2.0), (0.0, -1.0), (-2.0, -2.0)],
    [(10.0, 2.0), (-1.0, 3.0), (2.0, 0.0), (3.0, -5.0)]
  ]> : tensor<2x4xcomplex<f32>>

  %seq_ifft = func.call @sequential_ifft_batch_sharded(%in_ifft)
    : (tensor<2x4xcomplex<f32>>) -> tensor<2x4xcomplex<f32>>

  %i0 = stablehlo.slice %in_ifft [0:1, 0:4] : (tensor<2x4xcomplex<f32>>) -> tensor<1x4xcomplex<f32>>
  %i1 = stablehlo.slice %in_ifft [1:2, 0:4] : (tensor<2x4xcomplex<f32>>) -> tensor<1x4xcomplex<f32>>

  %par_ifft:2 = "interpreter.run_parallel"(%i0, %i1) {
    programs = [[@parallel_ifft_batch_sharded, @parallel_ifft_batch_sharded]]
  } : (tensor<1x4xcomplex<f32>>, tensor<1x4xcomplex<f32>>)
    -> (tensor<1x4xcomplex<f32>>, tensor<1x4xcomplex<f32>>)
  %out_ifft = "stablehlo.concatenate"(%par_ifft#0, %par_ifft#1) {dimension = 0 : i64}
    : (tensor<1x4xcomplex<f32>>, tensor<1x4xcomplex<f32>>) -> tensor<2x4xcomplex<f32>>
  "check.expect_eq"(%seq_ifft, %out_ifft) : (tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>) -> ()

  // Test 3: RFFT with sharded FFT dim on 2D mesh.
  %in_rfft = stablehlo.constant dense<[
    [1.0, -2.5, 3.25, -4.5, 5.75, -6.0],
    [0.5, 1.5, -2.25, 3.75, -4.0, 5.5],
    [-1.25, 2.75, -3.5, 4.25, -5.5, 6.25],
    [7.5, -8.25, 9.0, -10.5, 11.25, -12.0]
  ]> : tensor<4x6xf32>

  %seq_rfft = func.call @sequential_rfft_sharded_fft_dim(%in_rfft)
    : (tensor<4x6xf32>) -> tensor<4x4xcomplex<f32>>

  %r00 = stablehlo.slice %in_rfft [0:2, 0:3] : (tensor<4x6xf32>) -> tensor<2x3xf32>
  %r01 = stablehlo.slice %in_rfft [0:2, 3:6] : (tensor<4x6xf32>) -> tensor<2x3xf32>
  %r10 = stablehlo.slice %in_rfft [2:4, 0:3] : (tensor<4x6xf32>) -> tensor<2x3xf32>
  %r11 = stablehlo.slice %in_rfft [2:4, 3:6] : (tensor<4x6xf32>) -> tensor<2x3xf32>

  %par_rfft:4 = "interpreter.run_parallel"(%r00, %r01, %r10, %r11) {
    programs = [[@parallel_rfft_sharded_fft_dim, @parallel_rfft_sharded_fft_dim,
                 @parallel_rfft_sharded_fft_dim, @parallel_rfft_sharded_fft_dim]]
  } : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
    -> (tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>,
        tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>)
  %out_rfft_a = "stablehlo.concatenate"(%par_rfft#0, %par_rfft#2) {dimension = 0 : i64}
    : (tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
  %out_rfft_b = "stablehlo.concatenate"(%par_rfft#1, %par_rfft#3) {dimension = 0 : i64}
    : (tensor<2x4xcomplex<f32>>, tensor<2x4xcomplex<f32>>) -> tensor<4x4xcomplex<f32>>
  "check.expect_eq"(%seq_rfft, %out_rfft_a) : (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) -> ()
  "check.expect_eq"(%seq_rfft, %out_rfft_b) : (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) -> ()

  // Test 4: IRFFT with even fft_length (6) and sharded FFT dim on 2D mesh.
  %in_irfft = stablehlo.constant dense<[
    [(10.0, 0.0), (-2.0, 1.5), (0.5, -0.5), (-1.0, 0.0)],
    [(6.0, 0.0), (1.0, -2.0), (-1.5, 0.75), (2.0, 0.0)],
    [(-4.0, 0.0), (3.0, 0.5), (1.25, -1.0), (0.5, 0.0)],
    [(8.0, 0.0), (-1.5, -1.25), (2.0, 0.25), (-3.0, 0.0)]
  ]> : tensor<4x4xcomplex<f32>>

  %seq_irfft = func.call @sequential_irfft_sharded_fft_dim(%in_irfft)
    : (tensor<4x4xcomplex<f32>>) -> tensor<4x6xf32>

  %ir00 = stablehlo.slice %in_irfft [0:2, 0:2] : (tensor<4x4xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  %ir01 = stablehlo.slice %in_irfft [0:2, 2:4] : (tensor<4x4xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  %ir10 = stablehlo.slice %in_irfft [2:4, 0:2] : (tensor<4x4xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  %ir11 = stablehlo.slice %in_irfft [2:4, 2:4] : (tensor<4x4xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>

  %par_irfft:4 = "interpreter.run_parallel"(%ir00, %ir01, %ir10, %ir11) {
    programs = [[@parallel_irfft_sharded_fft_dim, @parallel_irfft_sharded_fft_dim,
                 @parallel_irfft_sharded_fft_dim, @parallel_irfft_sharded_fft_dim]]
  } : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>,
       tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
    -> (tensor<2x6xf32>, tensor<2x6xf32>, tensor<2x6xf32>, tensor<2x6xf32>)
  %out_irfft_a = "stablehlo.concatenate"(%par_irfft#0, %par_irfft#2) {dimension = 0 : i64}
    : (tensor<2x6xf32>, tensor<2x6xf32>) -> tensor<4x6xf32>
  %out_irfft_b = "stablehlo.concatenate"(%par_irfft#1, %par_irfft#3) {dimension = 0 : i64}
    : (tensor<2x6xf32>, tensor<2x6xf32>) -> tensor<4x6xf32>
  "check.expect_eq"(%seq_irfft, %out_irfft_a) : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
  "check.expect_eq"(%seq_irfft, %out_irfft_b) : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()

  // Test 5a: Indivisible batch padding on complex FFT.
  %in_fft_indiv = stablehlo.constant dense<[
    [(1.5, -0.5), (2.5, 1.0), (-3.5, 0.25), (4.5, -1.25)],
    [(-0.5, 2.0), (1.0, -1.5), (2.0, 0.75), (-3.0, 1.5)],
    [(3.25, 0.5), (-1.75, -0.25), (0.25, 2.25), (-2.25, -0.75)]
  ]> : tensor<3x4xcomplex<f32>>

  %seq_fft_indiv = func.call @sequential_fft_indivisible_batch(%in_fft_indiv)
    : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xcomplex<f32>>

  %par_fft_indiv:2 = "interpreter.run_parallel"(%in_fft_indiv, %in_fft_indiv) {
    programs = [[@parallel_fft_indivisible_batch, @parallel_fft_indivisible_batch]]
  } : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>)
    -> (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>)
  "check.expect_eq"(%seq_fft_indiv, %par_fft_indiv#0) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
  "check.expect_eq"(%seq_fft_indiv, %par_fft_indiv#1) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()

  // Test 5b: Indivisible batch padding on RFFT.
  %in_rfft_indiv = stablehlo.constant dense<[
    [2.0, -1.5, 0.75, 3.25, -4.0, 1.25],
    [-3.5, 4.25, -1.0, 2.5, 0.5, -2.75],
    [1.75, 0.25, -2.5, -3.0, 5.25, 1.5],
    [-0.25, 3.75, 4.5, -1.25, -2.0, 6.0]
  ]> : tensor<4x6xf32>

  %seq_rfft_indiv = func.call @sequential_rfft_indivisible_batch(%in_rfft_indiv)
    : (tensor<4x6xf32>) -> tensor<3x4xcomplex<f32>>

  %ri0 = stablehlo.slice %in_rfft_indiv [0:2, 0:6] : (tensor<4x6xf32>) -> tensor<2x6xf32>
  %ri1 = stablehlo.slice %in_rfft_indiv [2:4, 0:6] : (tensor<4x6xf32>) -> tensor<2x6xf32>

  %par_rfft_indiv:2 = "interpreter.run_parallel"(%ri0, %ri1) {
    programs = [[@parallel_rfft_indivisible_batch, @parallel_rfft_indivisible_batch]]
  } : (tensor<2x6xf32>, tensor<2x6xf32>)
    -> (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>)
  "check.expect_eq"(%seq_rfft_indiv, %par_rfft_indiv#0) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
  "check.expect_eq"(%seq_rfft_indiv, %par_rfft_indiv#1) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()

  return
}
