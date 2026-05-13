// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>

// This function computes the FFT of a 2x4 complex tensor.
// It is sharded along dimension 0 (batch), so each device performs a 4-point
// FFT.
func.func @parallel_fft(
  %arg0: tensor<2x4xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
  -> (tensor<2x4xcomplex<f64>> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = "stablehlo.fft"(%arg0) {
    fft_length = array<i64: 4>,
    fft_type = #stablehlo<fft_type FFT>,
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>
  } : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>
  return %0 : tensor<2x4xcomplex<f64>>
}

//--- part2.mlir

func.func @main() {
  // Input tensor with 8 unique complex values.
  %input = stablehlo.constant dense<[
    [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
    [(5.0, 0.0), (7.0, 0.0), (11.0, 0.0), (13.0, 0.0)]
  ]> : tensor<2x4xcomplex<f64>>

  %seq = func.call @sequential_fft(%input) : (tensor<2x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>

  %s0 = "stablehlo.slice"(%input) {start_indices=array<i64: 0, 0>, limit_indices=array<i64: 1, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xcomplex<f64>>) -> tensor<1x4xcomplex<f64>>
  %s1 = "stablehlo.slice"(%input) {start_indices=array<i64: 1, 0>, limit_indices=array<i64: 2, 4>, strides=array<i64: 1, 1>} : (tensor<2x4xcomplex<f64>>) -> tensor<1x4xcomplex<f64>>
  %pars:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@parallel_fft, @parallel_fft]]
  } : (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>) -> (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>)
  %par = "stablehlo.concatenate"(%pars#0, %pars#1) {dimension = 0 : i64}
    : (tensor<1x4xcomplex<f64>>, tensor<1x4xcomplex<f64>>) -> tensor<2x4xcomplex<f64>>

  "check.expect_eq"(%seq, %par) : (tensor<2x4xcomplex<f64>>, tensor<2x4xcomplex<f64>>) -> ()

  return
}
