// RUN: rm -rf %t && mkdir -p %t && split-file %s %t
// RUN: sdy_opt %t/part1.mlir --sdy-populate-op-sharding-rules --sdy-insert-explicit-reshards="enable-full-version=true mark-partial-result-with-unreduced-axes=true" --sdy-resolve-permutation-factors="enable-halo-exchange=true partition-count=4" --sdy-reshard-to-collectives --sdy-optimize-collectives --canonicalize --sdy-pad-for-divisibility --sdy-resolve-single-device-sharding="partition-count=4" --sdy-convert-global-to-local="partition-count=4" --sdy-drop-sharding-rules --sdy-inline-meshes --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > %t/part1_processed.mlir
// RUN: sed '1d; /^}/,$d' %t/part1_processed.mlir > %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @parallel_reduce_scatter_uneven_all_slice(
    %arg0: tensor<5x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<5x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>}
      : (tensor<5x4xf32>, tensor<4x4xf32>) -> tensor<5x4xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>}
      : tensor<5x4xf32>
  return %1 : tensor<5x4xf32>
}

//--- part2.mlir
func.func @sequential_reduce_scatter_uneven_all_slice(
    %arg0: tensor<5x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<5x4xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
      : (tensor<5x4xf32>, tensor<4x4xf32>) -> tensor<5x4xf32>
  %1 = stablehlo.add %0, %0 : tensor<5x4xf32>
  return %1 : tensor<5x4xf32>
}

func.func @main() {
  %lhs = stablehlo.constant dense<[
    [ 1.0,  2.0,  3.0,  4.0],
    [ 5.0,  6.0,  7.0,  8.0],
    [ 9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
    [17.0, 18.0, 19.0, 20.0]
  ]> : tensor<5x4xf32>
  %rhs = stablehlo.constant dense<[
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]> : tensor<4x4xf32>

  %expected = func.call @sequential_reduce_scatter_uneven_all_slice(%lhs, %rhs)
      : (tensor<5x4xf32>, tensor<4x4xf32>) -> tensor<5x4xf32>

  // Device 0 (x=0, y=0) & Device 1 (x=0, y=1): lhs cols [0:2], rhs rows [0:2]
  // Device 2 (x=1, y=0) & Device 3 (x=1, y=1): lhs cols [2:4], rhs rows [2:4]
  %lhs_x0 = "stablehlo.slice"(%lhs) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 5, 2>, strides = array<i64: 1, 1>} : (tensor<5x4xf32>) -> tensor<5x2xf32>
  %lhs_x1 = "stablehlo.slice"(%lhs) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 5, 4>, strides = array<i64: 1, 1>} : (tensor<5x4xf32>) -> tensor<5x2xf32>
  %rhs_x0 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x4xf32>
  %rhs_x1 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xf32>) -> tensor<2x4xf32>

  %res:4 = "interpreter.run_parallel"(
      %lhs_x0, %rhs_x0,
      %lhs_x0, %rhs_x0,
      %lhs_x1, %rhs_x1,
      %lhs_x1, %rhs_x1) {
    programs = [[
      @parallel_reduce_scatter_uneven_all_slice,
      @parallel_reduce_scatter_uneven_all_slice,
      @parallel_reduce_scatter_uneven_all_slice,
      @parallel_reduce_scatter_uneven_all_slice
    ]]
  } : (tensor<5x2xf32>, tensor<2x4xf32>,
       tensor<5x2xf32>, tensor<2x4xf32>,
       tensor<5x2xf32>, tensor<2x4xf32>,
       tensor<5x2xf32>, tensor<2x4xf32>) ->
      (tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>, tensor<5x4xf32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<5x4xf32>, tensor<5x4xf32>) -> ()
  func.return
}
