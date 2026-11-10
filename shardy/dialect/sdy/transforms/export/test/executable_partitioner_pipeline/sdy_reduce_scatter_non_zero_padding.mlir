// RUN: rm -rf %t && mkdir -p %t && split-file %s %t
// RUN: sdy_opt %t/part1.mlir --sdy-pad-for-divisibility --sdy-resolve-single-device-sharding="partition-count=8" --sdy-convert-global-to-local="partition-count=8" --sdy-inline-meshes --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > %t/part1_processed.mlir
// RUN: sed '1d; /^}/,$d' %t/part1_processed.mlir > %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=4]>

// %in has shape 2x5 sharded across "y"=4 (padded to 2x8 with 0). %pad1 pads
// dim 1 from 5 to 6 with 1, masking columns [5..7] to 1 and registering kOne
// in PaddingCache. %rs reduce-scatters across "x"=2, summing columns [5..7]
// from 1 to 2. PaddingCache must not retain kOne on %rs; otherwise %pad2
// (which pads dim 1 from 6 to 7 with 1) skips masking column 6 back to 1.
func.func @parallel_reduce_scatter_non_zero_padding(
    %arg0: tensor<2x5xi32>)
    -> (tensor<2x7xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %in = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh, [{}, {"y"}]> : tensor<2x5xi32>
  %pad1 = stablehlo.pad %in, %c1, low = [0, 0], high = [0, 1], interior = [0, 0]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
      : (tensor<2x5xi32>, tensor<i32>) -> tensor<2x6xi32>
  %rs = sdy.reduce_scatter [{"x"}, {}] %pad1 out_sharding=<@mesh, [{"x"}, {"y"}]> : tensor<2x6xi32>
  %pad2 = stablehlo.pad %rs, %c1, low = [0, 0], high = [0, 1], interior = [0, 0]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
      : (tensor<2x6xi32>, tensor<i32>) -> tensor<2x7xi32>
  %ag = sdy.all_gather [{"x"}, {"y"}] %pad2 out_sharding=<@mesh, [{}, {}]> : tensor<2x7xi32>
  return %ag : tensor<2x7xi32>
}

//--- part2.mlir
func.func @sequential_reduce_scatter_non_zero_padding(
    %arg0: tensor<2x5xi32>) -> tensor<2x7xi32> {
  %c1 = stablehlo.constant dense<1> : tensor<i32>
  %pad1 = stablehlo.pad %arg0, %c1, low = [0, 0], high = [0, 1], interior = [0, 0]
      : (tensor<2x5xi32>, tensor<i32>) -> tensor<2x6xi32>
  %sum_x = stablehlo.add %pad1, %pad1 : tensor<2x6xi32>
  %pad2 = stablehlo.pad %sum_x, %c1, low = [0, 0], high = [0, 1], interior = [0, 0]
      : (tensor<2x6xi32>, tensor<i32>) -> tensor<2x7xi32>
  return %pad2 : tensor<2x7xi32>
}

func.func @main() {
  %in = stablehlo.constant dense<[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
  ]> : tensor<2x5xi32>

  %expected = func.call @sequential_reduce_scatter_non_zero_padding(%in)
      : (tensor<2x5xi32>) -> tensor<2x7xi32>

  %res:8 = "interpreter.run_parallel"(
      %in, %in, %in, %in, %in, %in, %in, %in) {
    programs = [[
      @parallel_reduce_scatter_non_zero_padding,
      @parallel_reduce_scatter_non_zero_padding,
      @parallel_reduce_scatter_non_zero_padding,
      @parallel_reduce_scatter_non_zero_padding,
      @parallel_reduce_scatter_non_zero_padding,
      @parallel_reduce_scatter_non_zero_padding,
      @parallel_reduce_scatter_non_zero_padding,
      @parallel_reduce_scatter_non_zero_padding
    ]]
  } : (tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x5xi32>,
       tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x5xi32>) ->
      (tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>,
       tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>, tensor<2x7xi32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#4, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#5, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#6, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  "check.expect_eq"(%res#7, %expected) : (tensor<2x7xi32>, tensor<2x7xi32>) -> ()
  func.return
}
