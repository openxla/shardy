// RUN: rm -rf %t && mkdir -p %t && split-file %s %t
// RUN: sdy_opt %t/part1.mlir --sdy-export-pipeline="partitioner-stage=convert-global-to-local partition-count=4" --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > %t/part1_processed.mlir
// RUN: sed '1d; /^}/,$d' %t/part1_processed.mlir > %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=2, "y"=2], device_ids=[1, 0, 3, 2]>

func.func @parallel_all_slice_of_all_gather_different_mesh(
    %arg0: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>},
    %arg1: tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<5x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh_non_iota, [{}, {}]> : tensor<4x2xi32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x", "y"}, {}]>]>} : tensor<4x2xi32>
  %2 = stablehlo.compare GT, %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x", "y"}, {}]>]>} : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi1>
  %3 = stablehlo.select %2, %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x", "y"}, {}]>]>} : tensor<4x2xi1>, tensor<4x2xi32>
  %4 = sdy.reshard %3 <@mesh, [{"x", "y"}, {}]> : tensor<4x2xi32>
  %5 = sdy.reshard %4 <@mesh, [{}, {}]> : tensor<4x2xi32>

  %6 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}]>]>} %arg1 : tensor<5x2xi32>
  %7 = sdy.reshard %6 <@mesh_non_iota, [{}, {}]> : tensor<5x2xi32>
  %8 = stablehlo.negate %7 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x", "y"}, {}]>]>} : tensor<5x2xi32>
  %9 = stablehlo.compare GT, %7, %8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x", "y"}, {}]>]>} : (tensor<5x2xi32>, tensor<5x2xi32>) -> tensor<5x2xi1>
  %10 = stablehlo.select %9, %7, %8 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x", "y"}, {}]>]>} : tensor<5x2xi1>, tensor<5x2xi32>
  %11 = sdy.reshard %10 <@mesh, [{"x", "y"}, {}]> : tensor<5x2xi32>
  %12 = sdy.reshard %11 <@mesh, [{}, {}]> : tensor<5x2xi32>

  return %5, %12 : tensor<4x2xi32>, tensor<5x2xi32>
}

//--- part2.mlir
func.func @sequential_abs_4x2(%arg0: tensor<4x2xi32>) -> tensor<4x2xi32> {
  %0 = stablehlo.negate %arg0 : tensor<4x2xi32>
  %1 = stablehlo.compare GT, %arg0, %0 : (tensor<4x2xi32>, tensor<4x2xi32>) -> tensor<4x2xi1>
  %2 = stablehlo.select %1, %arg0, %0 : tensor<4x2xi1>, tensor<4x2xi32>
  return %2 : tensor<4x2xi32>
}

func.func @sequential_abs_5x2(%arg0: tensor<5x2xi32>) -> tensor<5x2xi32> {
  %0 = stablehlo.negate %arg0 : tensor<5x2xi32>
  %1 = stablehlo.compare GT, %arg0, %0 : (tensor<5x2xi32>, tensor<5x2xi32>) -> tensor<5x2xi1>
  %2 = stablehlo.select %1, %arg0, %0 : tensor<5x2xi1>, tensor<5x2xi32>
  return %2 : tensor<5x2xi32>
}

func.func @main() {
  %in4 = stablehlo.constant dense<[
    [ 10, -11],
    [-20,  21],
    [ 30, -31],
    [-40,  41]
  ]> : tensor<4x2xi32>
  %exp4 = func.call @sequential_abs_4x2(%in4) : (tensor<4x2xi32>) -> tensor<4x2xi32>

  %d0 = "stablehlo.slice"(%in4) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>
  %d1 = "stablehlo.slice"(%in4) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>
  %d2 = "stablehlo.slice"(%in4) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>
  %d3 = "stablehlo.slice"(%in4) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>

  %in5 = stablehlo.constant dense<[
    [ 10, -11],
    [-20,  21],
    [ 30, -31],
    [-40,  41],
    [ 50, -51]
  ]> : tensor<5x2xi32>
  %exp5 = func.call @sequential_abs_5x2(%in5) : (tensor<5x2xi32>) -> tensor<5x2xi32>

  %res:8 = "interpreter.run_parallel"(%d0, %in5, %d1, %in5, %d2, %in5, %d3, %in5) {
    programs = [[
      @parallel_all_slice_of_all_gather_different_mesh,
      @parallel_all_slice_of_all_gather_different_mesh,
      @parallel_all_slice_of_all_gather_different_mesh,
      @parallel_all_slice_of_all_gather_different_mesh
    ]]
  } : (tensor<1x2xi32>, tensor<5x2xi32>,
       tensor<1x2xi32>, tensor<5x2xi32>,
       tensor<1x2xi32>, tensor<5x2xi32>,
       tensor<1x2xi32>, tensor<5x2xi32>) ->
      (tensor<4x2xi32>, tensor<5x2xi32>,
       tensor<4x2xi32>, tensor<5x2xi32>,
       tensor<4x2xi32>, tensor<5x2xi32>,
       tensor<4x2xi32>, tensor<5x2xi32>)

  "check.expect_eq"(%res#0, %exp4) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#1, %exp5) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
  "check.expect_eq"(%res#2, %exp4) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#3, %exp5) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
  "check.expect_eq"(%res#4, %exp4) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#5, %exp5) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
  "check.expect_eq"(%res#6, %exp4) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#7, %exp5) : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
  func.return
}
