// RUN: rm -rf %t && mkdir -p %t && split-file %s %t
// RUN: sdy_opt %t/part1.mlir --sdy-export-pipeline="partitioner-stage=convert-global-to-local partition-count=4" --sdy-drop-sharding-and-mesh --allow-unregistered-dialect > %t/part1_processed.mlir
// RUN: sed '1d; /^}/,$d' %t/part1_processed.mlir > %t/combined.mlir
// RUN: cat %t/part2.mlir >> %t/combined.mlir
// RUN: stablehlo-translate --interpret %t/combined.mlir

//--- part1.mlir
sdy.mesh @mesh = <["x"=2, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=2, "y"=2], device_ids=[1, 0, 3, 2]>

func.func @parallel_reduce_scatter_fusion_different_mesh(
    %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>},
    %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>})
    -> (tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x", "y"}>]>} : (tensor<4x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<4x2xi32>
  %2 = sdy.reshard %1 <@mesh_non_iota, [{"x", "y"}, {}]> : tensor<4x2xi32>
  %3 = stablehlo.negate %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x", "y"}, {}]>]>} : tensor<4x2xi32>
  %4 = sdy.reshard %3 <@mesh, [{}, {}]> : tensor<4x2xi32>
  return %4 : tensor<4x2xi32>
}

//--- part2.mlir
func.func @sequential_reduce_scatter_fusion_different_mesh(
    %arg0: tensor<4x4xi32>, %arg1: tensor<4x2xi32>) -> tensor<4x2xi32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<4x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>
  %1 = stablehlo.negate %0 : tensor<4x2xi32>
  return %1 : tensor<4x2xi32>
}

func.func @main() {
  %lhs = stablehlo.constant dense<[
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16]
  ]> : tensor<4x4xi32>
  %rhs = stablehlo.constant dense<[
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 0]
  ]> : tensor<4x2xi32>
  %expected = func.call @sequential_reduce_scatter_fusion_different_mesh(%lhs, %rhs)
      : (tensor<4x4xi32>, tensor<4x2xi32>) -> tensor<4x2xi32>

  %l0 = "stablehlo.slice"(%lhs) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 4, 1>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x1xi32>
  %l1 = "stablehlo.slice"(%lhs) {start_indices = array<i64: 0, 1>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x1xi32>
  %l2 = "stablehlo.slice"(%lhs) {start_indices = array<i64: 0, 2>, limit_indices = array<i64: 4, 3>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x1xi32>
  %l3 = "stablehlo.slice"(%lhs) {start_indices = array<i64: 0, 3>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>} : (tensor<4x4xi32>) -> tensor<4x1xi32>
  %r0 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 0, 0>, limit_indices = array<i64: 1, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>
  %r1 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 1, 0>, limit_indices = array<i64: 2, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>
  %r2 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 2, 0>, limit_indices = array<i64: 3, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>
  %r3 = "stablehlo.slice"(%rhs) {start_indices = array<i64: 3, 0>, limit_indices = array<i64: 4, 2>, strides = array<i64: 1, 1>} : (tensor<4x2xi32>) -> tensor<1x2xi32>

  %res:4 = "interpreter.run_parallel"(%l0, %r0, %l1, %r1, %l2, %r2, %l3, %r3) {
    programs = [[
      @parallel_reduce_scatter_fusion_different_mesh,
      @parallel_reduce_scatter_fusion_different_mesh,
      @parallel_reduce_scatter_fusion_different_mesh,
      @parallel_reduce_scatter_fusion_different_mesh
    ]]
  } : (tensor<4x1xi32>, tensor<1x2xi32>,
       tensor<4x1xi32>, tensor<1x2xi32>,
       tensor<4x1xi32>, tensor<1x2xi32>,
       tensor<4x1xi32>, tensor<1x2xi32>) ->
      (tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>, tensor<4x2xi32>)

  "check.expect_eq"(%res#0, %expected) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#2, %expected) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  "check.expect_eq"(%res#3, %expected) : (tensor<4x2xi32>, tensor<4x2xi32>) -> ()
  func.return
}
