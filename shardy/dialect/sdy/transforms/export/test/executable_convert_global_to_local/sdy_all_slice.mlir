// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

func.func @all_slice(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_2, [{"x"}, {}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

func.func @all_slice_subaxis(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x", "y":(2)2}]>}) {
  %0 = sdy.all_slice [{}, {"x", "y":(2)2}] %arg0 out_sharding=<@mesh_2_4, [{}, {"x", "y":(2)2}]> : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

//--- part2.mlir

func.func @main() {
  %cst = stablehlo.constant dense<[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
  ]> : tensor<4x4xi32>

  %res:2 = "interpreter.run_parallel"(%cst, %cst) {
    programs = [[@all_slice, @all_slice]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>) -> (tensor<2x4xi32>, tensor<2x4xi32>)

  %e0 = "stablehlo.slice"(%cst) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 2, 4>, strides = array<i64: 1, 1>
  } : (tensor<4x4xi32>) -> tensor<2x4xi32>
  %e1 = "stablehlo.slice"(%cst) {
    start_indices = array<i64: 2, 0>, limit_indices = array<i64: 4, 4>, strides = array<i64: 1, 1>
  } : (tensor<4x4xi32>) -> tensor<2x4xi32>

  // Check element-wise correctness for both devices.
  "check.expect_eq"(%res#0, %e0) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()
  "check.expect_eq"(%res#1, %e1) : (tensor<2x4xi32>, tensor<2x4xi32>) -> ()

  // Subaxis test
  %cst_sub = stablehlo.constant dense<[
    [1,  2,  3,  4,  5,  6,  7,  8],
    [9,  10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56],
    [57, 58, 59, 60, 61, 62, 63, 64]
  ]> : tensor<8x8xi32>

  %res_sub:8 = "interpreter.run_parallel"(%cst_sub, %cst_sub, %cst_sub, %cst_sub, %cst_sub, %cst_sub, %cst_sub, %cst_sub) {
    programs = [[
      @all_slice_subaxis, @all_slice_subaxis, @all_slice_subaxis, @all_slice_subaxis,
      @all_slice_subaxis, @all_slice_subaxis, @all_slice_subaxis, @all_slice_subaxis
    ]]
  } : (tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>,
       tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>, tensor<8x8xi32>)
    -> (tensor<8x2xi32>, tensor<8x2xi32>, tensor<8x2xi32>, tensor<8x2xi32>,
        tensor<8x2xi32>, tensor<8x2xi32>, tensor<8x2xi32>, tensor<8x2xi32>)

  %e_sub0 = "stablehlo.slice"(%cst_sub) {
    start_indices = array<i64: 0, 0>, limit_indices = array<i64: 8, 2>, strides = array<i64: 1, 1>
  } : (tensor<8x8xi32>) -> tensor<8x2xi32>
  %e_sub1 = "stablehlo.slice"(%cst_sub) {
    start_indices = array<i64: 0, 2>, limit_indices = array<i64: 8, 4>, strides = array<i64: 1, 1>
  } : (tensor<8x8xi32>) -> tensor<8x2xi32>
  %e_sub2 = "stablehlo.slice"(%cst_sub) {
    start_indices = array<i64: 0, 4>, limit_indices = array<i64: 8, 6>, strides = array<i64: 1, 1>
  } : (tensor<8x8xi32>) -> tensor<8x2xi32>
  %e_sub3 = "stablehlo.slice"(%cst_sub) {
    start_indices = array<i64: 0, 6>, limit_indices = array<i64: 8, 8>, strides = array<i64: 1, 1>
  } : (tensor<8x8xi32>) -> tensor<8x2xi32>

  "check.expect_eq"(%res_sub#0, %e_sub0) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()
  "check.expect_eq"(%res_sub#1, %e_sub1) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()
  "check.expect_eq"(%res_sub#2, %e_sub0) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()
  "check.expect_eq"(%res_sub#3, %e_sub1) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()
  "check.expect_eq"(%res_sub#4, %e_sub2) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()
  "check.expect_eq"(%res_sub#5, %e_sub3) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()
  "check.expect_eq"(%res_sub#6, %e_sub2) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()
  "check.expect_eq"(%res_sub#7, %e_sub3) : (tensor<8x2xi32>, tensor<8x2xi32>) -> ()

  return
}
