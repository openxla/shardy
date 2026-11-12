// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=true
// RUN: %S/run_sdy_interpreter_test.sh %s %t --enable_halo_exchange=false

//--- part1.mlir

sdy.mesh @mesh_x2_y2 = <["x"=2, "y"=2]>

// Non-zero pad on both non-contracting (dim 0: 2->4) and sharded contracting
// (dim 1: 2->4) dimensions feeding dot_general and dot.
func.func @parallel_dot_general_nonzero_pad(
  %arg0: tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
  %arg1: tensor<4x5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
  -> (tensor<4x5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh_x2_y2, [{"x"}, {"y"}]> : tensor<2x2xi32>
  %cst = stablehlo.constant dense<3> : tensor<i32>
  %pad = stablehlo.pad %0, %cst, low = [1, 1], high = [1, 1], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{"x"}, {"y"}]>]>}
    : (tensor<2x2xi32>, tensor<i32>) -> tensor<4x4xi32>
  %1 = sdy.reshard %arg1 <@mesh_x2_y2, [{"y"}, {}]> : tensor<4x5xi32>
  %2 = stablehlo.dot_general %pad, %1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{"x"}, {}], unreduced={"y"}>]>}
    : (tensor<4x4xi32>, tensor<4x5xi32>) -> tensor<4x5xi32>
  %3 = sdy.reshard %2 <@mesh_x2_y2, [{}, {}]> : tensor<4x5xi32>
  return %3 : tensor<4x5xi32>
}

func.func @sequential_dot_general_nonzero_pad(
  %arg0: tensor<2x2xi32>, %arg1: tensor<4x5xi32>) -> tensor<4x5xi32> {
  %cst = stablehlo.constant dense<3> : tensor<i32>
  %pad = stablehlo.pad %arg0, %cst, low = [1, 1], high = [1, 1], interior = [0, 0]
    : (tensor<2x2xi32>, tensor<i32>) -> tensor<4x4xi32>
  %0 = stablehlo.dot_general %pad, %arg1, contracting_dims = [1] x [0]
    : (tensor<4x4xi32>, tensor<4x5xi32>) -> tensor<4x5xi32>
  return %0 : tensor<4x5xi32>
}

func.func @parallel_dot_general_noncontracting_nonzero_pad(
  %arg0: tensor<3x4xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>},
  %arg1: tensor<4x5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>})
  -> (tensor<5x5xi32> {sdy.sharding = #sdy.sharding<@mesh_x2_y2, [{}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh_x2_y2, [{}, {"y"}]> : tensor<3x4xi32>
  %cst = stablehlo.constant dense<-2> : tensor<i32>
  %pad = stablehlo.pad %0, %cst, low = [1, 0], high = [1, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x2_y2, [{}, {"y"}]>]>}
    : (tensor<3x4xi32>, tensor<i32>) -> tensor<5x4xi32>
  %1 = sdy.reshard %arg1 <@mesh_x2_y2, [{"y"}, {}]> : tensor<4x5xi32>
  %2 = stablehlo.dot_general %pad, %1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_x2_y2, [{}, {}], unreduced={"y"}>]>}
    : (tensor<5x4xi32>, tensor<4x5xi32>) -> tensor<5x5xi32>
  %3 = sdy.reshard %2 <@mesh_x2_y2, [{}, {}]> : tensor<5x5xi32>
  return %3 : tensor<5x5xi32>
}

func.func @sequential_dot_general_noncontracting_nonzero_pad(
  %arg0: tensor<3x4xi32>, %arg1: tensor<4x5xi32>) -> tensor<5x5xi32> {
  %cst = stablehlo.constant dense<-2> : tensor<i32>
  %pad = stablehlo.pad %arg0, %cst, low = [1, 0], high = [1, 0], interior = [0, 0]
    : (tensor<3x4xi32>, tensor<i32>) -> tensor<5x4xi32>
  %0 = stablehlo.dot_general %pad, %arg1, contracting_dims = [1] x [0] : (tensor<5x4xi32>, tensor<4x5xi32>) -> tensor<5x5xi32>
  return %0 : tensor<5x5xi32>
}

//--- part2.mlir

func.func @main() {
  %lhs_dg = stablehlo.constant dense<[[2, -5], [7, 11]]> : tensor<2x2xi32>
  %lhs_dot = stablehlo.constant dense<[[1, 2, 3, 4], [-5, 6, -7, 8], [9, -10, 11, -12]]> : tensor<3x4xi32>
  %rhs = stablehlo.constant dense<[
    [1, -2, 3, -4, 5],
    [6, 7, -8, 9, -10],
    [-11, 12, 13, -14, 15],
    [16, -17, 18, 19, -20]
  ]> : tensor<4x5xi32>

  %seq_dg = func.call @sequential_dot_general_nonzero_pad(%lhs_dg, %rhs) : (tensor<2x2xi32>, tensor<4x5xi32>) -> tensor<4x5xi32>
  %res_dg:4 = "interpreter.run_parallel"(
    %lhs_dg, %rhs, %lhs_dg, %rhs, %lhs_dg, %rhs, %lhs_dg, %rhs
  ) {
    programs = [[
      @parallel_dot_general_nonzero_pad,
      @parallel_dot_general_nonzero_pad,
      @parallel_dot_general_nonzero_pad,
      @parallel_dot_general_nonzero_pad
    ]]
  } : (
    tensor<2x2xi32>, tensor<4x5xi32>,
    tensor<2x2xi32>, tensor<4x5xi32>,
    tensor<2x2xi32>, tensor<4x5xi32>,
    tensor<2x2xi32>, tensor<4x5xi32>
  ) -> (
    tensor<4x5xi32>, tensor<4x5xi32>, tensor<4x5xi32>, tensor<4x5xi32>
  )
  "check.expect_eq"(%res_dg#0, %seq_dg) : (tensor<4x5xi32>, tensor<4x5xi32>) -> ()
  "check.expect_eq"(%res_dg#1, %seq_dg) : (tensor<4x5xi32>, tensor<4x5xi32>) -> ()
  "check.expect_eq"(%res_dg#2, %seq_dg) : (tensor<4x5xi32>, tensor<4x5xi32>) -> ()
  "check.expect_eq"(%res_dg#3, %seq_dg) : (tensor<4x5xi32>, tensor<4x5xi32>) -> ()

  %seq_dot = func.call @sequential_dot_general_noncontracting_nonzero_pad(%lhs_dot, %rhs) : (tensor<3x4xi32>, tensor<4x5xi32>) -> tensor<5x5xi32>
  %res_dot:4 = "interpreter.run_parallel"(
    %lhs_dot, %rhs, %lhs_dot, %rhs, %lhs_dot, %rhs, %lhs_dot, %rhs
  ) {
    programs = [[
      @parallel_dot_general_noncontracting_nonzero_pad,
      @parallel_dot_general_noncontracting_nonzero_pad,
      @parallel_dot_general_noncontracting_nonzero_pad,
      @parallel_dot_general_noncontracting_nonzero_pad
    ]]
  } : (
    tensor<3x4xi32>, tensor<4x5xi32>,
    tensor<3x4xi32>, tensor<4x5xi32>,
    tensor<3x4xi32>, tensor<4x5xi32>,
    tensor<3x4xi32>, tensor<4x5xi32>
  ) -> (
    tensor<5x5xi32>, tensor<5x5xi32>, tensor<5x5xi32>, tensor<5x5xi32>
  )
  "check.expect_eq"(%res_dot#0, %seq_dot) : (tensor<5x5xi32>, tensor<5x5xi32>) -> ()
  "check.expect_eq"(%res_dot#1, %seq_dot) : (tensor<5x5xi32>, tensor<5x5xi32>) -> ()
  "check.expect_eq"(%res_dot#2, %seq_dot) : (tensor<5x5xi32>, tensor<5x5xi32>) -> ()
  "check.expect_eq"(%res_dot#3, %seq_dot) : (tensor<5x5xi32>, tensor<5x5xi32>) -> ()

  return
}
