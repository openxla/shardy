
// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// All-reduce across the entire mesh (both "x" and "y" axes).
func.func @all_reduce_xy(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>}) {
  %0 = sdy.all_reduce {"x", "y"} %arg0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// All-reduce across only the "x" axis.
// In a 2x2 mesh, this creates two replica groups: {0, 2} and {1, 3}.
func.func @all_reduce_x(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>})
  -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {}]>}) {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2_2, [{}, {}]> : tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

//--- part2.mlir

func.func @main() {
  %c1 = stablehlo.constant dense<1> : tensor<4x4xi32>
  %c10 = stablehlo.constant dense<10> : tensor<4x4xi32>
  %c100 = stablehlo.constant dense<100> : tensor<4x4xi32>
  %c1000 = stablehlo.constant dense<1000> : tensor<4x4xi32>

  // 1. Test All-reduce across both axes ("x" and "y").
  // Sum = 1 + 10 + 100 + 1000 = 1111.
  %res_xy:4 = "interpreter.run_parallel"(%c1, %c10, %c100, %c1000) {
    programs = [[@all_reduce_xy, @all_reduce_xy, @all_reduce_xy, @all_reduce_xy]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) ->
      (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>)

  %expected_xy = stablehlo.constant dense<1111> : tensor<4x4xi32>
  "check.expect_eq"(%res_xy#0, %expected_xy) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_xy#3, %expected_xy) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()

  // 2. Test All-reduce across only "x" axis.
  // Device grid (x, y):
  // (0,0): dev 0, input 1
  // (0,1): dev 1, input 10
  // (1,0): dev 2, input 100
  // (1,1): dev 3, input 1000
  // Replica groups for "x" (dim 0): {0, 2} and {1, 3}.
  %res_x:4 = "interpreter.run_parallel"(%c1, %c10, %c100, %c1000) {
    programs = [[@all_reduce_x, @all_reduce_x, @all_reduce_x, @all_reduce_x]]
  } : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) ->
      (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>)

  %expected_x_02 = stablehlo.constant dense<101> : tensor<4x4xi32> // 1 + 100
  %expected_x_13 = stablehlo.constant dense<1010> : tensor<4x4xi32> // 10 + 1000

  "check.expect_eq"(%res_x#0, %expected_x_02) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_x#2, %expected_x_02) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_x#1, %expected_x_13) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()
  "check.expect_eq"(%res_x#3, %expected_x_13) : (tensor<4x4xi32>, tensor<4x4xi32>) -> ()

  return
}
