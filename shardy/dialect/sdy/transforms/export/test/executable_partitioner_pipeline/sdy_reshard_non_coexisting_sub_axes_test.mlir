// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir
sdy.mesh @mesh_6 = <["a"=6]>

func.func @parallel_disjoint(%arg0: tensor<6x6xi32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"a":(1)2}, {}]>}) -> (tensor<6x6xi32> {sdy.sharding = #sdy.sharding<@mesh_6, [{}, {"a":(3)2}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{}, {"a":(3)2}]>]>} : tensor<6x6xi32>
  return %0 : tensor<6x6xi32>
}

func.func @parallel_overlapping(%arg0: tensor<6x6xi32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"a":(1)2}, {}]>}) -> (tensor<6x6xi32> {sdy.sharding = #sdy.sharding<@mesh_6, [{}, {"a":(1)3}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{}, {"a":(1)3}]>]>} : tensor<6x6xi32>
  return %0 : tensor<6x6xi32>
}

func.func @parallel_same_dim(%arg0: tensor<6x6xi32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"a":(1)2}, {}]>}) -> (tensor<6x6xi32> {sdy.sharding = #sdy.sharding<@mesh_6, [{"a":(3)2}, {}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_6, [{"a":(3)2}, {}]>]>} : tensor<6x6xi32>
  return %0 : tensor<6x6xi32>
}

//--- part2.mlir
func.func @main() {
  %flat = stablehlo.iota dim = 0 : tensor<36xi32>
  %c42 = stablehlo.constant dense<42> : tensor<36xi32>
  %flat42 = stablehlo.add %flat, %c42 : tensor<36xi32>
  %input = stablehlo.reshape %flat42 : (tensor<36xi32>) -> tensor<6x6xi32>
  %in_r0 = stablehlo.slice %input [0:3, 0:6] : (tensor<6x6xi32>) -> tensor<3x6xi32>
  %in_r1 = stablehlo.slice %input [3:6, 0:6] : (tensor<6x6xi32>) -> tensor<3x6xi32>
  %exp = func.call @sequential_disjoint(%input) : (tensor<6x6xi32>) -> tensor<6x6xi32>

  %exp_c0 = stablehlo.slice %exp [0:6, 0:3] : (tensor<6x6xi32>) -> tensor<6x3xi32>
  %exp_c1 = stablehlo.slice %exp [0:6, 3:6] : (tensor<6x6xi32>) -> tensor<6x3xi32>
  %d:6 = "interpreter.run_parallel"(%in_r0, %in_r0, %in_r0, %in_r1, %in_r1, %in_r1) {
    programs = [[@parallel_disjoint, @parallel_disjoint, @parallel_disjoint, @parallel_disjoint, @parallel_disjoint, @parallel_disjoint]]
  } : (tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>) -> (tensor<6x3xi32>, tensor<6x3xi32>, tensor<6x3xi32>, tensor<6x3xi32>, tensor<6x3xi32>, tensor<6x3xi32>)
  "check.expect_eq"(%d#0, %exp_c0) : (tensor<6x3xi32>, tensor<6x3xi32>) -> ()
  "check.expect_eq"(%d#1, %exp_c1) : (tensor<6x3xi32>, tensor<6x3xi32>) -> ()
  "check.expect_eq"(%d#2, %exp_c0) : (tensor<6x3xi32>, tensor<6x3xi32>) -> ()
  "check.expect_eq"(%d#3, %exp_c1) : (tensor<6x3xi32>, tensor<6x3xi32>) -> ()
  "check.expect_eq"(%d#4, %exp_c0) : (tensor<6x3xi32>, tensor<6x3xi32>) -> ()
  "check.expect_eq"(%d#5, %exp_c1) : (tensor<6x3xi32>, tensor<6x3xi32>) -> ()

  %exp_t0 = stablehlo.slice %exp [0:6, 0:2] : (tensor<6x6xi32>) -> tensor<6x2xi32>
  %exp_t1 = stablehlo.slice %exp [0:6, 2:4] : (tensor<6x6xi32>) -> tensor<6x2xi32>
  %exp_t2 = stablehlo.slice %exp [0:6, 4:6] : (tensor<6x6xi32>) -> tensor<6x2xi32>
  %o:6 = "interpreter.run_parallel"(%in_r0, %in_r0, %in_r0, %in_r1, %in_r1, %in_r1) {
    programs = [[@parallel_overlapping, @parallel_overlapping, @parallel_overlapping, @parallel_overlapping, @parallel_overlapping, @parallel_overlapping]]
  } : (tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>) -> (tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>, tensor<6x2xi32>)
  "check.expect_eq"(%o#0, %exp_t0) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%o#1, %exp_t0) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%o#2, %exp_t1) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%o#3, %exp_t1) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%o#4, %exp_t2) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()
  "check.expect_eq"(%o#5, %exp_t2) : (tensor<6x2xi32>, tensor<6x2xi32>) -> ()

  %exp_r0 = stablehlo.slice %exp [0:3, 0:6] : (tensor<6x6xi32>) -> tensor<3x6xi32>
  %exp_r1 = stablehlo.slice %exp [3:6, 0:6] : (tensor<6x6xi32>) -> tensor<3x6xi32>
  %s:6 = "interpreter.run_parallel"(%in_r0, %in_r0, %in_r0, %in_r1, %in_r1, %in_r1) {
    programs = [[@parallel_same_dim, @parallel_same_dim, @parallel_same_dim, @parallel_same_dim, @parallel_same_dim, @parallel_same_dim]]
  } : (tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>) -> (tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>, tensor<3x6xi32>)
  "check.expect_eq"(%s#0, %exp_r0) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%s#1, %exp_r1) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%s#2, %exp_r0) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%s#3, %exp_r1) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%s#4, %exp_r0) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  "check.expect_eq"(%s#5, %exp_r1) : (tensor<3x6xi32>, tensor<3x6xi32>) -> ()
  return
}
