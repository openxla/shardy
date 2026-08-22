// No need to run this test without HALO export.
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

//--- part1.mlir

sdy.mesh @mesh = <["x"=4]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>
sdy.mesh @single_dev_2 = <[], device_ids=[2]>

func.func @single_device_switch(
    %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %part_id = stablehlo.partition_id : tensor<ui32>
  %part_id_i32 = stablehlo.convert %part_id : (tensor<ui32>) -> tensor<i32>
  %part_id_bc = stablehlo.broadcast_in_dim %part_id_i32, dims=[] : (tensor<i32>) -> tensor<4xi32>

  %c10 = stablehlo.constant dense<10> : tensor<4xi32>
  %part_id_x10 = stablehlo.multiply %part_id_bc, %c10 : tensor<4xi32>

  %0 = stablehlo.add %arg0, %part_id_bc {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_2, []>]>
  } : tensor<4xi32>

  %1 = stablehlo.add %0, %part_id_x10 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<4xi32>

  return %1 : tensor<4xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 1>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 1>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s2 = "stablehlo.slice"(%input) {start_indices = array<i64: 2>, limit_indices = array<i64: 3>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>
  %s3 = "stablehlo.slice"(%input) {start_indices = array<i64: 3>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<1xi32>

  %res_chained:4 = "interpreter.run_parallel"(%s0, %s1, %s2, %s3) {
    programs = [[@single_device_switch, @single_device_switch, @single_device_switch, @single_device_switch]]
  } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<4xi32>, tensor<4xi32>, tensor<4xi32>, tensor<4xi32>)

  %expected_chained = stablehlo.constant dense<[13, 14, 15, 16]> : tensor<4xi32>
  "check.expect_eq"(%res_chained#0, %expected_chained) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res_chained#1, %expected_chained) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res_chained#2, %expected_chained) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res_chained#3, %expected_chained) : (tensor<4xi32>, tensor<4xi32>) -> ()

  return
}
