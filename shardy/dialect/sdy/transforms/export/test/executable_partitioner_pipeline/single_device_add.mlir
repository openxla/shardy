// No need to run this test without HALO export.
// RUN: %S/run_sdy_interpreter_test.sh %s %t "false"

// It doesn't make much sense to run an element-wise op on single-device in
// in real practice. But since we can't hook up an arbitrary custom-call op
// to the StablleHLO interpreter straight-forwardly, we use a simple add op
// here to test the single-device sharding handling in the partitioner
// pipeline.

//--- part1.mlir

sdy.mesh @mesh = <["x"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>

func.func @single_device_add(
    %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  %part_id = stablehlo.partition_id : tensor<ui32>
  %part_id_i32 = stablehlo.convert %part_id : (tensor<ui32>) -> tensor<i32>
  %part_id_bc = stablehlo.broadcast_in_dim %part_id_i32, dims=[] : (tensor<i32>) -> tensor<4xi32>
  %0 = stablehlo.add %arg0, %part_id_bc {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<4xi32>
  return %0 : tensor<4xi32>
}

//--- part2.mlir

func.func @main() {
  %input = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>

  %s0 = "stablehlo.slice"(%input) {start_indices = array<i64: 0>, limit_indices = array<i64: 2>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>
  %s1 = "stablehlo.slice"(%input) {start_indices = array<i64: 2>, limit_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<4xi32>) -> tensor<2xi32>

  %res:2 = "interpreter.run_parallel"(%s0, %s1) {
    programs = [[@single_device_add, @single_device_add]]
  } : (tensor<2xi32>, tensor<2xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  %expected = stablehlo.constant dense<[2, 3, 4, 5]> : tensor<4xi32>
  "check.expect_eq"(%res#0, %expected) : (tensor<4xi32>, tensor<4xi32>) -> ()
  "check.expect_eq"(%res#1, %expected) : (tensor<4xi32>, tensor<4xi32>) -> ()

  return
}
