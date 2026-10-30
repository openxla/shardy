// RUN: %S/run_sdy_interpreter_test.sh %s %t

//--- part1.mlir

sdy.mesh @mesh_non_iota = <["x"=2], device_ids=[1, 0]>

func.func @parallel_replicated(%arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}]>})
    -> (tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}], unreduced={"x"}>}) {
  %0 = sdy.replicated_to_unreduced {"x"} %arg0 out_sharding=<@mesh_non_iota, [{}], unreduced={"x"}> : tensor<4xi32>
  return %0 : tensor<4xi32>
}

//--- part2.mlir

func.func @main() {
  %cst = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %zero = stablehlo.constant dense<[0, 0, 0, 0]> : tensor<4xi32>

  %res:2 = "interpreter.run_parallel"(%cst, %cst) {
    programs = [[@parallel_replicated, @parallel_replicated]]
  } : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)

  // Device 0 is non-leader (index 1 in device_ids=[1, 0]), masked to zero.
  "check.expect_eq"(%res#0, %zero) : (tensor<4xi32>, tensor<4xi32>) -> ()
  // Device 1 is leader (index 0 in device_ids=[1, 0]), receives input.
  "check.expect_eq"(%res#1, %cst) : (tensor<4xi32>, tensor<4xi32>) -> ()

  return
}
