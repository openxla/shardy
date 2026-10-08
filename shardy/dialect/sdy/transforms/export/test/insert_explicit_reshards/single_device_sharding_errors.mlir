// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' -verify-diagnostics -split-input-file

sdy.mesh @single_dev_0 = <[], device_ids=[0]>

// expected-error @+1 {{function argument 0 cannot have a single-device (maximal) sharding attribute}}
func.func @single_device_argument(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@single_dev_0, []>}) -> tensor<8x16xf32> {
  return %arg0 : tensor<8x16xf32>
}

// -----

sdy.mesh @single_dev_0 = <[], device_ids=[0]>

// expected-error @+1 {{function result 0 cannot have a single-device (maximal) sharding attribute}}
func.func @single_device_result(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@single_dev_0, []>}) {
  return %arg0 : tensor<8x16xf32>
}

// -----

sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>

func.func @conflicting_single_device_results(%arg0: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // expected-error @+1 {{has conflicting single-device result shardings}}
  %0:2 = stablehlo.custom_call @ConflictingCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@single_dev_1, []>]>
  } : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %0#0, %0#1 : tensor<8x16xf32>, tensor<8x16xf32>
}
