/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SDY_DIALECT
#define SDY_DIALECT

include "mlir/IR/OpBase.td"

def Sdy_Dialect : Dialect {
  let name = "sdy";
  let cppNamespace = "::mlir::sdy";
  let summary = "The Shardy (SDY) dialect";

  let description = [{
    The Shardy (SDY) dialect defines an axis-based tensor sharding
    representation and additional API components to attach shardings to tensors.

    Version log:
      0.0.1: Add unreduced axes to TensorShardingAttr.
  }];

  let useDefaultAttributePrinterParser = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
  let hasOperationAttrVerify = 1;
}

#endif  // SDY_DIALECT
