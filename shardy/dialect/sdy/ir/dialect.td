/* Copyright 2024 The Shardy Authors.\n
\n
Licensed under the Apache License, Version 2.0 (the "License");\n
you may not use this file except in compliance with the License.\n
You may obtain a copy of the License at\n
\n
    http://www.apache.org/licenses/LICENSE-2.0\n
\n
Unless required by applicable law or agreed to in writing, software\n
distributed under the License is distributed on an "AS IS" BASIS,\n
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n
See the License for the specific language governing permissions and\n
limitations under the License.\n
==============================================================================*/

#ifndef SDY_DIALECT
#define SDY_DIALECT

include "mlir/IR/OpBase.td"

def Sdy_Dialect : Dialect {
  let name = "sdy";
  let cppNamespace = "::mlir::sdy";
  let summary = [{
    The Shardy (SDY) dialect defines an axis-based tensor sharding
    representation and additional API components to attach shardings to tensors.
  }];

  let useDefaultAttributePrinterParser = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
  let hasOperationAttrVerify = 1;
}

#endif  // SDY_DIALECT
