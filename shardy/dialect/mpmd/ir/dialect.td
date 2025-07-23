/* Copyright 2025 The MPMD Authors.

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

#ifndef MPMD_DIALECT
#define MPMD_DIALECT

include "mlir/IR/OpBase.td"

// Introduce the MPMD dialect to ODS framework.
// The MPMD dialect allows partitioning a single MLIR program into multiple
// fragments, each represents a sub-computation of an SPMD program on a mesh
// within an MPMD topology of meshes.
def Mpmd_Dialect : Dialect {
  let name = "mpmd";
  let cppNamespace = "::mlir::mpmd";

  let dependentDialects = [
      // We need to load the SDY dialect as it contains the sharding attributes
      // and interfaces.
      "sdy::SdyDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif
