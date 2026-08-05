/* Copyright 2026 The Shardy Authors.

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

#ifndef SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PASSES_TD
#define SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PASSES_TD

include "mlir/Pass/PassBase.td"

def InlineMeshesPass : Pass<"sdy-inline-meshes", "ModuleOp"> {
  let summary = "Inlines the mesh definition into sharding attributes.";
  let description = [{
    Inlines the mesh definition into sharding attributes.

    This pass is used to make sharding attributes self-contained, so that
    we can safely erase `sdy.mesh` operations during export without leaving
    dangling symbol references.
  }];
  let dependentDialects = ["mlir::sdy::SdyDialect", "mlir::stablehlo::StablehloDialect"];
}

#endif // SHARDY_DIALECT_SDY_TRANSFORMS_COMMON_PASSES_TD
