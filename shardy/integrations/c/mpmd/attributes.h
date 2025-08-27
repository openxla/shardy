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

#ifndef SHARDY_INTEGRATIONS_C_MPMD_ATTRIBUTES_H_
#define SHARDY_INTEGRATIONS_C_MPMD_ATTRIBUTES_H_

#include <stdint.h>
#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// UserOriginAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mpmdAttributeIsAUserOriginAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mpmdUserOriginAttrGet(MlirContext ctx,
                                                       MlirStringRef userName,
                                                       int64_t transposeCount);

MLIR_CAPI_EXPORTED MlirStringRef
mpmdUserOriginAttrGetUserName(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t
mpmdUserOriginAttrGetTransposeCount(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif  // SHARDY_INTEGRATIONS_C_MPMD_ATTRIBUTES_H_
