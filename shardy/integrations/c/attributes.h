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

#ifndef SHARDY_INTEGRATIONS_C_ATTRIBUTES_H_
#define SHARDY_INTEGRATIONS_C_ATTRIBUTES_H_

#include <stdint.h>
#include <sys/types.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// MeshAxisAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool sdyAttributeIsAMeshAxisAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute sdyMeshAxisAttrGet(MlirContext ctx,
                                                    MlirStringRef name,
                                                    int64_t size);

MLIR_CAPI_EXPORTED MlirStringRef sdyMeshAxisAttrGetName(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t sdyMeshAxisAttrGetSize(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// MeshAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool sdyAttributeIsAMeshAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute sdyMeshAttrGet(MlirContext ctx, intptr_t nAxes,
                                                const MlirAttribute* axes);

MLIR_CAPI_EXPORTED intptr_t sdyMeshAttrGetAxesSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute sdyMeshAttrGetAxesElem(MlirAttribute attr,
                                                        intptr_t pos);

//===----------------------------------------------------------------------===//
// SubAxisInfoAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool sdyAttributeIsASubAxisInfoAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute sdySubAxisInfoAttrGet(MlirContext ctx,
                                                       int64_t preSize,
                                                       int64_t size);

MLIR_CAPI_EXPORTED int64_t sdySubAxisInfoAttrGetPreSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED int64_t sdySubAxisInfoAttrGetSize(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// AxisRefAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool sdyAttributeIsAnAxisRefAttr(MlirAttribute attr);

// NOTE: Pass a null subAxisInfo if the attr has none.
MLIR_CAPI_EXPORTED MlirAttribute sdyAxisRefAttrGet(MlirContext ctx,
                                                   MlirStringRef name,
                                                   MlirAttribute subAxisInfo);

MLIR_CAPI_EXPORTED MlirStringRef sdyAxisRefAttrGetName(MlirAttribute attr);

// NOTE: Attr is null if there is no sub axis info.
MLIR_CAPI_EXPORTED MlirAttribute
sdyAxisRefAttrGetSubAxisInfo(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// DimensionShardingAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool sdyAttributeIsADimensionShardingAttr(
    MlirAttribute attr);

// NOTE: Specify -1 if the attr has no priority.
MLIR_CAPI_EXPORTED MlirAttribute sdyDimensionShardingAttrGet(
    MlirContext ctx, intptr_t nAxes, const MlirAttribute* axes, bool isClosed,
    int64_t priority);

MLIR_CAPI_EXPORTED intptr_t
sdyDimensionShardingAttrGetAxesSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
sdyDimensionShardingAttrGetAxesElem(MlirAttribute attr, intptr_t pos);

MLIR_CAPI_EXPORTED bool sdyDimensionShardingAttrGetIsClosed(MlirAttribute attr);

// NOTE: returns -1 if the attr has no priority.
MLIR_CAPI_EXPORTED int64_t
sdyDimensionShardingAttrGetPriority(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TensorShardingAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool sdyAttributeIsATensorShardingAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute sdyTensorShardingAttrGet(
    MlirContext ctx, MlirStringRef meshName, intptr_t nDimShardings,
    const MlirAttribute* dimShardings, intptr_t nReplicatedAxes,
    const MlirAttribute* replicatedAxes);

MLIR_CAPI_EXPORTED MlirStringRef
sdyTensorShardingAttrGetMeshName(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t
sdyTensorShardingAttrGetDimShardingsSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
sdyTensorShardingAttrGetDimShardingsElem(MlirAttribute attr, intptr_t pos);

MLIR_CAPI_EXPORTED intptr_t
sdyTensorShardingAttrGetReplicatedAxesSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
sdyTensorShardingAttrGetReplicatedAxesElem(MlirAttribute attr, intptr_t pos);

//===----------------------------------------------------------------------===//
// TensorShardingPerValueAttr
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool sdyAttributeIsATensorShardingPerValueAttr(
    MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute sdyTensorShardingPerValueAttrGet(
    MlirContext ctx, intptr_t nShardings, const MlirAttribute* shardings);

MLIR_CAPI_EXPORTED intptr_t
sdyTensorShardingPerValueAttrGetShardingsSize(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute
sdyTensorShardingPerValueAttrGetShardingsElem(MlirAttribute attr, intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif  // SHARDY_INTEGRATIONS_C_ATTRIBUTES_H_
