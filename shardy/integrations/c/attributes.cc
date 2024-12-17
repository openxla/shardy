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

#include "shardy/integrations/c/attributes.h"

#include <cstdint>
#include <optional>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace {

namespace sdy = ::mlir::sdy;

template <typename AttrTy>
AttrTy unwrapAttr(MlirAttribute attr) {
  return mlir::cast<AttrTy>(unwrap(attr));
}

template <typename AttrTy>
mlir::ArrayRef<AttrTy> unwrapAttrs(const MlirAttribute* attrs,
                                   intptr_t nAttrs) {
  return mlir::ArrayRef(reinterpret_cast<const AttrTy*>(attrs), nAttrs);
}

}  // namespace

//===----------------------------------------------------------------------===//
// MeshAxisAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsAMeshAxisAttr(MlirAttribute attr) {
  return mlir::isa<sdy::MeshAxisAttr>(unwrap(attr));
}

MlirAttribute sdyMeshAxisAttrGet(MlirContext ctx, MlirStringRef name,
                                 int64_t size) {
  return wrap(sdy::MeshAxisAttr::get(unwrap(ctx), unwrap(name), size));
}

MlirStringRef sdyMeshAxisAttrGetName(MlirAttribute attr) {
  return wrap(unwrapAttr<sdy::MeshAxisAttr>(attr).getName());
}

int64_t sdyMeshAxisAttrGetSize(MlirAttribute attr) {
  return unwrapAttr<sdy::MeshAxisAttr>(attr).getSize();
}

//===----------------------------------------------------------------------===//
// MeshAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsAMeshAttr(MlirAttribute attr) {
  return mlir::isa<sdy::MeshAttr>(unwrap(attr));
}

MlirAttribute sdyMeshAttrGet(MlirContext ctx, intptr_t nAxes,
                             const MlirAttribute* axes, intptr_t nDeviceIds,
                             const int64_t* deviceIds) {
  return wrap(sdy::MeshAttr::get(unwrap(ctx),
                                 unwrapAttrs<sdy::MeshAxisAttr>(axes, nAxes),
                                 mlir::ArrayRef(deviceIds, nDeviceIds)));
}

int64_t sdyMeshAttrGetDeviceIdsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::MeshAttr>(attr).getDeviceIds().size();
}

int64_t sdyMeshAttrGetDeviceIdsElem(MlirAttribute attr, int64_t pos) {
  return unwrapAttr<sdy::MeshAttr>(attr).getDeviceIds()[pos];
}

intptr_t sdyMeshAttrGetAxesSize(MlirAttribute attr) {
  return unwrapAttr<sdy::MeshAttr>(attr).getAxes().size();
}

MlirAttribute sdyMeshAttrGetAxesElem(MlirAttribute attr, intptr_t pos) {
  return wrap(unwrapAttr<sdy::MeshAttr>(attr).getAxes()[pos]);
}

//===----------------------------------------------------------------------===//
// SubAxisInfoAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsASubAxisInfoAttr(MlirAttribute attr) {
  return mlir::isa<sdy::SubAxisInfoAttr>(unwrap(attr));
}

MlirAttribute sdySubAxisInfoAttrGet(MlirContext ctx, int64_t preSize,
                                    int64_t size) {
  return wrap(sdy::SubAxisInfoAttr::get(unwrap(ctx), preSize, size));
}

int64_t sdySubAxisInfoAttrGetPreSize(MlirAttribute attr) {
  return unwrapAttr<sdy::SubAxisInfoAttr>(attr).getPreSize();
}

int64_t sdySubAxisInfoAttrGetSize(MlirAttribute attr) {
  return unwrapAttr<sdy::SubAxisInfoAttr>(attr).getSize();
}

//===----------------------------------------------------------------------===//
// AxisRefAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsAnAxisRefAttr(MlirAttribute attr) {
  return mlir::isa<sdy::AxisRefAttr>(unwrap(attr));
}

MlirAttribute sdyAxisRefAttrGet(MlirContext ctx, MlirStringRef name,
                                MlirAttribute subAxisInfo) {
  return wrap(sdy::AxisRefAttr::get(
      unwrap(ctx), unwrap(name),
      subAxisInfo.ptr != nullptr
          ? unwrapAttr<sdy::SubAxisInfoAttr>(subAxisInfo)
          : sdy::SubAxisInfoAttr()));
}

MlirStringRef sdyAxisRefAttrGetName(MlirAttribute attr) {
  return wrap(unwrapAttr<sdy::AxisRefAttr>(attr).getName());
}

MlirAttribute sdyAxisRefAttrGetSubAxisInfo(MlirAttribute attr) {
  sdy::SubAxisInfoAttr subAsisInfo =
      unwrapAttr<sdy::AxisRefAttr>(attr).getSubAxisInfo();
  return subAsisInfo ? wrap(subAsisInfo) : MlirAttribute();
}

//===----------------------------------------------------------------------===//
// DimensionShardingAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsADimensionShardingAttr(MlirAttribute attr) {
  return mlir::isa<sdy::DimensionShardingAttr>(unwrap(attr));
}

MlirAttribute sdyDimensionShardingAttrGet(MlirContext ctx, intptr_t nAxes,
                                          const MlirAttribute* axes,
                                          bool isClosed, int64_t priority) {
  return wrap(sdy::DimensionShardingAttr::get(
      unwrap(ctx), unwrapAttrs<sdy::AxisRefAttr>(axes, nAxes), isClosed,
      priority == -1 ? std::nullopt : std::make_optional(priority)));
}

intptr_t sdyDimensionShardingAttrGetAxesSize(MlirAttribute attr) {
  return unwrapAttr<sdy::DimensionShardingAttr>(attr).getAxes().size();
}

MlirAttribute sdyDimensionShardingAttrGetAxesElem(MlirAttribute attr,
                                                  intptr_t pos) {
  return wrap(
      unwrapAttr<sdy::DimensionShardingAttr>(attr).getAxes()[pos]);
}

bool sdyDimensionShardingAttrGetIsClosed(MlirAttribute attr) {
  return unwrapAttr<sdy::DimensionShardingAttr>(attr).getIsClosed();
}

int64_t sdyDimensionShardingAttrGetPriority(MlirAttribute attr) {
  std::optional<int64_t> priority =
      unwrapAttr<sdy::DimensionShardingAttr>(attr).getPriority();
  return priority.has_value() ? *priority : -1;
}

//===----------------------------------------------------------------------===//
// TensorShardingAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsATensorShardingAttr(MlirAttribute attr) {
  return mlir::isa<sdy::TensorShardingAttr>(unwrap(attr));
}

MlirAttribute sdyTensorShardingAttrGet(MlirContext ctx, MlirAttribute meshOrRef,
                                       intptr_t nDimShardings,
                                       const MlirAttribute* dimShardings,
                                       intptr_t nReplicatedAxes,
                                       const MlirAttribute* replicatedAxes) {
  return wrap(sdy::TensorShardingAttr::get(
      unwrap(ctx), unwrap(meshOrRef),
      unwrapAttrs<sdy::DimensionShardingAttr>(dimShardings, nDimShardings),
      unwrapAttrs<sdy::AxisRefAttr>(replicatedAxes, nReplicatedAxes)));
}

MlirAttribute sdyTensorShardingAttrGetMeshOrRef(MlirAttribute attr) {
  return wrap(unwrapAttr<sdy::TensorShardingAttr>(attr).getMeshOrRef());
}

intptr_t sdyTensorShardingAttrGetDimShardingsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::TensorShardingAttr>(attr).getDimShardings().size();
}

MlirAttribute sdyTensorShardingAttrGetDimShardingsElem(MlirAttribute attr,
                                                       intptr_t pos) {
  return wrap(
      unwrapAttr<sdy::TensorShardingAttr>(attr).getDimShardings()[pos]);
}

intptr_t sdyTensorShardingAttrGetReplicatedAxesSize(MlirAttribute attr) {
  return unwrapAttr<sdy::TensorShardingAttr>(attr).getReplicatedAxes().size();
}

MlirAttribute sdyTensorShardingAttrGetReplicatedAxesElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return wrap(
      unwrapAttr<sdy::TensorShardingAttr>(attr).getReplicatedAxes()[pos]);
}

//===----------------------------------------------------------------------===//
// TensorShardingPerValueAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsATensorShardingPerValueAttr(MlirAttribute attr) {
  return mlir::isa<sdy::TensorShardingPerValueAttr>(unwrap(attr));
}

MlirAttribute sdyTensorShardingPerValueAttrGet(MlirContext ctx,
                                               intptr_t nShardings,
                                               const MlirAttribute* shardings) {
  return wrap(sdy::TensorShardingPerValueAttr::get(
      unwrap(ctx),
      unwrapAttrs<sdy::TensorShardingAttr>(shardings, nShardings)));
}

intptr_t sdyTensorShardingPerValueAttrGetShardingsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::TensorShardingPerValueAttr>(attr)
      .getShardings()
      .size();
}

MlirAttribute sdyTensorShardingPerValueAttrGetShardingsElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return wrap(
      unwrapAttr<sdy::TensorShardingPerValueAttr>(attr).getShardings()[pos]);
}

//===----------------------------------------------------------------------===//
// DimMappingAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsADimMappingAttr(MlirAttribute attr) {
  return mlir::isa<sdy::DimMappingAttr>(unwrap(attr));
}

MlirAttribute sdyDimMappingAttrGet(MlirContext ctx, intptr_t nFactorIndices,
                                   const int64_t* factorIndices) {
  return wrap(sdy::DimMappingAttr::get(
      unwrap(ctx), mlir::ArrayRef(factorIndices, nFactorIndices)));
}

intptr_t sdyDimMappingAttrGetFactorIndicesSize(MlirAttribute attr) {
  return unwrapAttr<sdy::DimMappingAttr>(attr).getFactorIndices().size();
}

int64_t sdyDimMappingAttrGetFactorIndicesElem(MlirAttribute attr,
                                              intptr_t pos) {
  return unwrapAttr<sdy::DimMappingAttr>(attr).getFactorIndices()[pos];
}

//===----------------------------------------------------------------------===//
// TensorMappingAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsATensorMappingAttr(MlirAttribute attr) {
  return mlir::isa<sdy::TensorMappingAttr>(unwrap(attr));
}

MlirAttribute sdyTensorMappingAttrGet(MlirContext ctx, intptr_t nMappings,
                                      const MlirAttribute* mappings) {
  return wrap(sdy::TensorMappingAttr::get(
      unwrap(ctx), unwrapAttrs<sdy::DimMappingAttr>(mappings, nMappings)));
}

intptr_t sdyTensorMappingAttrGetRank(MlirAttribute attr) {
  return unwrapAttr<sdy::TensorMappingAttr>(attr).getRank();
}

intptr_t sdyTensorMappingAttrGetDimMappingsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::TensorMappingAttr>(attr).getDimMappings().size();
}

MlirAttribute sdyTensorMappingAttrGetDimMappingsElem(MlirAttribute attr,
                                                     intptr_t pos) {
  return wrap(unwrapAttr<sdy::TensorMappingAttr>(attr).getDimMappings()[pos]);
}

//===----------------------------------------------------------------------===//
// OpShardingRuleAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsAOpShardingRuleAttr(MlirAttribute attr) {
  return mlir::isa<sdy::OpShardingRuleAttr>(unwrap(attr));
}

MlirAttribute sdyOpShardingRuleAttrGet(
    MlirContext ctx, intptr_t nFactorSizes, const int64_t* factorSizes,
    intptr_t nOperandMappings, const MlirAttribute* operandMappings,
    intptr_t nResultMappings, const MlirAttribute* resultMappings,
    intptr_t nReductionFactors, const int64_t* reductionFactors,
    intptr_t nNeedReplicationFactors, const int64_t* needReplicationFactors,
    bool isCustomRule) {
  return wrap(sdy::OpShardingRuleAttr::get(
      unwrap(ctx), mlir::ArrayRef(factorSizes, nFactorSizes),
      unwrapAttrs<sdy::TensorMappingAttr>(operandMappings, nOperandMappings),
      unwrapAttrs<sdy::TensorMappingAttr>(resultMappings, nResultMappings),
      mlir::ArrayRef(reductionFactors, nReductionFactors),
      mlir::ArrayRef(needReplicationFactors, nNeedReplicationFactors),
      isCustomRule));
}

bool sdyOpShardingRuleAttrGetIsCustom(MlirAttribute attr) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr).isCustom();
}

intptr_t sdyOpShardingRuleAttrGetFactorSizesSize(MlirAttribute attr) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr).getFactorSizes().size();
}

int64_t sdyOpShardingRuleAttrGetFactorSizesElem(MlirAttribute attr,
                                                intptr_t pos) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr).getFactorSizes()[pos];
}

intptr_t sdyOpShardingRuleAttrGetOperandMappingsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr).getOperandMappings().size();
}

MlirAttribute sdyOpShardingRuleAttrGetOperandMappingsElem(MlirAttribute attr,
                                                          intptr_t pos) {
  return wrap(
      unwrapAttr<sdy::OpShardingRuleAttr>(attr).getOperandMappings()[pos]);
}

intptr_t sdyOpShardingRuleAttrGetResultMappingsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr).getResultMappings().size();
}

MlirAttribute sdyOpShardingRuleAttrGetResultMappingsElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return wrap(
      unwrapAttr<sdy::OpShardingRuleAttr>(attr).getResultMappings()[pos]);
}

int64_t sdyOpShardingRuleAttrGetReductionFactorsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr).getReductionFactors().size();
}

intptr_t sdyOpShardingRuleAttrGetReductionFactorsElem(MlirAttribute attr,
                                                      intptr_t pos) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr).getReductionFactors()[pos];
}

int64_t sdyOpShardingRuleAttrGetNeedReplicationFactorsSize(MlirAttribute attr) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr)
      .getNeedReplicationFactors()
      .size();
}

intptr_t sdyOpShardingRuleAttrGetNeedReplicationFactorsElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return unwrapAttr<sdy::OpShardingRuleAttr>(attr)
      .getNeedReplicationFactors()[pos];
}

//===----------------------------------------------------------------------===//
// ManualAxesAttr
//===----------------------------------------------------------------------===//

bool sdyAttributeIsAManualAxesAttr(MlirAttribute attr) {
  return mlir::isa<sdy::ManualAxesAttr>(unwrap(attr));
}

MlirAttribute sdyManualAxesAttrGet(
    MlirContext ctx, intptr_t nAxes, const MlirAttribute* axes) {
  return wrap(sdy::ManualAxesAttr::get(
      unwrap(ctx), unwrapAttrs<mlir::StringAttr>(axes, nAxes)));
}

intptr_t sdyManualAxesAttrGetAxesSize(MlirAttribute attr) {
  return unwrapAttr<sdy::ManualAxesAttr>(attr).size();
}

MlirStringRef sdyManualAxesAttrGetAxesElem(
  MlirAttribute attr, intptr_t pos) {
  return wrap(unwrapAttr<sdy::ManualAxesAttr>(attr)[pos].getValue());
}
