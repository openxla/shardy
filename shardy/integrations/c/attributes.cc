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
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace {

namespace sdy = ::mlir::sdy;

template <typename AttrTy>
AttrTy unwrapAttr(MlirAttribute attr) {
  return mlir::cast<AttrTy>(unwrap(attr));
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
                             const MlirAttribute* axes) {
  return wrap(sdy::MeshAttr::get(
      unwrap(ctx),
      mlir::ArrayRef(reinterpret_cast<const sdy::MeshAxisAttr*>(axes), nAxes)));
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
      unwrap(ctx),
      mlir::ArrayRef(reinterpret_cast<const sdy::AxisRefAttr*>(axes), nAxes),
      isClosed, priority == -1 ? std::nullopt : std::make_optional(priority)));
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

MlirAttribute sdyTensorShardingAttrGet(MlirContext ctx, MlirStringRef meshName,
                                       intptr_t nDimShardings,
                                       const MlirAttribute* dimShardings,
                                       intptr_t nReplicatedAxes,
                                       const MlirAttribute* replicatedAxes) {
  return wrap(sdy::TensorShardingAttr::get(
      unwrap(ctx), unwrap(meshName),
      mlir::ArrayRef(
          reinterpret_cast<const sdy::DimensionShardingAttr*>(dimShardings),
          nDimShardings),
      mlir::ArrayRef(reinterpret_cast<const sdy::AxisRefAttr*>(replicatedAxes),
                     nReplicatedAxes)));
}

MlirStringRef sdyTensorShardingAttrGetMeshName(MlirAttribute attr) {
  return wrap(mlir::cast<sdy::TensorShardingAttr>(unwrap(attr))
                  .getMeshSymName()
                  .getValue());
}

intptr_t sdyTensorShardingAttrGetDimShardingsSize(MlirAttribute attr) {
  return mlir::cast<sdy::TensorShardingAttr>(unwrap(attr))
      .getDimShardings()
      .size();
}

MlirAttribute sdyTensorShardingAttrGetDimShardingsElem(MlirAttribute attr,
                                                       intptr_t pos) {
  return wrap(
      unwrapAttr<sdy::TensorShardingAttr>(attr).getDimShardings()[pos]);
}

intptr_t sdyTensorShardingAttrGetReplicatedAxesSize(MlirAttribute attr) {
  return mlir::cast<sdy::TensorShardingAttr>(unwrap(attr))
      .getReplicatedAxes()
      .size();
}

MlirAttribute sdyTensorShardingAttrGetReplicatedAxesElem(MlirAttribute attr,
                                                         intptr_t pos) {
  return wrap(mlir::cast<sdy::TensorShardingAttr>(unwrap(attr))
                  .getReplicatedAxes()[pos]);
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
      mlir::ArrayRef(
          reinterpret_cast<const sdy::TensorShardingAttr*>(shardings),
          nShardings)));
}

intptr_t sdyTensorShardingPerValueAttrGetShardingsSize(MlirAttribute attr) {
  return mlir::cast<sdy::TensorShardingPerValueAttr>(unwrap(attr))
      .getShardings()
      .size();
}

MlirAttribute sdyTensorShardingPerValueAttrGetShardingsElem(MlirAttribute attr,
                                                            intptr_t pos) {
  return wrap(mlir::cast<sdy::TensorShardingPerValueAttr>(unwrap(attr))
                  .getShardings()[pos]);
}
