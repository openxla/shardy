/* Copyright 2025 The Shardy Authors.

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
//===----------------------------------------------------------------------===//
//
// This is the SDY bytecode reader/writer definition file.
//
//===----------------------------------------------------------------------===//

#ifndef SDY_BYTECODE
#define SDY_BYTECODE

include "mlir/IR/BytecodeBase.td"
include "mlir/IR/BuiltinDialectBytecode.td"

// -----------------------------------------------------------------------------
// Helper classes
// -----------------------------------------------------------------------------

def StringParam : WithGetter<"$_attrType.$_getMember()", String>;
class OptionalAttr<AttributeKind attr> :
  WithParser<"succeeded($_reader.readOptionalAttribute($_var))",
  WithPrinter<"$_writer.writeOptionalAttribute($_getter)", attr>>;
def OptionalVarInt :
  WithPrinter<"writeOptionalVarInt($_writer, $_getter)",
  WithParser<"succeeded(readOptionalVarInt($_reader, $_var))",
  WithType<"std::optional<uint64_t>">>>;

// -----------------------------------------------------------------------------
// Sdy dialect attributes
// -----------------------------------------------------------------------------

//===----------------------------------------------------------------------===//
// ManualAxesAttr - assigned index 0
//===----------------------------------------------------------------------===//
def Sdy_ManualAxesAttrBc : DialectAttribute<(attr
  Array<StringAttr>:$value
)> {
  let cType = "ManualAxesAttr";
}

//===----------------------------------------------------------------------===//
// MeshAxisAttr - assigned index 1
//===----------------------------------------------------------------------===//
def Sdy_MeshAxisAttrBc : DialectAttribute<(attr
  StringParam:$name,
  SignedVarInt:$size
)> {
  let cType = "MeshAxisAttr";
}

//===----------------------------------------------------------------------===//
// MeshAttr - assigned index 2
//===----------------------------------------------------------------------===//
def Sdy_MeshAttrBc : DialectAttribute<(attr
  Array<Sdy_MeshAxisAttrBc>:$axes,
  Array<SignedVarIntList>:$device_ids
)> {
  let cType = "MeshAttr";
}

// ===----------------------------------------------------------------------===//
// SubAxisInfoAttr - assigned index 3
//===----------------------------------------------------------------------===//
def Sdy_SubAxisInfoAttrBc : DialectAttribute<(attr
  SignedVarInt:$pre_size,
  SignedVarInt:$size
)> {
  let cType = "SubAxisInfoAttr";
}


//===----------------------------------------------------------------------===//
// AxisRefAttr
//===----------------------------------------------------------------------===//
// assigned index 4
def Sdy_AxisRefAttrBc: DialectAttribute<(attr
  StringParam:$name,
  OptionalAttr<Sdy_SubAxisInfoAttrBc>:$sub_axis_info
)> {
  let cType = "AxisRefAttr";
}


//===----------------------------------------------------------------------===//
// DimensionShardingAttr
//===----------------------------------------------------------------------===//
// assigned index 5
def Sdy_DimensionShardingAttrBc: DialectAttribute<(attr
  Array<Sdy_AxisRefAttrBc>:$axes,
  Bool:$is_closed,
  OptionalVarInt:$priority
)> {
  let cType = "DimensionShardingAttr";
}


//===----------------------------------------------------------------------===//
// TensorShardingAttr V1 - assigned index 6
//===----------------------------------------------------------------------===//
def Sdy_TensorShardingAttrV1Bc : DialectAttribute<(attr
  Attribute:$mesh_or_ref,
  Array<Sdy_DimensionShardingAttrBc>:$dim_shardings,
  Array<Sdy_AxisRefAttrBc>:$replicated_axes
)> {
  let cType = "TensorShardingAttr";
  let cBuilder = [{
    $_resultType::get(context, mesh_or_ref, dim_shardings, replicated_axes,
      /*unreduced_axes=*/{})
  }];
  let printerPredicate = "$_val.getUnreducedAxes().empty()";
}

//===----------------------------------------------------------------------===//
// TensorShardingAttr V2 - assigned index 15
//===----------------------------------------------------------------------===//
def Sdy_TensorShardingAttrV2Bc : DialectAttribute<(attr
  Attribute:$mesh_or_ref,
  Array<Sdy_DimensionShardingAttrBc>:$dim_shardings,
  Array<Sdy_AxisRefAttrBc>:$replicated_axes,
  Array<Sdy_AxisRefAttrBc>:$unreduced_axes
)> {
  let cType = "TensorShardingAttr";
  let printerPredicate = "!$_val.getUnreducedAxes().empty()";
}

//===----------------------------------------------------------------------===//
// TensorShardingPerValueAttr V1 - assigned index 7
//===----------------------------------------------------------------------===//
def Sdy_TensorShardingPerValueAttrV1Bc : DialectAttribute<(attr
  Array<Sdy_TensorShardingAttrV1Bc>:$shardings
)> {
  let cType = "TensorShardingPerValueAttr";
  let printerPredicate = "!$_val.anyShardingHasUnreducedAxes()";
}

//===----------------------------------------------------------------------===//
// TensorShardingPerValueAttr V2 - assigned index 16
//===----------------------------------------------------------------------===//
def Sdy_TensorShardingPerValueAttrV2Bc : DialectAttribute<(attr
  Array<Sdy_TensorShardingAttrV2Bc>:$shardings
)> {
  let cType = "TensorShardingPerValueAttr";
  let printerPredicate = "$_val.anyShardingHasUnreducedAxes()";
}

//===----------------------------------------------------------------------===//
// DimMappingAttr - assigned index 8
//===----------------------------------------------------------------------===//
def Sdy_DimMappingAttrBc : DialectAttribute<(attr
  Array<SignedVarIntList>:$factor_indices
)> {
  let cType = "DimMappingAttr";
}

//===----------------------------------------------------------------------===//
// TensorMappingAttr - assigned index 9
//===----------------------------------------------------------------------===//
def Sdy_TensorMappingAttrBc : DialectAttribute<(attr
  Array<Sdy_DimMappingAttrBc>:$dim_mappings
)> {
  let cType = "TensorMappingAttr";
}

//===----------------------------------------------------------------------===//
// OpShardingRuleAttr - assigned index 10
//===----------------------------------------------------------------------===//
def Sdy_OpShardingRuleAttrBc : DialectAttribute<(attr
  Array<SignedVarIntList>:$factor_sizes,
  Array<Sdy_TensorMappingAttrBc>:$operand_mappings,
  Array<Sdy_TensorMappingAttrBc>:$result_mappings,
  Array<SignedVarIntList>:$reduction_factors,
  Array<SignedVarIntList>:$need_replication_factors,
  Array<SignedVarIntList>:$permutation_factors,
  Array<SignedVarIntList>:$blocked_propagation_factors,
  Bool:$is_custom_rule
)> {
  let cType = "OpShardingRuleAttr";
}

//===----------------------------------------------------------------------===//
// AxisRefListAttr - assigned index 11
//===----------------------------------------------------------------------===//
def Sdy_AxisRefListAttrBc : DialectAttribute<(attr
  Array<Sdy_AxisRefAttrBc>:$value
)> {
  let cType = "AxisRefListAttr";
}

//===----------------------------------------------------------------------===//
// ListOfAxisRefListsAttr - assigned index 12
//===----------------------------------------------------------------------===//
def Sdy_ListOfAxisRefListsAttrBc: DialectAttribute<(attr
  Array<Sdy_AxisRefListAttrBc>:$value
)> {
  let cType = "ListOfAxisRefListsAttr";
}

//===----------------------------------------------------------------------===//
// AllToAllParamAttr - assigned index 13
//===----------------------------------------------------------------------===//
def Sdy_AllToAllParamAttrBc: DialectAttribute<(attr
  Array<Sdy_AxisRefAttrBc>:$axes,
  SignedVarInt:$src_dim,
  SignedVarInt:$tgt_dim
)> {
  let cType = "AllToAllParamAttr";
}

//===----------------------------------------------------------------------===//
// AllToAllParamListAttr - assigned index 14
//===----------------------------------------------------------------------===//
def Sdy_AllToAllParamListAttrBc: DialectAttribute<(attr
  Array<Sdy_AllToAllParamAttrBc>:$value
)> {
  let cType = "AllToAllParamListAttr";
}

// This enum contains marker codes used to indicate which attribute is
// currently being decoded, and how it should be decoded. The order of these
// codes should generally be unchanged, as any changes will inevitably break
// compatibility with older bytecode.
def SdyDialectTypes : DialectAttributes<"sdy"> {
  // Take care when chaging this list.
  // If needed to remove use ReservedOrDead
  let elems = [
    /*index  0*/ Sdy_ManualAxesAttrBc,
    /*index  1*/ Sdy_MeshAxisAttrBc,
    /*index  2*/ Sdy_MeshAttrBc,
    /*index  3*/ Sdy_SubAxisInfoAttrBc,
    /*index  4*/ Sdy_AxisRefAttrBc,
    /*index  5*/ Sdy_DimensionShardingAttrBc,
    /*index  6*/ Sdy_TensorShardingAttrV1Bc,
    /*index  7*/ Sdy_TensorShardingPerValueAttrV1Bc,
    /*index  8*/ Sdy_DimMappingAttrBc,
    /*index  9*/ Sdy_TensorMappingAttrBc,
    /*index 10*/ Sdy_OpShardingRuleAttrBc,
    /*index 11*/ Sdy_AxisRefListAttrBc,
    /*index 12*/ Sdy_ListOfAxisRefListsAttrBc,
    /*index 13*/ Sdy_AllToAllParamAttrBc,
    /*index 14*/ Sdy_AllToAllParamListAttrBc,
    /*index 15*/ Sdy_TensorShardingAttrV2Bc,
    /*index 16*/ Sdy_TensorShardingPerValueAttrV2Bc,
  ];
}
#endif // SDY_BYTECODE
