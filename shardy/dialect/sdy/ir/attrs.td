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

#ifndef SDY_ATTRS
#define SDY_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "shardy/dialect/sdy/ir/dialect.td"

// NOTE: we use `` in assemblyFormat to avoid whitespaces between literals and
// parameters.

// A list of axes that a ManualComputationOp is manual on. ManualComputationOp
// doesn't use an array of AxisRefAttr since all axes must be full/can't be
// split.
def Sdy_ManualAxes : ArrayOfAttr<Sdy_Dialect, "ManualAxes",
                                 "manual_axes", "StringAttr"> {
  let assemblyFormat = "`{` (`}`) : ($value^ `` `}`)?";
}

def Sdy_MeshAxis : AttrDef<Sdy_Dialect, "MeshAxis"> {
  let mnemonic = "mesh_axis";
  let summary = "Named axis in a mesh";
  let parameters = (ins
      StringRefParameter<"name">:$name,
      "int64_t":$size
  );
  let assemblyFormat = "`` $name `` `=` `` $size";
  let genVerifyDecl = 1;
}

def Sdy_Mesh : AttrDef<Sdy_Dialect, "Mesh"> {
  let mnemonic = "mesh";
  let summary = "Mesh of axes and a list of devices";
  let description = [{
    A mesh is a list of axes and an optional list of device IDs specifying the
    device ordering.

    If the list of axes is empty, the mesh has an implicit unnamed axis of
    size 1. In this case, if a device ID list is not provided, the implicit
    device ID list is [0]; if a device ID list is provided, it must
    contains a single integer of any non-negative value. We call this
    maximal-sharding case.

    For all non-maximal-sharding cases, if a device ID list is specified, the
    product of the axis sizes should match the number of devices. If a device ID
    list is not specified, the implicit device ID list is iota(product(axes)).
    For simplicity, we also disallow specifying a device ID list that is the
    same as iota(product(axes)); in this case, a device ID list shouldn't be
    specified.

  Here are some examples of meshes:

    - An empty mesh represents a placeholder mesh that can be replaced during
      propagation: <[]>
    - A mesh with an unnamed axis and an explicit device ID, which is typically
      used to represent maximal sharding: <[], device_ids=[3]>
    - A mesh with two axes and implicit device IDs iota(6): <["a"=2, "b"=3]>
    - A mesh with two axes and explicit device IDs specifying the device
      ordering: <["a"=3, "b"=2], device_ids=[0, 2, 4, 1, 3, 5]>
  }];
  let parameters = (ins
      OptionalArrayRefParameter<"MeshAxisAttr">:$axes,
      OptionalArrayRefParameter<"int64_t">: $device_ids
  );

  let assemblyFormat = [{
    `<` `[` (`]`):($axes^ `]`)?
      (`,` `device_ids` `` `=` `` `[` $device_ids^ `]` )? `>`
  }];

  let genVerifyDecl = 1;

  let builders = [
    AttrBuilder<(ins "mlir::ArrayRef<MeshAxisAttr>":$axes), [{
      return $_get($_ctxt, axes, /*device_id=*/std::nullopt);
    }]>,
    AttrBuilder<(ins "int64_t":$device_id), [{
      return $_get($_ctxt, /*axes=*/ArrayRef<MeshAxisAttr>(), ArrayRef<int64_t>(device_id));
    }]>,
  ];

  let extraClassDeclaration = [{
    // Returns true if this mesh has no axes or device ids.
    bool empty() const;

    // Returns true if this mesh has an axis with the given `axisName`.
    bool hasAxis(StringRef axisName) const;

    // Returns the size of the axis with the given `axisName`.
    //
    // Assumes the axis is present in the mesh.
    int64_t getAxisSize(StringRef axisName) const;

    // Returns the total size of the mesh across all axes, as in the total
    // number of devices.
    int64_t getTotalSize() const;

    // Returns whether this mesh is a maximal-sharding mesh with `deviceId`.
    //
    // A maximal-sharding mesh is a mesh with an empty axis list and a single
    // device ID.
    bool isMaximal(int64_t deviceId) const;

    // If this mesh is a maximal-sharding mesh, returns the maximal device ID,
    // otherwise, returns std::nullopt.
    //
    // A maximal-sharding mesh is a mesh with an empty axis list and a single
    // device ID.
    std::optional<int64_t> getMaximalDeviceId() const;

    // Returns a comparator that orders axis names w.r.t. their order in this
    // mesh.
    std::function<bool(StringRef lhs, StringRef rhs)> getAxisNameComparator()
    const;
  }];
}

def Sdy_SubAxisInfo : AttrDef<Sdy_Dialect, "SubAxisInfo"> {
  let mnemonic = "sub_axis_info";
  let summary = "Info about how this sub-axis is derived from the full axis";
  let description = [{
    When splitting a full axis into n sub-axes, the axis is reshaped into
    [k_1,...,k_n], and the ith sub-axis can be expressed by the product of all
    axis sizes to its left `m=prod(k_1,...,k_(i-1))` (aka pre-size) and size
    k_i. Therefore, the sub-axis-info attribute holds those two numbers and is
    denoted as follows: `(m)k` for pre-size m and size k.
  }];
  let parameters = (ins
      "int64_t":$pre_size,
      "int64_t":$size
  );
  let assemblyFormat = "`(` $pre_size `)` `` $size";

  let extraClassDeclaration = [{
    // Sub-axes of the same full axis are ordered by their pre-size, and then by
    // their size (overlap is only possible for two sub-axes that shard
    // different tensors), e.g. [1(2), 4(2), 4(4)].
    bool operator<(const SubAxisInfoAttr &rhs) const;

    // Returns the pre-size of the next sub-axis (that is minor to this
    // sub-axis), or the size of the full axis if this is the minor-most
    // sub-axis.
    //
    // The next pre-size is equal to `pre-size * size` of this sub-axis.
    int64_t getNextPreSize() const {
      return getPreSize() * getSize();
    }
  }];
}

def Sdy_AxisRef : AttrDef<Sdy_Dialect, "AxisRef"> {
  let mnemonic = "axis_ref";
  let summary = "Reference to either a full axis or a split sub-axis";
  let parameters = (ins
      StringRefParameter<"name">:$name,
      OptionalParameter<"SubAxisInfoAttr">:$sub_axis_info
  );
  let assemblyFormat = "`` $name (`` `:` `` $sub_axis_info^)?";

  let builders = [
    AttrBuilder<(ins "StringRef":$name), [{
      return $_get($_ctxt, name, /*sub_axis_info=*/nullptr);
    }]>,
    AttrBuilder<(ins "StringRef":$name, "int64_t":$pre_size, "int64_t":$size), [{
      return $_get($_ctxt, name, SubAxisInfoAttr::get($_ctxt, pre_size, size));
    }]>
  ];

  let extraClassDeclaration = [{
    // Returns a comparator that orders axis names w.r.t. their order in the
    // given `mesh`.
    static std::function<bool(AxisRefAttr lhs, AxisRefAttr rhs)>
    getMeshComparator(MeshAttr mesh);

    std::string toString() const;

    // Returns the size of this axis or sub-axis.
    int64_t getSize(MeshAttr mesh) const;

    // If this is a sub-axis, returns its pre-size, otherwise returns 1.
    int64_t getSubAxisPreSize() const;

    // Returns whether this axis or sub-axis contains `other`, i.e., this axis
    // or sub-axis is equal to `other` or can be split into multiple sub-axes
    // such that one of them is `other`.
    //
    // For example:
    //  "a", "a":(2)2       -> true
    //  "a":(1)8, "a":(1)4  -> true
    //  "a":(2)16, "a":(4)2 -> true
    //  "a", "a"            -> true
    //  "a":(2)2, "a":(2)2  -> true
    //  "a":(1)4, "a":(2)4  -> false
    //  "a":(2)4, "a":(1)2  -> false
    //  "a", "b":(1)2       -> false
    bool contains(AxisRefAttr other) const;

    // Returns whether this axis or sub-axis strictly contains `other`.
    // "a.strictlyContains(b)" is equivalent to "a.contains(b) && a != b".
    //
    // For example:
    //  "a", "a":(2)2       -> true
    //  "a":(1)8, "a":(1)4  -> true
    //  "a":(2)16, "a":(4)2 -> true
    //  "a", "a"            -> false
    //  "a":(2)2, "a":(2)2  -> false
    //  "a":(1)4, "a":(2)4  -> false
    //  "a":(2)4, "a":(1)2  -> false
    //  "a", "b":(1)2       -> false
    bool strictlyContains(AxisRefAttr other) const;

    // Returns whether this axis or sub-axis is a prefix of `other`, i.e.,
    // `other` is equal to this axis ref or can be split into two sub-axes such
    // that the major one is this sub-axis.
    //
    // For example:
    //  "a":(1)2, "a"      -> true
    //  "a":(2)2, "a":(2)4 -> true
    //  "a", "a"           -> true
    //  "a":(2)4, "a":(2)4 -> true
    //  "a":(1)4, "a":(1)2 -> false
    //  "a":(1)4, "a":(2)8 -> false
    //  "a":(1)2, "b"      -> false
    bool prefixOf(AxisRefAttr other) const;

    // Returns whether this axis or sub-axis is a strict prefix of `other`.
    // "a.strictPrefixOf(b)" is equivalent to "a.prefixOf(b) && a != b".
    //
    // For example:
    //  "a":(1)2, "a"      -> true
    //  "a":(2)2, "a":(2)4 -> true
    //  "a", "a"           -> false
    //  "a":(2)4, "a":(2)4 -> false
    //  "a":(1)4, "a":(1)2 -> false
    //  "a":(1)4, "a":(2)8 -> false
    //  "a":(1)2, "b"      -> false
    bool strictPrefixOf(AxisRefAttr other) const;

    // Returns whether this axis or sub-axis overlaps with `other`, i.e., they
    // are equal or there is a sub-axis that is contained in both axis refs.
    //
    // For example:
    //  "a", "a":(2)2      -> true
    //  "a":(2)2, "a":(2)2 -> true
    //  "a":(1)4, "a":(2)4 -> true
    //  "a":(2)4, "a":(1)4 -> true
    //  "a":(1)4, "a":(1)2 -> true
    //  "a":(2)8, "a":(4)2 -> true
    //  "a":(1)4, "a":(4)2 -> false
    //  "a":(1)2, "a":(4)2 -> false
    //  "a":(1)4, "b":(2)4 -> false
    bool overlaps(AxisRefAttr other) const;

    // If there is no overlap between this and other axes, return this axis.
    // Otherwise, return the largest prefix of this axis by removing the
    // overlapping suffix with `other`. Return `std::nullopt` if the prefix does
    // not exist.
    //
    // For example:
    //  "a", "a":(2)2      -> "a":(1)2
    //  "a":(2)2, "a":(2)2 -> std::nullopt
    //  "a":(1)4, "a":(2)4 -> "a":(1)2
    //  "a":(2)4, "a":(1)4 -> std::nullopt
    //  "a":(1)4, "a":(1)2 -> std::nullopt
    //  "a":(2)8, "a":(4)2 -> "a":(2)2
    //  "a":(1)4, "a":(4)2 -> "a":(1)4
    //  "a":(1)2, "a":(4)2 -> "a":(1)2
    //  "a":(1)4, "b":(2)4 -> "a":(1)4
    std::optional<AxisRefAttr> getPrefixWithoutOverlap(AxisRefAttr other) const;

    // Returns the greatest common prefix of this and `other`. If the two
    // axes do not have common prefix, return `std::nullopt`.
    //
    // For example:
    //  "a", "a" -> "a"
    //  "a", "b" -> std::nullopt
    //  "a":(1)2, "a":(2)4 -> std::nullopt
    //  "a":(1)4, "a" -> "a":(1)4
    //  "a", "a":(1)4 -> "a":(1)4
    //  "a":(1)2, "a":(1)4 -> "a":(1)2
    //  "a":(2)8, "a":(2)4 -> "a":(2)4
    std::optional<AxisRefAttr> getGreatestCommonPrefix(AxisRefAttr other) const;

    // Returns whether this axis-ref can be merged with `other`, i.e., they are
    // consecutive sub-axes of the same full axis and this sub-axis is major to
    // `other`.
    //
    // For example:
    //  "a":(2)4, "a":(8)2 -> true
    //  "b":(1)2, "b":(2)4 -> true
    //  "c":(1)2, "c":(4)2 -> false
    //  "d":(2)4, "d":(1)2 -> false
    bool canMerge(AxisRefAttr other) const;

    // Merges this axis-ref with the `other`, assuming `canMerge(other)` is
    // true, i.e., they are consecutive sub-axes of the same full axis and this
    // sub-axis is major to `other`.
    //
    // The mesh is needed for the size of the full axis (see 2nd example below).
    //
    // For example:
    //  "a":(2)4, "a":(8)2 ~> "a":(2)8
    //  "b":(1)2, "b":(2)4 ~> "b"
    AxisRefAttr merge(AxisRefAttr other, MeshAttr mesh) const;
  }];
}

def Sdy_AxisRefs : OptionalArrayRefParameter<"AxisRefAttr", "list of axis refs">;

def Sdy_DimensionSharding : AttrDef<Sdy_Dialect, "DimensionSharding"> {
  let mnemonic = "dimension_sharding";
  let summary = "Dimension sharding";
  let description = [{
    List of axis names to shard a tensor dimension on from major to minor, a
    boolean indicating whether the dimension can be further sharded, and an
    optional integer denoting the priority of this dimension sharding, which
    will respected during sharding propagation. Priorities originate from user
    sharding annotations and a lower value denotes a higher priority. The
    highest priority is assumed when the priority is missing in the annotation.
  }];

  let parameters = (ins
    Sdy_AxisRefs:$axes,
    "bool":$is_closed,
    OptionalParameter<"std::optional<int64_t>">:$priority
  );

  let builders = [
    AttrBuilder<(ins "ArrayRef<AxisRefAttr>":$axes,
                     "bool":$is_closed), [{
      return $_get($_ctxt, axes, is_closed, /*priority=*/std::nullopt);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ArrayRef<AxisRefAttr>::iterator axis_begin() const {
      return getAxes().begin();
    }
    ArrayRef<AxisRefAttr>::iterator axis_end() const {
      return getAxes().end();
    }

    // Returns true if this dimension sharding has no axes.
    bool emptyAxes() const { return getAxes().empty(); }

    // Shards this dimension further along `axisName`.
    //
    // Assumes it is it not closed or already sharded on `axisName`.
    //
    // Attributes are immutable, so we can't update the sharding in place and
    // must return a new instance.
    DimensionShardingAttr getSharded(StringRef axisName) const;

    // Returns the sharded size of this dimension,
    // i.e., the product of sharding axis sizes.
    int64_t getShardedSize(MeshAttr mesh) const;

    // Drops the first `N` sharding axes, and keeps `M` sharding axes.
    DimensionShardingAttr sliceShardingAxes(size_t N, size_t M) const;

    // Drops the first `N` sharding axes.
    DimensionShardingAttr dropFrontShardingAxes(size_t N) const;

    // Takes the first `N` sharding axes.
    DimensionShardingAttr takeFrontShardingAxes(size_t N) const;

    // Drops the priority of this dimension sharding, if present.
    DimensionShardingAttr dropPriority() const;

    // Returns the priority of this dimension sharding, if present, or the
    // default priority otherwise.
    int64_t getPriorityOrDefault() const;
  }];
}

// Either a `MeshAttr` or a symbol name, referencing a corresponding `MeshOp`
// symbol.
def Sdy_MeshOrRef : AnyAttrOf<[Sdy_Mesh, FlatSymbolRefAttr],
                              "mesh attr or flat mesh symbol reference attr"> {
  string cppType = "::mlir::Attribute";
}

def Sdy_TensorSharding : AttrDef<Sdy_Dialect, "TensorSharding"> {
  let mnemonic = "sharding";
  let summary = "Tensor sharding";
  let description = [{
    A tensor sharding is bound to a specific mesh, and can only reference axis
    names from that mesh. The dimension shardings tell us for each dimension of
    the tensor, along which axes (or sub-axes) it is sharded from major to
    minor. All other axes that don’t shard a dimension are either implicitly or
    explicitly (if they appear in the list of replicated axes) replicated.

    The mesh this sharding is bound to can either be specified by a symbol
    name, referencing a corresponding `MeshOp` symbol, or an inlined `MeshAttr`.
  }];
  let parameters = (ins
      Sdy_MeshOrRef:$mesh_or_ref,
      OptionalArrayRefParameter<"DimensionShardingAttr">:$dim_shardings,
      Sdy_AxisRefs:$replicated_axes
  );
  let assemblyFormat = [{
    `<` custom<MeshOrRef>($mesh_or_ref) `,` `[` (`]`):($dim_shardings^ `]`)? ``
        (`,` `replicated` `` `=` `` `{` $replicated_axes^ `}`)? `>`
  }];

  let builders = [
    AttrBuilder<(ins "StringAttr":$mesh_name,
                     "ArrayRef<DimensionShardingAttr>":$dim_shardings,
                     "ArrayRef<AxisRefAttr>":$replicated_axes), [{
      return $_get($_ctxt, FlatSymbolRefAttr::get(mesh_name),
                   dim_shardings, replicated_axes);
    }]>,
    AttrBuilder<(ins "StringRef":$mesh_name,
                     "ArrayRef<DimensionShardingAttr>":$dim_shardings,
                     "ArrayRef<AxisRefAttr>":$replicated_axes), [{
      return $_get($_ctxt, FlatSymbolRefAttr::get($_ctxt, mesh_name),
                   dim_shardings, replicated_axes);
    }]>
  ];

  let extraClassDeclaration = [{
    int64_t getRank() const {
      return getDimShardings().size();
    }

    DimensionShardingAttr getDimSharding(int64_t dim) const {
      return getDimShardings()[dim];
    }

    bool isClosed(int64_t dim) const {
      return getDimSharding(dim).getIsClosed();
    }

    bool isFullyClosed() const {
      return llvm::all_of(getDimShardings(),
                      [](const DimensionShardingAttr dimSharding) {
                         return dimSharding.getIsClosed();
                      });
    }

    bool isFullyReplicated() const {
      return llvm::all_of(getDimShardings(),
                      [](const DimensionShardingAttr dimSharding) {
                         return dimSharding.emptyAxes();
                      });
    }

    // Returns the mesh `FlatSymbolRefAttr` this sharding references, assuming
    // it doesn't have an inlined `MeshAttr`.
    FlatSymbolRefAttr getMeshSymName() const {
      return mlir::cast<FlatSymbolRefAttr>(getMeshOrRef());
    }

    // Returns the mesh name this sharding references, assuming it doesn't have
    // an inlined `MeshAttr`.
    StringRef getMeshName() const {
      return getMeshSymName().getValue();
    }

    // If this sharding has an inlined `MeshAttr`, returns it, otherwise looks
    // up the mesh symbol with the referenced name in `symbolTable`, and returns
    // its `MeshAttr` if it exists in the table, or nullptr otherwise.
    MeshAttr getMesh(const SymbolTable& symbolTable) const;

    // If this sharding has an inlined `MeshAttr`, returns it, otherwise looks
    // up the mesh symbol with the referenced name in the symbol table of the
    // enclosing module of `op`, and returns its `MeshAttr` if it exists in the
    // table, or nullptr otherwise.
    MeshAttr getMesh(Operation* op) const;

    // Returns true if all dimension shardings are empty and there are no
    // replicated axes.
    bool emptyAxes() const;

    // Like `llvm::any_of` but checks the predicate against all dimension
    // sharding and replicated `AxisRefAttr`s.
    bool anyOfAxisRef(std::function<bool(AxisRefAttr)> predicate) const;

    // Like `llvm::for_each` but applies the `callback` against all dimension
    // sharding and replicated `AxisRefAttr`s.
    void forEachAxisRef(std::function<void(AxisRefAttr)> callback) const;

    // Returns true if `axisName` or a sub-axis of it is used to shard any
    // dimension or is replicated.
    bool isBound(StringRef axisName) const;

    // Returns true if dimension `dim` can be further sharded on the full
    // `axisName`.
    bool canShard(int64_t dim, StringRef axisName) const;

    // Returns true if the tensor can be replicated on the full `axisName`.
    bool canReplicate(StringRef axisName) const;

    // Closes sharding dimensions at the specified dimension indices.
    TensorShardingAttr closeShardingDims(ArrayRef<int64_t> dimIndices) const;

    // Opens sharding dimensions at the specified dimension indices.
    TensorShardingAttr openShardingDims(ArrayRef<int64_t> dimIndices) const;

    // Sets the sharding of dimension `dim`.
    //
    // Assumes `dim < getRank()`.
    //
    // Attributes are immutable, so we can't update the sharding in place and
    // must return a new instance.
    TensorShardingAttr replaceDimSharding(
        int64_t dim, DimensionShardingAttr sharding) const;

    // Shards dimension `dim` further along `axisName`.
    //
    // Assumes `canShard(dim, axisName)` is true.
    //
    // Attributes are immutable, so we can't update the sharding in place and
    // must return a new instance.
    TensorShardingAttr getSharded(int64_t dim, StringRef axisName) const;

    // Replicates the tensor along `axisName`.
    //
    // Assumes `canReplicate(axisName)` is true. The `mesh` is needed to keep
    // the replicated axes sorted.
    //
    // Attributes are immutable, so we can't update the sharding in place and
    // must return a new instance.
    TensorShardingAttr getReplicated(StringRef axisName, MeshAttr mesh) const;


    // Verifies that this `TensorShardingAttr` is valid w.r.t the given
    // tensor type and mesh.
    //
    // If `type` isn't a `ShapedType`, the sharding must have rank 0
    // and no replicated axes. Otherwise, the `ShapedType` must have a static
    // shape.

    //
    // If `checkDivisibility` is true, verifies that each dimension size
    // is divisible by its sharded size.
    mlir::LogicalResult verifyForType(
        Type type, MeshAttr mesh,
        std::function<InFlightDiagnostic(StringRef)> emitError,
        bool checkDivisibility = true);

    // Builds a `TensorShardingAttr` with all dim shardings and replicated axes
    // being marked closed (cannot be further replicated/sharded).
    static TensorShardingAttr getFullyClosed(
        MLIRContext* context, int64_t rank, StringRef meshName);

    // Builds a fully open `TensorShardingAttr` with all dim shardings and
    // replicated axes being marked closed and matching `sharding` in
    // `mesh_sym_name` and rank.
    static TensorShardingAttr getFullyClosedLike(TensorShardingAttr sharding);

    // Builds a `TensorShardingAttr` with all dim shardings and replicated axes
    // being marked open (can be further replicated/sharded).
    static TensorShardingAttr getFullyOpen(
        MLIRContext* context, int64_t rank, StringRef meshName);

    // Builds a fully open `TensorShardingAttr` matching `sharding` in
    // `mesh_sym_name` and rank.
    static TensorShardingAttr getFullyOpenLike(TensorShardingAttr sharding);
  }];
}

def Sdy_TensorShardingPerValue : AttrDef<Sdy_Dialect, "TensorShardingPerValue"> {
  let mnemonic = "sharding_per_value";
  let summary = "Tensor sharding per operand/result of an op";
  let parameters = (ins
      OptionalArrayRefParameter<"TensorShardingAttr">:$shardings
  );
  let assemblyFormat = "`<` `[` (`]`):($shardings^ `]`)? `>`";

  let extraClassDeclaration = [{
    // Builds a `TensorSharding` for each type in `types`, with all dimension
    // shardings marked open (can be further replicated/sharded).
    static TensorShardingPerValueAttr getFullyOpen(
        MLIRContext* context, TypeRange types, StringRef meshName);

    // Builds an open `TensorSharding` for each type in `types`, but
    // with the sharding at `index` replaced with `sharding`.
    static TensorShardingPerValueAttr getOpenWithShardingAtIndex(
        MLIRContext* context, TypeRange types, int64_t index,
        TensorShardingAttr sharding);

    // Returns whether there are no values.
    bool empty() const { return getShardings().empty(); }

    // Returns the number of values.
    int64_t size() const { return getShardings().size(); }

    // Returns the sharding of a value at `operandIndex`.
    //
    // Assumes `operandIndex < size()`.
    TensorShardingAttr getSharding(int64_t operandIndex) const {
      assert(operandIndex < size());
      return getShardings()[operandIndex];
    }

    // Sets the sharding of a value at `index`.
    //
    // Assumes `index < size()`.
    //
    // Attributes are immutable, so we can't update the sharding in place and
    // must return a new instance.
    TensorShardingPerValueAttr replaceValueSharding(
        int64_t index, TensorShardingAttr sharding) const;
  }];
}

def Sdy_DimMapping : AttrDef<Sdy_Dialect, "DimMapping"> {
  let mnemonic = "dim_mapping";
  let summary = "List of factor indices for a dimension";
  let description = [{
    All factor indices must be in the range [0, num_factors) and an empty list
    indicates that this is a null mapping (this is parsed/printed with `*`),
    i.e. the dimension isn't mapped to any factors.
   }];
  let parameters = (ins
    OptionalArrayRefParameter<"int64_t">:$factor_indices
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Returns whether the given `factorIndex` is the minor-most factor.
    bool isMinorMost(int64_t factorIndex) const {
      return !getFactorIndices().empty() &&
              getFactorIndices().back() == factorIndex;
    }
  }];
}

def Sdy_TensorMapping : AttrDef<Sdy_Dialect, "TensorMapping"> {
  let mnemonic = "tensor_mapping";
  let summary = "Factor mappings for each dimension of a tensor.";
  let parameters = (ins
      OptionalArrayRefParameter<"DimMappingAttr">:$dim_mappings
  );

  let assemblyFormat = "`` `[` (`]`):($dim_mappings^ `]`)? ``";

  let extraClassDeclaration = [{
    int64_t getRank() const { return getDimMappings().size(); }
  }];
}


def Sdy_OpShardingRule : AttrDef<Sdy_Dialect, "OpShardingRule"> {
  let mnemonic = "op_sharding_rule";
  let summary = "Specifies how an operation can be partitioned.";
  let description = [{
    A sharding rule specifies how an operation can be partitioned according to
    various properties on the op - any attributes, the shape of operands,
    the shape of the results, etc. For example:

    ```
    %0 = stablehlo.add %arg0, %arg1 {
        sdy.sharding_rule = #sdy.op_sharding_rule<
            ([i, j],[i, j])->([i, j])
            {i=8, j=8}>
    } : tensor<8x8xf32>
    ```

    ```
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] {
      sdy.sharding_rule = #sdy.op_sharding_rule<
          ([i, k],[k, j])->([i, j])
          {i=8, j=16, k=8}>
    }: (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
    ```

    Note that we allow factors with size 1 even though they cannot be sharded,
    this is mainly for completeness as many ops such as pointwise ops have size
    one dimensions that correspond across operands and results.

    `is_custom_rule` describes whether this is a rule defined by a user for a
    `stablehlo.custom_call` op. The partitioner doesn't know how to partition
    these ops, so a user must tell it how. When it is a custom rule, then the
    rule is always preserved/never removed. `is_custom_rule` can only be true
    for `stablehlo.custom_call` ops.
  }];

  let parameters = (ins
      OptionalArrayRefParameter<"int64_t">:$factor_sizes,
      OptionalArrayRefParameter<"TensorMappingAttr">:$operand_mappings,
      OptionalArrayRefParameter<"TensorMappingAttr">:$result_mappings,
      DefaultValuedParameter<"bool", "false">:$is_custom_rule
  );

  let assemblyFormat = [{
    `<`
    `(`$operand_mappings`)`
    `` `->` ``
    `(`$result_mappings`)` ``
    custom<FactorSizes>($factor_sizes)
    ``custom<IsCustomRule>($is_custom_rule)
    `>`
  }];

  let builders = [
    AttrBuilder<(ins "ArrayRef<int64_t>":$factor_sizes,
                     "ArrayRef<TensorMappingAttr>":$operand_mappings,
                     "ArrayRef<TensorMappingAttr>":$result_mappings), [{
      return $_get($_ctxt, factor_sizes, operand_mappings, result_mappings,
                   /*is_custom_rule=*/false);
    }]>
  ];

  let extraClassDeclaration = [{
    int64_t getNumFactors() const { return getFactorSizes().size(); }
    int64_t getNumOperands() const { return getOperandMappings().size(); }
    int64_t getNumResults() const { return getResultMappings().size(); }

    int64_t getFactorSize(int64_t factorIndex) const {
      return getFactorSizes()[factorIndex];
    }
    TensorMappingAttr getOperandMapping(int64_t operandNum) const {
      return getOperandMappings()[operandNum];
    }
    TensorMappingAttr getResultMapping(int64_t resultNum) const {
      return getResultMappings()[resultNum];
    }

    bool isCustom() const { return getIsCustomRule(); }
  }];
}

#endif  // SDY_ATTRS
