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

// Verifies optimality of `-sdy-reshard-to-collectives` via shortest-path
// search.
//
// For a small mesh, every sharding of a tensor is a node of a graph, and every
// collective the pass can emit (all-slice, all-gather, single-parameter
// all-to-all, collective-permute) is an edge weighted by the number of
// elements each device receives. Dijkstra gives the cheapest sequence of
// collectives between any two shardings, which we compare against the
// sequence the pass actually emits, for all pairs of shardings.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"  // IWYU pragma: keep
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include <gtest/gtest.h>

namespace mlir {
namespace sdy {
namespace {

struct MeshAxis {
  std::string name;
  int64_t size;
};

// A mesh axis or sub-axis `axes[axis]:(preSize)size` (`size >= 2`,
// `axes[axis].size % (preSize * size) == 0`).
struct SubAxis {
  int axis;
  int64_t preSize;
  int64_t size;

  bool operator==(const SubAxis& other) const {
    return std::tie(axis, preSize, size) ==
           std::tie(other.axis, other.preSize, other.size);
  }
};

// Canonical merged sub-axes sharding each tensor dimension, major to minor.
using State = std::vector<std::vector<SubAxis>>;

enum class MoveKind { kAllSlice, kAllGather, kAllToAll, kCollectivePermute };

StringRef moveName(MoveKind kind) {
  switch (kind) {
    case MoveKind::kAllSlice:
      return "all_slice";
    case MoveKind::kAllGather:
      return "all_gather";
    case MoveKind::kAllToAll:
      return "all_to_all";
    case MoveKind::kCollectivePermute:
      return "collective_permute";
  }
  llvm_unreachable("unknown MoveKind");
}

// Cost of a sequence of collectives: elements received per device, then the
// number of collectives.
struct Cost {
  int64_t elems = 0;
  int64_t ops = 0;

  bool operator<(const Cost& other) const {
    return std::tie(elems, ops) < std::tie(other.elems, other.ops);
  }
  bool operator==(const Cost& other) const {
    return elems == other.elems && ops == other.ops;
  }
  Cost operator+(const Cost& other) const {
    return {elems + other.elems, ops + other.ops};
  }
};

struct Edge {
  int to;
  Cost cost;
  MoveKind kind;
};

struct Step {
  MoveKind kind;
  int state;
};

// Returns all distinct prime-factor decompositions of `axes[axisIndex]` (of
// size `n`), each as an ordered list of prime sub-axis atoms from major to
// minor. Prime powers p^k have one ordering; composite axes like 6 or 36 have
// multiple orderings (e.g. :(1)2, :(2)3 and :(1)3, :(3)2).
std::vector<std::vector<SubAxis>> primeFactorChains(int axisIndex, int64_t n) {
  std::vector<int64_t> primes;
  int64_t rem = n;
  for (int64_t p = 2; p * p <= rem; ++p) {
    while (rem % p == 0) {
      primes.push_back(p);
      rem /= p;
    }
  }
  if (rem > 1) {
    primes.push_back(rem);
  }
  std::vector<std::vector<SubAxis>> chains;
  do {
    std::vector<SubAxis>& chain = chains.emplace_back();
    chain.reserve(primes.size());
    int64_t pre = 1;
    for (int64_t p : primes) {
      chain.push_back({axisIndex, pre, p});
      pre *= p;
    }
  } while (std::next_permutation(primes.begin(), primes.end()));
  return chains;
}

void appendSubAxis(std::vector<SubAxis>& dimAxes, SubAxis sub) {
  if (!dimAxes.empty() && dimAxes.back().axis == sub.axis &&
      dimAxes.back().preSize * dimAxes.back().size == sub.preSize) {
    dimAxes.back().size *= sub.size;
  } else {
    dimAxes.push_back(sub);
  }
}

bool popSuffixSubAxes(std::vector<SubAxis>& dimAxes,
                      llvm::ArrayRef<SubAxis> suffix) {
  for (auto it = suffix.rbegin(); it != suffix.rend(); ++it) {
    if (dimAxes.empty() || dimAxes.back().axis != it->axis ||
        dimAxes.back().preSize * dimAxes.back().size !=
            it->preSize * it->size) {
      return false;
    }
    if (dimAxes.back().size == it->size) {
      dimAxes.pop_back();
    } else {
      if (dimAxes.back().size < it->size ||
          it->preSize % dimAxes.back().preSize != 0) {
        return false;
      }
      dimAxes.back().size = it->preSize / dimAxes.back().preSize;
    }
  }
  return true;
}

bool subAxesCanCoexist(SubAxis a, SubAxis b) {
  if (a.axis != b.axis) {
    return true;
  }
  int64_t aNext = a.preSize * a.size;
  int64_t bNext = b.preSize * b.size;
  auto [minPre, maxPre] = std::minmax(a.preSize, b.preSize);
  auto [minNext, maxNext] = std::minmax(aNext, bNext);
  if (minNext > maxPre) {
    return minNext % maxPre == 0 && maxPre % minPre == 0 &&
           maxNext % minNext == 0;
  }
  return maxPre % minNext == 0;
}

bool statesCanCoexist(const State& a, const State& b) {
  for (const std::vector<SubAxis>& dimA : a) {
    for (SubAxis x : dimA) {
      for (const std::vector<SubAxis>& dimB : b) {
        for (SubAxis y : dimB) {
          if (!subAxesCanCoexist(x, y)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// State graph of all shardings of a tensor on a mesh, across all prime-factor
// decompositions of every mesh axis.
//
// Precondition: each `shape[d]` must be positive. Local shard sizes use
// divisible padding (`alignTo(shape[d], dimShards)`), and sharded-to-sharded
// collective edges preserve padded dimension sizes.
class ReshardGraph {
 public:
  struct Node {
    State state;
    std::vector<int64_t> dimShards;
    std::vector<int64_t> paddedShape;
    int64_t localElems = 0;
    std::vector<Edge> edges;
  };

  ReshardGraph(std::vector<MeshAxis> axes, std::vector<int64_t> shape)
      : axes_(std::move(axes)), shape_(std::move(shape)) {
    assert(llvm::all_of(shape_, [](int64_t d) { return d > 0; }) &&
           "each tensor dimension must be positive");
    numDevices_ = 1;
    std::vector<std::vector<std::vector<SubAxis>>> chainsPerAxis;
    chainsPerAxis.reserve(axes_.size());
    for (auto [axisIndex, axis] : llvm::enumerate(axes_)) {
      assert(axis.size > 0 && "mesh axis size must be positive");
      numDevices_ *= axis.size;
      chainsPerAxis.push_back(
          primeFactorChains(static_cast<int>(axisIndex), axis.size));
    }
    std::vector<SubAxis> curAtoms;
    forEachAtomChain(chainsPerAxis, 0, curAtoms,
                     [&](llvm::ArrayRef<SubAxis> atoms) { buildChain(atoms); });
    for (Node& node : nodes_) {
      llvm::sort(node.edges, [](const Edge& a, const Edge& b) {
        return std::tie(a.to, a.kind, a.cost) < std::tie(b.to, b.kind, b.cost);
      });
      node.edges.erase(llvm::unique(node.edges,
                                    [](const Edge& a, const Edge& b) {
                                      return a.to == b.to && a.kind == b.kind;
                                    }),
                       node.edges.end());
    }
  }

  int numStates() const { return nodes_.size(); }
  int rank() const { return shape_.size(); }
  const Node& node(int i) const { return nodes_[i]; }
  const State& state(int i) const { return nodes_[i].state; }
  const std::vector<MeshAxis>& axes() const { return axes_; }
  const std::vector<int64_t>& shape() const { return shape_; }

  std::optional<int> indexOf(const State& s) const {
    auto it = index_.find(key(s));
    if (it == index_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  // Dijkstra from `src`; fills `parent` with the last step of each best path.
  std::vector<Cost> shortestFrom(int src, std::vector<Step>* parent) const {
    constexpr int64_t kInf = std::numeric_limits<int64_t>::max();
    std::vector<Cost> dist(numStates(), Cost{kInf, kInf});
    parent->assign(numStates(), Step{MoveKind::kAllSlice, -1});
    using Item = std::pair<Cost, int>;
    auto greater = [](const Item& a, const Item& b) {
      return b.first < a.first;
    };
    std::priority_queue<Item, std::vector<Item>, decltype(greater)> queue(
        greater);
    dist[src] = Cost{};
    queue.push({dist[src], src});
    while (!queue.empty()) {
      auto [d, s] = queue.top();
      queue.pop();
      if (dist[s] < d) {
        continue;
      }
      for (const Edge& e : nodes_[s].edges) {
        Cost nd = d + e.cost;
        if (nd < dist[e.to]) {
          dist[e.to] = nd;
          (*parent)[e.to] = Step{e.kind, s};
          queue.push({nd, e.to});
        }
      }
    }
    return dist;
  }

  // Information-theoretic lower bound on elements received per device: the
  // maximum over all devices of target shard elements not already held in
  // `src`.
  int64_t lowerBound(int src, int tgt) const {
    std::vector<int64_t> axisCoord(axes_.size(), 0);
    int64_t maxMissingElems = 0;
    for (int64_t device = 0; device < numDevices_; ++device) {
      int64_t rem = device;
      for (int a = static_cast<int>(axes_.size()) - 1; a >= 0; --a) {
        axisCoord[a] = rem % axes_[a].size;
        rem /= axes_[a].size;
      }
      int64_t tgtElems = 1, overlap = 1;
      for (int d = 0; d < rank(); ++d) {
        auto [srcBegin, srcLen] = interval(nodes_[src].state[d], d, axisCoord);
        auto [tgtBegin, tgtLen] = interval(nodes_[tgt].state[d], d, axisCoord);
        tgtElems *= tgtLen;
        overlap *= std::max<int64_t>(
            0, std::min(srcBegin + srcLen, tgtBegin + tgtLen) -
                   std::max(srcBegin, tgtBegin));
      }
      maxMissingElems = std::max(maxMissingElems, tgtElems - overlap);
    }
    return maxMissingElems;
  }

  // Prints the sharding of node `s` in sdy syntax.
  std::string stateString(int s) const {
    std::string result = "[";
    for (auto [d, dimAxes] : llvm::enumerate(nodes_[s].state)) {
      if (d > 0) {
        result += ", ";
      }
      result += '{';
      for (auto [i, sub] : llvm::enumerate(dimAxes)) {
        if (i > 0) {
          result += ", ";
        }
        const MeshAxis& axis = axes_[sub.axis];
        result += "\"" + axis.name + "\"";
        if (sub.size != axis.size) {
          result += ":(" + std::to_string(sub.preSize) + ")" +
                    std::to_string(sub.size);
        }
      }
      result += '}';
    }
    return result + "]";
  }

  // Converts an SDY `AxisRefAttr` into a `SubAxis`, or nullopt if invalid.
  std::optional<SubAxis> subAxisOf(AxisRefAttr axisRef) const {
    auto axisIt = llvm::find_if(
        axes_, [&](const MeshAxis& a) { return a.name == axisRef.getName(); });
    if (axisIt == axes_.end()) {
      return std::nullopt;
    }
    int axisIndex = axisIt - axes_.begin();
    int64_t pre = 1, size = axisIt->size;
    if (SubAxisInfoAttr info = axisRef.getSubAxisInfo()) {
      pre = info.getPreSize();
      size = info.getSize();
      if (size == axisIt->size) {
        return std::nullopt;
      }
    }
    if (pre < 1 || size < 2 || axisIt->size % (pre * size) != 0) {
      return std::nullopt;
    }
    return SubAxis{axisIndex, pre, size};
  }

  std::optional<int> indexOf(TensorShardingAttr sharding) const {
    State s;
    s.reserve(sharding.getRank());
    for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
      std::vector<SubAxis>& dimAxes = s.emplace_back();
      for (AxisRefAttr axisRef : dimSharding.getAxes()) {
        std::optional<SubAxis> sub = subAxisOf(axisRef);
        if (!sub) {
          return std::nullopt;
        }
        appendSubAxis(dimAxes, *sub);
      }
    }
    return indexOf(s);
  }

  static int64_t subAxesSize(llvm::ArrayRef<SubAxis> subAxes) {
    int64_t size = 1;
    for (const SubAxis& sub : subAxes) {
      size *= sub.size;
    }
    return size;
  }

  // Returns true if every dimension sharded in both `from` and `to` keeps the
  // same divisible padded size; padding/trimming only happens on unsharded
  // dims.
  static bool isValidPaddingTransition(const Node& from, const Node& to) {
    for (auto [fromShards, toShards, fromPadded, toPadded] : llvm::zip_equal(
             from.dimShards, to.dimShards, from.paddedShape, to.paddedShape)) {
      if (fromShards > 1 && toShards > 1 && fromPadded != toPadded) {
        return false;
      }
    }
    return true;
  }

  // Per-device elements received when all-gathering `from` into `to`.
  static int64_t allGatherElems(const Node& from, const Node& to) {
    int64_t groupSize = 1;
    for (auto [fromShards, toShards] :
         llvm::zip_equal(from.dimShards, to.dimShards)) {
      groupSize *= fromShards / toShards;
    }
    return from.localElems * (groupSize - 1);
  }

  // Per-device elements received when all-to-all-ing `from` into `to` over
  // `groupSize` devices, accounting for pre-padding unsharded target dims.
  static int64_t allToAllElems(const Node& from, const Node& to,
                               int64_t groupSize) {
    int64_t inLocalElems = 1;
    for (auto [fromShards, fromPadded, toPadded] :
         llvm::zip_equal(from.dimShards, from.paddedShape, to.paddedShape)) {
      inLocalElems *= std::max(fromPadded, toPadded) / fromShards;
    }
    assert(inLocalElems % groupSize == 0);
    return inLocalElems / groupSize * (groupSize - 1);
  }

 private:
  using AtomState = std::vector<std::vector<int>>;

  static void forEachAtomChain(
      const std::vector<std::vector<std::vector<SubAxis>>>& chainsPerAxis,
      size_t axisIndex, std::vector<SubAxis>& curAtoms,
      llvm::function_ref<void(llvm::ArrayRef<SubAxis>)> fn) {
    if (axisIndex == chainsPerAxis.size()) {
      fn(curAtoms);
      return;
    }
    size_t prevSize = curAtoms.size();
    for (const std::vector<SubAxis>& chain : chainsPerAxis[axisIndex]) {
      curAtoms.insert(curAtoms.end(), chain.begin(), chain.end());
      forEachAtomChain(chainsPerAxis, axisIndex + 1, curAtoms, fn);
      curAtoms.resize(prevSize);
    }
  }

  State canonicalize(const AtomState& atomState,
                     llvm::ArrayRef<SubAxis> atoms) const {
    State s(rank());
    for (int d = 0; d < rank(); ++d) {
      for (int a : atomState[d]) {
        appendSubAxis(s[d], atoms[a]);
      }
    }
    return s;
  }

  int getOrInsertNode(State s) {
    std::string k = key(s);
    auto [it, inserted] = index_.try_emplace(std::move(k), nodes_.size());
    if (inserted) {
      nodes_.push_back(makeNode(std::move(s)));
    }
    return it->second;
  }

  // Builds the graph node and padded shard metadata for state `s`.
  Node makeNode(State s) const {
    Node node;
    node.dimShards.reserve(rank());
    node.paddedShape.reserve(rank());
    node.localElems = 1;
    for (int d = 0; d < rank(); ++d) {
      int64_t shards = subAxesSize(s[d]);
      int64_t padded = llvm::alignTo(shape_[d], shards);
      node.dimShards.push_back(shards);
      node.paddedShape.push_back(padded);
      node.localElems *= padded / shards;
    }
    node.state = std::move(s);
    return node;
  }

  // Encodes `s` into a byte string for hash-map lookup.
  static std::string key(const State& s) {
    auto append = [](std::string& k, int64_t v) {
      k.append(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    std::string k;
    for (const std::vector<SubAxis>& dimAxes : s) {
      for (const SubAxis& sub : dimAxes) {
        append(k, sub.axis);
        append(k, sub.preSize);
        append(k, sub.size);
      }
      append(k, -1);
    }
    return k;
  }

  // Global [begin, length) of the unpadded shard held by the device at
  // `axisCoord` along dimension `d`, when sharded by `dimAxes`.
  std::pair<int64_t, int64_t> interval(
      llvm::ArrayRef<SubAxis> dimAxes, int d,
      llvm::ArrayRef<int64_t> axisCoord) const {
    int64_t index = 0, numShards = 1;
    for (const SubAxis& sub : dimAxes) {
      int64_t stride = axes_[sub.axis].size / (sub.preSize * sub.size);
      int64_t subCoord = (axisCoord[sub.axis] / stride) % sub.size;
      index = index * sub.size + subCoord;
      numShards *= sub.size;
    }
    int64_t shardSize = llvm::divideCeil(shape_[d], numShards);
    int64_t begin = std::min(shape_[d], index * shardSize);
    int64_t end = std::min(shape_[d], (index + 1) * shardSize);
    return {begin, end - begin};
  }

  // Calls `fn` on every atom state obtained by appending unused atoms to `cur`.
  void forEachExtension(size_t numAtoms, AtomState cur,
                        llvm::function_ref<void(const AtomState&)> fn) const {
    llvm::BitVector used(numAtoms);
    for (const std::vector<int>& dimAtoms : cur) {
      for (int a : dimAtoms) {
        used.set(a);
      }
    }
    forEachExtension(numAtoms, 0, used, cur, fn);
  }

  void forEachExtension(size_t numAtoms, int dim, llvm::BitVector& used,
                        AtomState& cur,
                        llvm::function_ref<void(const AtomState&)> fn) const {
    if (dim == rank()) {
      fn(cur);
      return;
    }
    forEachExtension(numAtoms, dim + 1, used, cur, fn);
    for (int a = 0; a < static_cast<int>(numAtoms); ++a) {
      if (used.test(a)) {
        continue;
      }
      cur[dim].push_back(a);
      used.set(a);
      forEachExtension(numAtoms, dim, used, cur, fn);
      used.reset(a);
      cur[dim].pop_back();
    }
  }

  // Populates states and single-collective edges induced by one choice of
  // prime-factor ordering per mesh axis.
  void buildChain(llvm::ArrayRef<SubAxis> atoms) {
    std::map<AtomState, int> chainStates;
    std::map<std::vector<int64_t>, std::vector<int>> sameSizes;
    forEachExtension(atoms.size(), AtomState(rank()),
                     [&](const AtomState& atomState) {
                       int u = getOrInsertNode(canonicalize(atomState, atoms));
                       chainStates.try_emplace(atomState, u);
                       sameSizes[nodes_[u].dimShards].push_back(u);
                     });
    auto atomsProduct = [&](llvm::ArrayRef<int> indices) {
      int64_t size = 1;
      for (int i : indices) {
        size *= atoms[i].size;
      }
      return size;
    };
    for (const auto& [atomState, s] : chainStates) {
      // All-slice to any valid extension, and the reverse all-gather.
      forEachExtension(
          atoms.size(), atomState, [&](const AtomState& nextAtomState) {
            int ti = chainStates.at(nextAtomState);
            if (ti == s || !isValidPaddingTransition(nodes_[s], nodes_[ti])) {
              return;
            }
            nodes_[s].edges.push_back({ti, Cost{0, 1}, MoveKind::kAllSlice});
            nodes_[ti].edges.push_back(
                {s, Cost{allGatherElems(nodes_[ti], nodes_[s]), 1},
                 MoveKind::kAllGather});
          });
      // All-to-all of a suffix of one dimension to the end of another.
      for (int src = 0; src < rank(); ++src) {
        const std::vector<int>& srcAtoms = atomState[src];
        for (size_t start = 0; start < srcAtoms.size(); ++start) {
          llvm::ArrayRef<int> moved =
              llvm::ArrayRef(srcAtoms).drop_front(start);
          for (int tgt = 0; tgt < rank(); ++tgt) {
            if (tgt == src) {
              continue;
            }
            AtomState nextAtomState = atomState;
            nextAtomState[src].resize(start);
            nextAtomState[tgt].insert(nextAtomState[tgt].end(), moved.begin(),
                                      moved.end());
            int ti = chainStates.at(nextAtomState);
            if (!isValidPaddingTransition(nodes_[s], nodes_[ti])) {
              continue;
            }
            nodes_[s].edges.push_back(
                {ti,
                 Cost{allToAllElems(nodes_[s], nodes_[ti], atomsProduct(moved)),
                      1},
                 MoveKind::kAllToAll});
          }
        }
      }
      // Collective-permute to any coexisting state with the same per-dim sizes.
      for (int t : sameSizes[nodes_[s].dimShards]) {
        if (t != s) {
          nodes_[s].edges.push_back(
              {t, Cost{nodes_[s].localElems, 1}, MoveKind::kCollectivePermute});
        }
      }
    }
  }

  std::vector<MeshAxis> axes_;
  std::vector<int64_t> shape_;
  int64_t numDevices_ = 1;
  std::vector<Node> nodes_;
  std::unordered_map<std::string, int> index_;
};

// Result of running the pass on one reshard.
struct PassPlan {
  std::vector<Step> steps;
  Cost cost;
  std::string error;  // Empty iff the plan is valid and reaches the target.
};

std::string tensorType(llvm::ArrayRef<int64_t> shape) {
  std::string result = "tensor<";
  for (int64_t dimSize : shape) {
    result += std::to_string(dimSize) + "x";
  }
  return result + "f32>";
}

bool validateCollectiveStep(Operation& op, const ReshardGraph& graph,
                            const ReshardGraph::Node& from,
                            const ReshardGraph::Node& to, MoveKind& kind,
                            int64_t& elems, std::string& error) {
  auto toSubAxes = [&](llvm::ArrayRef<AxisRefAttr> refs,
                       std::vector<SubAxis>& out) {
    out.reserve(refs.size());
    for (AxisRefAttr axisRef : refs) {
      std::optional<SubAxis> sub = graph.subAxisOf(axisRef);
      if (!sub) {
        return false;
      }
      out.push_back(*sub);
    }
    return true;
  };
  elems = 0;
  bool valid = statesCanCoexist(from.state, to.state);
  if (auto allSlice = dyn_cast<AllSliceOp>(op)) {
    kind = MoveKind::kAllSlice;
    State expected = from.state;
    ArrayRef<AxisRefListAttr> slicingAxes = allSlice.getSlicingAxes();
    valid = valid && slicingAxes.size() == expected.size();
    for (size_t d = 0; valid && d < expected.size(); ++d) {
      std::vector<SubAxis> added;
      if (!toSubAxes(slicingAxes[d].getValue(), added) ||
          from.dimShards[d] * ReshardGraph::subAxesSize(added) !=
              to.dimShards[d]) {
        valid = false;
        break;
      }
      for (SubAxis sub : added) {
        appendSubAxis(expected[d], sub);
      }
    }
    valid = valid && from.state != to.state && expected == to.state &&
            ReshardGraph::isValidPaddingTransition(from, to);
  } else if (auto allGather = dyn_cast<AllGatherOp>(op)) {
    kind = MoveKind::kAllGather;
    State expected = from.state;
    ArrayRef<AxisRefListAttr> gatheringAxes = allGather.getGatheringAxes();
    valid = valid && gatheringAxes.size() == expected.size();
    for (size_t d = 0; valid && d < expected.size(); ++d) {
      std::vector<SubAxis> gathered;
      if (!toSubAxes(gatheringAxes[d].getValue(), gathered) ||
          !popSuffixSubAxes(expected[d], gathered)) {
        valid = false;
        break;
      }
    }
    valid = valid && from.state != to.state && expected == to.state &&
            ReshardGraph::isValidPaddingTransition(from, to);
    if (valid) {
      elems = ReshardGraph::allGatherElems(from, to);
    }
  } else if (auto allToAll = dyn_cast<AllToAllOp>(op)) {
    kind = MoveKind::kAllToAll;
    State expected = from.state;
    int64_t groupSize = 1;
    if (valid && allToAll.getParams().size() == 1) {
      AllToAllParamAttr param = allToAll.getParams().front();
      int64_t srcDim = param.getSrcDim();
      int64_t tgtDim = param.getTgtDim();
      int64_t rank = expected.size();
      std::vector<SubAxis> moved;
      if (srcDim < 0 || srcDim >= rank || tgtDim < 0 || tgtDim >= rank ||
          srcDim == tgtDim || !toSubAxes(param.getAxes(), moved) ||
          moved.empty() || !popSuffixSubAxes(expected[srcDim], moved)) {
        valid = false;
      } else {
        for (SubAxis sub : moved) {
          appendSubAxis(expected[tgtDim], sub);
        }
        groupSize = ReshardGraph::subAxesSize(moved);
      }
    } else {
      valid = false;
    }
    valid = valid && from.state != to.state && expected == to.state &&
            ReshardGraph::isValidPaddingTransition(from, to);
    if (valid) {
      elems = ReshardGraph::allToAllElems(from, to, groupSize);
    }
  } else if (isa<CollectivePermuteOp>(op)) {
    kind = MoveKind::kCollectivePermute;
    valid = valid && from.state != to.state && from.dimShards == to.dimShards;
    elems = from.localElems;
  } else {
    error = "unsupported collective";
    return false;
  }
  if (!valid) {
    error = "invalid " + moveName(kind).str();
    return false;
  }
  return true;
}

// Runs the pass on one function per (src, tgt) pair and decodes the emitted
// collectives into sharding states.
std::vector<PassPlan> runPass(MLIRContext& context, const ReshardGraph& graph,
                              llvm::ArrayRef<std::pair<int, int>> pairs) {
  std::string type = tensorType(graph.shape());
  std::string program = "sdy.mesh @mesh = <[";
  for (auto [i, axis] : llvm::enumerate(graph.axes())) {
    if (i > 0) {
      program += ", ";
    }
    program += "\"" + axis.name + "\"=" + std::to_string(axis.size);
  }
  program += "]>\n";
  for (auto [i, pair] : llvm::enumerate(pairs)) {
    program += "func.func @f" + std::to_string(i) + "(%arg0: " + type +
               " {sdy.sharding = #sdy.sharding<@mesh, " +
               graph.stateString(pair.first) + ">}) -> " + type +
               " {\n  %0 = sdy.reshard %arg0 <@mesh, " +
               graph.stateString(pair.second) + "> : " + type +
               "\n  return %0 : " + type + "\n}\n";
  }
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(program, &context);
  if (!module) {
    ADD_FAILURE() << "Failed to parse generated MLIR program";
    return std::vector<PassPlan>(pairs.size(),
                                 PassPlan{{}, {}, "parse failed"});
  }
  std::string diagnostics;
  LogicalResult passResult = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic& diag) {
      diagnostics += diag.str() + "\n";
      return success();
    });
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(createReshardToCollectivesPass());
    passResult = pm.run(*module);
  }
  if (failed(passResult)) {
    if (pairs.size() == 1) {
      return {PassPlan{{}, {}, "pass failed: " + diagnostics}};
    }
    // Binary-split the batch to isolate failing pairs in O(k log N) runs.
    size_t mid = pairs.size() / 2;
    std::vector<PassPlan> plans =
        runPass(context, graph, pairs.take_front(mid));
    llvm::append_range(plans, runPass(context, graph, pairs.drop_front(mid)));
    return plans;
  }

  std::vector<PassPlan> plans(pairs.size());
  int funcIndex = 0;
  for (func::FuncOp funcOp : module->getOps<func::FuncOp>()) {
    PassPlan& plan = plans[funcIndex];
    auto [src, tgt] = pairs[funcIndex++];
    int cur = src;
    Value curValue = funcOp.getArgument(0);
    for (Operation& op : funcOp.getBody().front()) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        if (returnOp.getOperand(0) != curValue) {
          plan.error = "bad return";
        }
        break;
      }
      auto collective = dyn_cast<CollectiveOpInterface>(op);
      if (!collective || op.getOperand(0) != curValue) {
        plan.error = "unexpected op";
        break;
      }
      std::optional<int> next = graph.indexOf(collective.getOutSharding());
      if (!next) {
        plan.error = "invalid sharding";
        break;
      }
      const ReshardGraph::Node& from = graph.node(cur);
      const ReshardGraph::Node& to = graph.node(*next);
      MoveKind kind;
      int64_t elems = 0;
      if (!validateCollectiveStep(op, graph, from, to, kind, elems,
                                  plan.error)) {
        break;
      }
      plan.steps.push_back({kind, *next});
      plan.cost = plan.cost + Cost{elems, 1};
      cur = *next;
      curValue = op.getResult(0);
    }
    if (plan.error.empty() && cur != tgt) {
      plan.error = "wrong final sharding";
    }
  }
  return plans;
}

struct SweepStats {
  int64_t pairs = 0;
  // Pairs where the pass fails or emits an invalid or incomplete sequence.
  int64_t numErrors = 0;
  std::vector<std::string> errors;
  int64_t subOptimalElems = 0;
  int64_t subOptimalOps = 0;
  int64_t totalPassElems = 0;
  int64_t totalOptimalElems = 0;
  double maxRatio = 1;
  // Histogram of greedy / optimal elements for bytes-suboptimal pairs.
  std::map<std::string, int64_t> ratioHistogram;
};

struct Counterexample {
  int src, tgt;
  Cost greedy, optimal;
  std::vector<Step> greedySteps, optimalSteps;
};

std::string ratioBucket(double ratio) {
  if (ratio <= 1.25) {
    return "(1, 1.25]";
  }
  if (ratio <= 1.5) {
    return "(1.25, 1.5]";
  }
  if (ratio <= 2) {
    return "(1.5, 2]";
  }
  if (ratio <= 4) {
    return "(2, 4]";
  }
  return "> 4";
}

class ReshardOptimalityTest : public ::testing::Test {
 protected:
  void SetUp() override { loadAllRequiredDialects(&context); }

  // Runs the pass on every (src, tgt) pair with src in `srcs`, checks it is
  // correct and never beats the shortest-path cost, and returns stats.
  SweepStats sweep(const ReshardGraph& graph, const std::vector<int>& srcs,
                   std::vector<Counterexample>* counterexamples,
                   const std::string& name) {
    SweepStats stats;
    // Optionally dumps per-pair costs, to diff two versions of the pass.
    std::unique_ptr<std::ofstream> dump;
    if (const char* dir = std::getenv("RESHARD_OPTIMALITY_DUMP_DIR")) {
      dump = std::make_unique<std::ofstream>(std::string(dir) + "/" + name +
                                             ".tsv");
    }
    constexpr int kBatch = 4096;
    std::vector<std::pair<int, int>> batch;
    std::vector<std::vector<Cost>> dists;
    std::vector<std::vector<Step>> parents;
    std::unordered_map<int, int> srcSlot;
    auto flush = [&]() {
      std::vector<PassPlan> plans = runPass(context, graph, batch);
      for (auto [pair, plan] : llvm::zip_equal(batch, plans)) {
        auto [src, tgt] = pair;
        const Cost& best = dists[srcSlot[src]][tgt];
        if (!plan.error.empty()) {
          if (stats.errors.size() < 10) {
            stats.errors.push_back(plan.error + ": " + graph.stateString(src) +
                                   " -> " + graph.stateString(tgt));
          }
          ++stats.numErrors;
          continue;
        }
        // Shortest-path cost is exact, so the pass can never be cheaper.
        ASSERT_FALSE(plan.cost < best)
            << graph.stateString(src) << " -> " << graph.stateString(tgt);
        ASSERT_GE(best.elems, graph.lowerBound(src, tgt));
        ++stats.pairs;
        stats.totalPassElems += plan.cost.elems;
        stats.totalOptimalElems += best.elems;
        if (dump) {
          *dump << graph.stateString(src) << " -> " << graph.stateString(tgt)
                << "\t" << plan.cost.elems << "\t" << plan.cost.ops << "\t"
                << best.elems << "\t" << best.ops << "\n";
        }
        if (plan.cost.elems > best.elems) {
          ++stats.subOptimalElems;
          double ratio =
              best.elems == 0
                  ? 1e9
                  : static_cast<double>(plan.cost.elems) / best.elems;
          stats.maxRatio = std::max(stats.maxRatio, ratio);
          ++stats.ratioHistogram[ratioBucket(ratio)];
        } else if (plan.cost.ops > best.ops) {
          ++stats.subOptimalOps;
        }
        if (counterexamples && best < plan.cost) {
          std::vector<Step> optimalSteps;
          for (int s = tgt; s != src;) {
            const Step& step = parents[srcSlot[src]][s];
            optimalSteps.push_back({step.kind, s});
            s = step.state;
          }
          std::reverse(optimalSteps.begin(), optimalSteps.end());
          counterexamples->push_back(
              {src, tgt, plan.cost, best, plan.steps, optimalSteps});
        }
      }
      batch.clear();
      dists.clear();
      parents.clear();
      srcSlot.clear();
    };
    for (int src : srcs) {
      srcSlot[src] = dists.size();
      parents.emplace_back();
      dists.push_back(graph.shortestFrom(src, &parents.back()));
      for (int tgt = 0; tgt < graph.numStates(); ++tgt) {
        if (tgt != src) {
          batch.push_back({src, tgt});
        }
      }
      if (batch.size() >= kBatch) {
        flush();
      }
      if (HasFatalFailure()) {
        return stats;
      }
    }
    if (!batch.empty()) {
      flush();
    }
    return stats;
  }

  std::string describe(const ReshardGraph& graph, const Counterexample& c) {
    std::ostringstream os;
    os << graph.stateString(c.src) << " -> " << graph.stateString(c.tgt)
       << "\n  greedy (" << c.greedy.elems << " elems, " << c.greedy.ops
       << " ops):";
    for (const Step& step : c.greedySteps) {
      os << " " << std::string_view(moveName(step.kind)) << " "
         << graph.stateString(step.state) << ";";
    }
    os << "\n  optimal (" << c.optimal.elems << " elems, " << c.optimal.ops
       << " ops):";
    for (const Step& step : c.optimalSteps) {
      os << " " << std::string_view(moveName(step.kind)) << " "
         << graph.stateString(step.state) << ";";
    }
    return os.str();
  }

  void report(const ReshardGraph& graph, const SweepStats& stats,
              std::vector<Counterexample>& counterexamples, int numToPrint) {
    std::ostringstream os;
    double excessPct =
        stats.totalOptimalElems == 0
            ? 0.0
            : 100.0 * (stats.totalPassElems - stats.totalOptimalElems) /
                  stats.totalOptimalElems;
    os << "states=" << graph.numStates() << " pairs=" << stats.pairs
       << " suboptimal_elems=" << stats.subOptimalElems << " ("
       << 100.0 * stats.subOptimalElems / std::max<int64_t>(1, stats.pairs)
       << "%) excess_elems_pct=" << excessPct
       << "% suboptimal_ops_only=" << stats.subOptimalOps
       << " max_ratio=" << stats.maxRatio << " errors=" << stats.numErrors
       << "\n";
    for (const std::string& error : stats.errors) {
      os << "  error: " << error << "\n";
    }
    for (const auto& [bucket, count] : stats.ratioHistogram) {
      os << "  ratio " << bucket << ": " << count << "\n";
    }
    // Smallest counterexamples first: fewest sub-axes, then largest gap.
    auto numSubAxes = [&](const Counterexample& c) {
      int n = 0;
      for (int s : {c.src, c.tgt}) {
        for (const auto& dimAxes : graph.state(s)) {
          n += dimAxes.size();
        }
      }
      return n;
    };
    llvm::stable_sort(
        counterexamples, [&](const Counterexample& a, const Counterexample& b) {
          bool aElems = a.greedy.elems > a.optimal.elems;
          bool bElems = b.greedy.elems > b.optimal.elems;
          return std::make_tuple(!aElems, numSubAxes(a),
                                 -(a.greedy.elems - a.optimal.elems)) <
                 std::make_tuple(!bElems, numSubAxes(b),
                                 -(b.greedy.elems - b.optimal.elems));
        });
    // Most common (greedy -> optimal) collective sequences among
    // bytes-suboptimal pairs.
    auto signature = [](const std::vector<Step>& steps) {
      std::string result;
      for (const Step& step : steps) {
        if (!result.empty()) {
          result += ',';
        }
        result += moveName(step.kind);
      }
      return result;
    };
    std::map<std::string, int64_t> patterns;
    std::map<std::string, const Counterexample*> patternExample;
    for (const Counterexample& c : counterexamples) {
      if (c.greedy.elems > c.optimal.elems) {
        std::string pattern =
            signature(c.greedySteps) + " => " + signature(c.optimalSteps);
        ++patterns[pattern];
        patternExample.try_emplace(pattern, &c);
      }
    }
    std::vector<std::pair<int64_t, std::string>> sortedPatterns;
    for (const auto& [pattern, count] : patterns) {
      sortedPatterns.push_back({count, pattern});
    }
    std::sort(sortedPatterns.rbegin(), sortedPatterns.rend());
    for (int i = 0; i < std::min<int>(12, sortedPatterns.size()); ++i) {
      os << "  pattern x" << sortedPatterns[i].first << ": "
         << sortedPatterns[i].second << "\n    e.g. "
         << describe(graph, *patternExample[sortedPatterns[i].second]) << "\n";
    }
    for (int i = 0; i < std::min<int>(numToPrint, counterexamples.size());
         ++i) {
      os << describe(graph, counterexamples[i]) << "\n";
    }
    llvm::errs() << os.str();
  }

  MLIRContext context{MLIRContext::Threading::DISABLED};
};

std::vector<int> allStates(const ReshardGraph& graph) {
  std::vector<int> result(graph.numStates());
  for (int i = 0; i < graph.numStates(); ++i) {
    result[i] = i;
  }
  return result;
}

TEST_F(ReshardOptimalityTest, ShortestPathMatchesHandComputedCosts) {
  ReshardGraph graph({{"x", 2}, {"y", 4}}, {8, 16});
  const SubAxis x{0, 1, 2}, y{1, 1, 4}, y1{1, 1, 2};
  std::vector<Step> parent;
  int replicated = *graph.indexOf(State{{}, {}});
  std::vector<Cost> dist = graph.shortestFrom(replicated, &parent);
  // Slicing is free.
  EXPECT_EQ(dist[*graph.indexOf(State{{x}, {y}})], (Cost{0, 1}));
  int xOnDim0 = *graph.indexOf(State{{x}, {}});
  dist = graph.shortestFrom(xOnDim0, &parent);
  // Gathering "x" receives the other half of the tensor (64 elements).
  EXPECT_EQ(dist[replicated], (Cost{64, 1}));
  // Moving "x" to dim 1 is one all-to-all sending half the local shard.
  EXPECT_EQ(dist[*graph.indexOf(State{{}, {x}})], (Cost{32, 1}));
  // Replacing "x" with "y":(1)2 is one collective-permute.
  EXPECT_EQ(dist[*graph.indexOf(State{{y1}, {}})], (Cost{64, 1}));
  EXPECT_EQ(graph.lowerBound(xOnDim0, *graph.indexOf(State{{y1}, {}})), 64);
  int xy1OnDim0 = *graph.indexOf(State{{x, y1}, {}});
  dist = graph.shortestFrom(xy1OnDim0, &parent);
  // Two all-to-alls ("y":(1)2 then "x") cost 16 + 16 = 32 < cp(32) + a2a(24).
  EXPECT_EQ(dist[*graph.indexOf(State{{}, {y1, x}})], (Cost{32, 2}));
  // Moving ["x", "y":(1)2] to dim 1 then gathering "y":(1)2 costs 24 + 32 = 56.
  EXPECT_EQ(dist[*graph.indexOf(State{{}, {x}})], (Cost{56, 2}));
}

TEST_F(ReshardOptimalityTest, UnevenShapeShortestPathMatchesHandComputedCosts) {
  ReshardGraph graph2x3({{"a", 2}, {"b", 3}}, {5, 9});
  const SubAxis a{0, 1, 2}, b{1, 1, 3};
  std::vector<Step> parent;
  int replicated2x3 = *graph2x3.indexOf(State{{}, {}});
  int aOnDim0 = *graph2x3.indexOf(State{{a}, {}});
  int bOnDim0 = *graph2x3.indexOf(State{{b}, {}});
  int bOnDim1 = *graph2x3.indexOf(State{{}, {b}});

  // All-gathering "b" on dim 0 gathers 2x9 padded shards across 3 peers (36
  // elements received, vs naive 45 - 18 = 27).
  std::vector<Cost> dist = graph2x3.shortestFrom(bOnDim0, &parent);
  EXPECT_EQ(dist[replicated2x3], (Cost{36, 1}));
  EXPECT_EQ(graph2x3.lowerBound(bOnDim0, replicated2x3), 36);
  EXPECT_EQ(dist[bOnDim1], (Cost{12, 1}));
  EXPECT_EQ(graph2x3.lowerBound(bOnDim0, bOnDim1), 12);

  // All-to-all from [{}, {"b"}] to [{"b"}, {}] pads dim 0 from 5 to 6 first.
  dist = graph2x3.shortestFrom(bOnDim1, &parent);
  EXPECT_EQ(dist[bOnDim0], (Cost{12, 1}));
  EXPECT_EQ(graph2x3.lowerBound(bOnDim1, bOnDim0), 12);

  // Since alignTo(5, 2) == alignTo(5, 6) == alignTo(5, 3) == 6, [{"a"}, {}] ->
  // [{"b"}, {}] can slice + permute + gather without unsharding.
  dist = graph2x3.shortestFrom(aOnDim0, &parent);
  EXPECT_EQ(dist[bOnDim0], (Cost{18, 3}));
  EXPECT_EQ(graph2x3.lowerBound(aOnDim0, bOnDim0), 18);

  ReshardGraph graph2x4({{"x", 2}, {"y", 4}}, {5});
  const SubAxis x2{0, 1, 2}, y2{1, 1, 2}, yFull{1, 1, 4};
  int x = *graph2x4.indexOf(State{{x2}});
  int xy1 = *graph2x4.indexOf(State{{x2, y2}});
  int xy = *graph2x4.indexOf(State{{x2, yFull}});

  // Direct all-slice from 2-way (padded 6) to 4-way (padded 8) is blocked, so
  // resharding must gather to replicated then slice.
  dist = graph2x4.shortestFrom(x, &parent);
  EXPECT_EQ(dist[xy1], (Cost{3, 2}));

  // 4-way and 8-way share padded size 8, so direct slice/gather are valid.
  dist = graph2x4.shortestFrom(xy1, &parent);
  EXPECT_EQ(dist[xy], (Cost{0, 1}));
  dist = graph2x4.shortestFrom(xy, &parent);
  EXPECT_EQ(dist[xy1], (Cost{1, 1}));
  EXPECT_EQ(dist[x], (Cost{7, 2}));
  EXPECT_EQ(graph2x4.lowerBound(xy, x), 3);
}

TEST_F(ReshardOptimalityTest,
       CompositeAxisShortestPathMatchesHandComputedCosts) {
  // Size 6 has 2 prime-factor orderings (2*3 and 3*2), yielding 19 rank-2
  // states (vs. 11 along a single ordering).
  ReshardGraph graph6({{"a", 6}}, {6, 12});
  EXPECT_EQ(graph6.numStates(), 19);
  const SubAxis a1_2{0, 1, 2}, a1_3{0, 1, 3}, a3_2{0, 3, 2};
  std::vector<Step> parent;
  int s1_2 = *graph6.indexOf(State{{a1_2}, {}});
  int s1_3 = *graph6.indexOf(State{{a1_3}, {}});
  int s3_2 = *graph6.indexOf(State{{a3_2}, {}});

  // [{"a":(1)2}, {}] -> [{"a":(1)3}, {}] slices "a":(2)3 to full "a" along
  // ordering 2*3 (0 elems), then gathers "a":(3)2 along ordering 3*2 (12
  // elems), matching the 12-element overlap lower bound (vs. 36 to unshard).
  std::vector<Cost> dist = graph6.shortestFrom(s1_2, &parent);
  EXPECT_EQ(dist[s1_3], (Cost{12, 2}));
  EXPECT_EQ(graph6.lowerBound(s1_2, s1_3), 12);

  // [{"a":(1)2}, {}] and [{"a":(3)2}, {}] cannot coexist in a single SDY
  // collective; all-gathering "a":(1)2 (36 elems) then slicing "a":(3)2 is
  // optimal and matches the 36-element overlap lower bound.
  EXPECT_EQ(dist[s3_2], (Cost{36, 2}));
  EXPECT_EQ(graph6.lowerBound(s1_2, s3_2), 36);

  // Size 36 = 2^2 * 3^2 has C(4, 2) = 6 prime-factor orderings (1,283 rank-2
  // states vs. 261 on a single ordering). On {36, 72}, [{"a":(1)4}, {}] ->
  // [{"a":(1)6}, {}] slices "a":(4)3 to shared LCM [{"a":(1)12}, {}] along
  // 2*2*3*3 (0 elems), then gathers "a":(6)2 along 2*3*2*3 (216 elems),
  // beating both GCD gather via [{"a":(1)2}, {}] (648) and unsharding (1,944).
  ReshardGraph graph36({{"a", 36}}, {36, 72});
  EXPECT_EQ(graph36.numStates(), 1283);
  int s36_1_4 = *graph36.indexOf(State{{{0, 1, 4}}, {}});
  int s36_1_6 = *graph36.indexOf(State{{{0, 1, 6}}, {}});
  dist = graph36.shortestFrom(s36_1_4, &parent);
  EXPECT_EQ(dist[s36_1_6], (Cost{216, 2}));
  EXPECT_EQ(graph36.lowerBound(s36_1_4, s36_1_6), 216);
}

TEST_F(ReshardOptimalityTest, ValidateCompositeAxisCollectives) {
  // Parses `op` from `from` to `to` without verification, then validates it.
  auto isValidStep = [&](const ReshardGraph& graph, const State& from,
                         const State& to, const std::string& op) {
    int src = *graph.indexOf(from), tgt = *graph.indexOf(to);
    const MeshAxis& axis = graph.axes().front();
    std::string type = tensorType(graph.shape());
    std::string program =
        "sdy.mesh @mesh = <[\"" + axis.name +
        "\"=" + std::to_string(axis.size) + "]>\nfunc.func @f(%arg0: " + type +
        " {sdy.sharding = #sdy.sharding<@mesh, " + graph.stateString(src) +
        ">}) -> " + type + " {\n  %0 = " + op + " %arg0 out_sharding=<@mesh, " +
        graph.stateString(tgt) + "> : " + type + "\n  return %0 : " + type +
        "\n}\n";
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
        program, ParserConfig(&context, /*verifyAfterParse=*/false));
    if (!module) {
      ADD_FAILURE() << "Failed to parse:\n" << program;
      return false;
    }
    auto funcOp = *module->getOps<func::FuncOp>().begin();
    MoveKind kind;
    int64_t elems = 0;
    std::string error;
    return validateCollectiveStep(funcOp.getBody().front().front(), graph,
                                  graph.node(src), graph.node(tgt), kind, elems,
                                  error);
  };
  ReshardGraph graph12({{"a", 12}}, {12, 24});
  const SubAxis a1_6{0, 1, 6}, a1_2{0, 1, 2}, a2_3{0, 2, 3}, a2_2{0, 2, 2};
  // Moving or gathering "a":(2)3 splits merged "a":(1)6 into "a":(1)2.
  EXPECT_TRUE(isValidStep(graph12, State{{a1_6}, {}}, State{{a1_2}, {a2_3}},
                          "sdy.all_to_all [{\"a\":(2)3}: 0->1]"));
  EXPECT_TRUE(isValidStep(graph12, State{{a1_6}, {}}, State{{a1_2}, {}},
                          "sdy.all_gather [{\"a\":(2)3}, {}]"));
  // "a":(1)2 and "a":(2)2 coexist on "a"=12.
  EXPECT_TRUE(isValidStep(graph12, State{{a1_2}, {}}, State{{a2_2}, {}},
                          "sdy.collective_permute"));
  // "a":(1)2 and "a":(3)2 cannot coexist on "a"=6.
  ReshardGraph graph6({{"a", 6}}, {6, 12});
  EXPECT_FALSE(isValidStep(graph6, State{{a1_2}, {}},
                           State{{SubAxis{0, 3, 2}}, {}},
                           "sdy.collective_permute"));
}

struct SweepCase {
  std::string name;
  std::vector<MeshAxis> axes;
  std::vector<int64_t> shape;
  // Number of pairs where the pass communicates more than optimal. Lower
  // this when the pass improves; it must never go up. Negative means unchecked.
  int64_t maxSubOptimalElems;
  // Number of pairs where the pass fails or emits a wrong sequence.
  int64_t maxErrors;
};

class ReshardOptimalitySweepTest
    : public ReshardOptimalityTest,
      public ::testing::WithParamInterface<SweepCase> {};

TEST_P(ReshardOptimalitySweepTest, AllPairs) {
  const SweepCase& c = GetParam();
  ReshardGraph graph(c.axes, c.shape);
  std::vector<Counterexample> counterexamples;
  SweepStats stats = sweep(graph, allStates(graph), &counterexamples, c.name);
  ASSERT_FALSE(HasFatalFailure());
  llvm::errs() << "== " << c.name << "\n";
  report(graph, stats, counterexamples, 8);
  if (c.maxErrors >= 0) {
    EXPECT_LE(stats.numErrors, c.maxErrors);
  }
  if (c.maxSubOptimalElems >= 0) {
    EXPECT_LE(stats.subOptimalElems, c.maxSubOptimalElems);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SmallMeshes, ReshardOptimalitySweepTest,
    ::testing::Values(
        SweepCase{"2x3_rank2", {{"a", 2}, {"b", 3}}, {6, 12}, 16, 0},
        SweepCase{"2x3_uneven_rank2", {{"a", 2}, {"b", 3}}, {5, 9}, 41, 0},
        SweepCase{"2x4_rank2", {{"a", 2}, {"b", 4}}, {8, 16}, 288, 0},
        SweepCase{"2x4_uneven_rank2", {{"a", 2}, {"b", 4}}, {6, 10}, 1326, 0},
        SweepCase{"2x4_rank3", {{"a", 2}, {"b", 4}}, {8, 16, 24}, 2574, 0},
        SweepCase{
            "2x4_uneven_rank3", {{"a", 2}, {"b", 4}}, {5, 6, 10}, 6678, 0},
        SweepCase{"2x2x3_uneven_rank2",
                  {{"a", 2}, {"b", 2}, {"c", 3}},
                  {6, 10},
                  1368,
                  0},
        SweepCase{
            "2x3x4_rank2", {{"a", 2}, {"b", 3}, {"c", 4}}, {24, 48}, 20106, 0},
        SweepCase{"2x3x4_uneven_rank2",
                  {{"a", 2}, {"b", 3}, {"c", 4}},
                  {5, 10},
                  47301,
                  0},
        SweepCase{"6_rank2", {{"a", 6}}, {6, 12}, 152, 0},
        SweepCase{"6_uneven_rank2", {{"a", 6}}, {5, 9}, 191, 0},
        SweepCase{"6_rank3", {{"a", 6}}, {6, 12, 18}, 552, 0},
        SweepCase{"2x6_rank2", {{"a", 2}, {"b", 6}}, {12, 24}, 3978, 0},
        SweepCase{"12_rank2", {{"a", 12}}, {12, 24}, 10276, 0},
        SweepCase{
            "2x2x4_rank2", {{"a", 2}, {"b", 2}, {"c", 4}}, {16, 32}, 7080, 0}),
    [](const ::testing::TestParamInfo<SweepCase>& info) {
      return info.param.name;
    });

std::vector<SweepCase> largeSweepCases() {
  const char* env = std::getenv("SDY_RUN_LARGE_OPTIMALITY_SWEEP");
  if (!env || std::string_view(env) != "1") {
    return {};
  }
  return {
      SweepCase{"2x3x4_rank3",
                {{"a", 2}, {"b", 3}, {"c", 4}},
                {24, 48, 72},
                196824,
                0},
      SweepCase{"2x3x4_uneven_rank3",
                {{"a", 2}, {"b", 3}, {"c", 4}},
                {5, 10, 14},
                402456,
                0},
      SweepCase{"2x2x4_rank3",
                {{"a", 2}, {"b", 2}, {"c", 4}},
                {16, 32, 48},
                103896,
                0},
      SweepCase{"2x8_rank3", {{"a", 2}, {"b", 8}}, {16, 32, 48}, 103896, 0},
      SweepCase{"2x3x4x5_rank2",
                {{"a", 2}, {"b", 3}, {"c", 4}, {"d", 5}},
                {120, 240},
                1218418,
                0},
      SweepCase{"2x6_rank3", {{"a", 2}, {"b", 6}}, {12, 24, 36}, 22650, 0},
      SweepCase{"12_rank3", {{"a", 12}}, {12, 24, 36}, 56622, 0},
  };
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ReshardOptimalitySweepTest);

INSTANTIATE_TEST_SUITE_P(LargeSweep, ReshardOptimalitySweepTest,
                         ::testing::ValuesIn(largeSweepCases()),
                         [](const ::testing::TestParamInfo<SweepCase>& info) {
                           return info.param.name;
                         });

}  // namespace
}  // namespace sdy
}  // namespace mlir
