//! Batched removal for [`BTreeMap`].
//!
//! [`BTreeMap::remove_many`] removes a stream of keys while holding the current
//! root-to-leaf path in memory, as [`BTreeMap::insert_many`] does, so that each modified
//! node is written to stable memory once the batch moves past it rather than once per key.
//!
//! # Underflow
//!
//! [`remove`](BTreeMap::remove) keeps every node at or above the minimum size on the way
//! down, merging or rotating before it descends, so that a single removal never has to
//! back up. A batch cannot work that way: it removes many keys from the same leaf, and
//! rebalancing after each one would write the leaf's siblings over and over.
//!
//! Instead the leaf at the bottom of the path is allowed to fall below the minimum, even to
//! empty, while it is held. It is put right once, when it leaves the path, by taking
//! entries from a sibling or merging with one through the parent, which is still held.
//! Taking an entry out of the parent can leave it under the minimum in turn, so the fix
//! cascades up the path, but only internal nodes, all of them still held, are fixed that
//! way, and they are fixed straight away. So every node off the path, and every internal
//! node on it, is always a valid B-tree node; only the held leaf may not be.
//!
//! The node kept by a rebalance is always the one being fixed, with the sibling merged
//! into it or lending it entries, so the path never has to switch to a different node.
//!
//! # Attributing the cost
//!
//! Building with the `bench_scope` feature splits a batch into three canbench scopes:
//!
//! - `bulk_remove_release` — a node leaving the path, including fixing an underflow
//! - `bulk_remove_rebalance` — the underflow fix, nested inside the previous one
//! - `bulk_remove_descend` — taking one child out of the cache and recording its bounds

use crate::btreemap::node::{Node, NodeType, B};
use crate::btreemap::path::PathLevel;
use crate::types::NULL;
use crate::{BTreeMap, Memory, Storable};

/// The fewest entries a node other than the root may hold.
const MIN_ENTRIES: usize = B - 1;

/// Removes a stream of keys while keeping the current root-to-leaf path in memory, so that
/// each modified node is written to stable memory once the batch moves past it rather
/// than once per key.
///
/// Dropping a `BulkRemove` puts right and writes back whatever it is still holding: the
/// remaining path, and the header if `length` or `root_addr` moved. That is how a batch is
/// completed, so the drop is load-bearing on every call.
pub(super) struct BulkRemove<'a, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    map: &'a mut BTreeMap<K, V, M>,

    /// Root first, leaf last. Empty before the first removal, and whenever the root has
    /// just been replaced.
    path: Vec<PathLevel<K>>,

    /// Whether `length` or `root_addr` changed and the header still needs writing.
    header_dirty: bool,
}

impl<'a, K, V, M> BulkRemove<'a, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    pub(super) fn new(map: &'a mut BTreeMap<K, V, M>) -> Self {
        Self {
            map,
            path: Vec::new(),
            header_dirty: false,
        }
    }

    pub(super) fn remove(&mut self, key: &K) {
        // Drop back up the path until we reach a node that still covers this key.
        while self.path.last().is_some_and(|level| !level.covers(key)) {
            let level = self.path.pop().expect("just checked that one exists");
            self.release(level);
        }

        if self.path.is_empty() {
            if self.map.root_addr == NULL {
                return;
            }
            self.push_root();
        }

        loop {
            let bottom = self.path.last().expect("the path always holds a root here");
            let search = bottom.node.search(key, self.map.memory());
            match (bottom.node.node_type(), search) {
                (NodeType::Leaf, Ok(idx)) => {
                    let leaf = self.path.last_mut().expect("checked above");
                    leaf.node.remove_entry(idx, self.map.memory());
                    leaf.dirty = true;
                    self.removed_one();
                    return;
                }
                (NodeType::Leaf, Err(_)) => return,
                (NodeType::Internal, Ok(idx)) => {
                    self.remove_from_internal(idx);
                    return;
                }
                (NodeType::Internal, Err(idx)) => self.push_child(idx),
            }
        }
    }

    fn removed_one(&mut self) {
        self.map.length -= 1;
        self.header_dirty = true;
    }

    /// Removes the entry at `idx` of the internal node at the bottom of the path by
    /// replacing it with its successor, the first entry of the leftmost leaf of the subtree
    /// to its right. The path descends to that leaf, which is where an ascending batch goes
    /// next, and the leaf is left to underflow if it must, like any other held leaf.
    fn remove_from_internal(&mut self, idx: usize) {
        let internal = self.path.len() - 1;
        self.push_child(idx + 1);
        while self.path.last().expect("just pushed").node.node_type() == NodeType::Internal {
            self.push_child(0);
        }

        let successor = {
            let leaf = self.path.last_mut().expect("just pushed");
            leaf.dirty = true;
            leaf.node.remove_entry(0, self.map.memory())
        };
        let successor_key = successor.0.clone();

        let level = &mut self.path[internal];
        level.node.swap_entry(idx, successor, self.map.memory());
        level.dirty = true;

        // The levels just pushed took the removed key as their lower bound, and it has
        // been replaced by the successor. Left as it was, a later removal of the
        // successor would be looked for in the leaf, which no longer holds it.
        for level in &mut self.path[internal + 1..] {
            level.lower = Some(successor_key.clone());
        }

        self.removed_one();
    }

    /// Starts a fresh path at the root.
    fn push_root(&mut self) {
        let node = self.map.take_or_load_node(self.map.root_addr);
        self.path.push(PathLevel {
            node,
            index_in_parent: 0,
            lower: None,
            upper: None,
            depth: 0,
            dirty: false,
        });
    }

    /// Descends one level, into the child at `idx` of the node currently at the bottom.
    fn push_child(&mut self, idx: usize) {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_remove_descend"); // May add significant overhead.

        let (address, lower, upper, depth) = {
            let bottom = self.path.last().expect("only called while descending");
            let (lower, upper) = bottom.child_bounds(idx, self.map.memory());
            (
                bottom.node.child(idx),
                lower,
                upper,
                bottom.depth.saturating_add(1),
            )
        };
        let node = self.map.take_or_load_node(address);
        self.path.push(PathLevel {
            node,
            index_in_parent: idx,
            lower,
            upper,
            depth,
            dirty: false,
        });
    }

    /// Releases a node that has left the path. A node under the minimum is first put right
    /// through its parent, which is now the bottom of the path; the root has no minimum,
    /// and is dropped once it is empty. The node is then written out if modified, and
    /// handed back to the node cache otherwise.
    fn release(&mut self, level: PathLevel<K>) {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_remove_release"); // May add significant overhead.

        let PathLevel {
            mut node,
            index_in_parent,
            depth,
            dirty,
            ..
        } = level;

        if self.path.is_empty() {
            self.release_root(node, dirty, depth);
            return;
        }

        if node.entries_len() >= MIN_ENTRIES {
            if dirty {
                self.map.save_node(&mut node);
            } else {
                self.map.return_node(node, depth);
            }
            return;
        }

        let parent = self.path.last_mut().expect("checked above");
        Self::rebalance(self.map, &mut node, index_in_parent, &mut parent.node);
        parent.dirty = true;
        self.map.save_node(&mut node);
        self.fix_path();

        // The rebalance may have left the root with no entries, and the released node as
        // its only child. The released node is then the root.
        if self.path.len() == 1 && self.path[0].node.entries_len() == 0 {
            let root = self.path.pop().expect("checked above");
            debug_assert_eq!(root.node.child(0), node.address());
            self.map.root_addr = node.address();
            self.map.deallocate_node(root.node);
            self.header_dirty = true;
        }
    }

    /// Releases the root, the last node on the path. An empty leaf root leaves the map
    /// empty; an internal root left with no entries hands the root over to its only child.
    fn release_root(&mut self, mut root: Node<K>, dirty: bool, depth: u8) {
        if root.entries_len() > 0 {
            if dirty {
                self.map.save_node(&mut root);
            } else {
                self.map.return_node(root, depth);
            }
            return;
        }
        self.map.root_addr = match root.node_type() {
            NodeType::Leaf => NULL,
            NodeType::Internal => root.child(0),
        };
        self.map.deallocate_node(root);
        self.header_dirty = true;
    }

    /// Puts right every internal node on the path left under the minimum by a rebalance
    /// beneath it, bottom-up, and drops a root left with no entries if its only child is
    /// still held. A root with nothing else on the path is left to [`Self::release`].
    fn fix_path(&mut self) {
        let mut level = self.path.len() - 1;
        while level > 0 && self.path[level].node.entries_len() < MIN_ENTRIES {
            let (above, rest) = self.path.split_at_mut(level);
            let parent = &mut above[level - 1];
            let (current, below) = rest.split_first_mut().expect("level is on the path");

            let moved = Self::rebalance(
                self.map,
                &mut current.node,
                current.index_in_parent,
                &mut parent.node,
            );
            parent.dirty = true;

            // The node is still held, so its own bookkeeping has to follow it.
            if moved.merged_left {
                current.index_in_parent -= 1;
            }
            let (lower, upper) = parent.child_bounds(current.index_in_parent, self.map.memory());
            current.lower = lower;
            current.upper = upper;
            current.dirty = true;
            if let Some(below) = below.first_mut() {
                below.index_in_parent += moved.prepended;
            }

            level -= 1;
        }

        if self.path.len() > 1 && self.path[0].node.entries_len() == 0 {
            // The root's last entry went down into its only child, which is still held:
            // that child becomes the root, and the tree a level shallower.
            let root = self.path.remove(0);
            debug_assert_eq!(root.node.child(0), self.path[0].node.address());
            self.map.root_addr = self.path[0].node.address();
            self.map.deallocate_node(root.node);
            self.header_dirty = true;
            for level in &mut self.path {
                level.depth = level.depth.saturating_sub(1);
            }
            let root = &mut self.path[0];
            root.index_in_parent = 0;
            root.lower = None;
            root.upper = None;
        }
    }

    /// Brings `node`, the child at `idx` of the held `parent`, back up to the minimum,
    /// using its right sibling if it has one and its left one otherwise. If the two hold
    /// too few entries between them to fill two nodes, the sibling and their separator are
    /// merged into `node` and the sibling is deallocated; otherwise entries rotate across
    /// through the parent until `node` is at the minimum and the sibling is written out.
    /// `node` and `parent` are left for the caller to save.
    fn rebalance(
        map: &mut BTreeMap<K, V, M>,
        node: &mut Node<K>,
        idx: usize,
        parent: &mut Node<K>,
    ) -> Moved {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_remove_rebalance"); // May add significant overhead.

        debug_assert!(parent.children_len() >= 2, "a held parent has a sibling");
        let sibling_is_left = idx + 1 == parent.children_len();
        let sibling_idx = if sibling_is_left { idx - 1 } else { idx + 1 };
        let separator_idx = if sibling_is_left { idx - 1 } else { idx };
        let mut sibling = map.take_or_load_node(parent.child(sibling_idx));

        let mut moved = Moved {
            merged_left: false,
            prepended: 0,
        };

        if node.entries_len() + sibling.entries_len() < 2 * MIN_ENTRIES + 1 {
            // Too few to fill two nodes: merge the sibling into `node`.
            let separator = parent.remove_entry(separator_idx, map.memory());
            parent.remove_child(sibling_idx);
            if sibling_is_left {
                moved.merged_left = true;
                moved.prepended = sibling.children_len();
            }
            let sibling = node.absorb(sibling, separator, sibling_is_left, map.memory());
            map.deallocate_node(sibling);
        } else {
            while node.entries_len() < MIN_ENTRIES {
                if sibling_is_left {
                    let entry = sibling
                        .pop_entry(map.memory())
                        .expect("the sibling can lend");
                    let separator = parent.swap_entry(separator_idx, entry, map.memory());
                    node.insert_entry(0, separator);
                    if let Some(child) = sibling.pop_child() {
                        node.insert_child(0, child);
                        moved.prepended += 1;
                    }
                } else {
                    let entry = sibling.remove_entry(0, map.memory());
                    let separator = parent.swap_entry(separator_idx, entry, map.memory());
                    node.push_entry(separator);
                    if sibling.node_type() == NodeType::Internal {
                        node.push_child(sibling.remove_child(0));
                    }
                }
            }
            map.save_node(&mut sibling);
        }

        moved
    }
}

#[cfg(test)]
impl<K, V, M> BulkRemove<'_, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    /// Checks everything the batch relies on between two keys: the path runs from the
    /// root down through each level's recorded child slot, every level's bounds are the
    /// separators either side of that slot, and nothing but a leaf at the bottom is below
    /// the minimum. Called after every key under test, so that a slip in the bookkeeping
    /// is caught where it happens rather than as a damaged tree several keys later.
    pub(super) fn assert_path_invariants(&self) {
        use crate::btreemap::node::CAPACITY;

        let memory = self.map.memory();
        for (i, level) in self.path.iter().enumerate() {
            let node = &level.node;
            let len = node.entries_len();
            assert_eq!(level.depth as usize, i, "depth of level {i}");

            if i == 0 {
                assert_eq!(
                    node.address(),
                    self.map.root_addr,
                    "the path starts at the root"
                );
                assert!(level.lower.is_none() && level.upper.is_none());
                assert_eq!(level.index_in_parent, 0);
            } else {
                let parent = &self.path[i - 1];
                assert_eq!(
                    parent.node.child(level.index_in_parent),
                    node.address(),
                    "level {i} is not at its recorded slot in its parent"
                );
                let (lower, upper) = parent.child_bounds(level.index_in_parent, memory);
                assert!(
                    lower == level.lower && upper == level.upper,
                    "the bounds of level {i} are not the separators around it"
                );
            }

            assert!(len <= CAPACITY, "level {i} is over capacity");
            match node.node_type() {
                NodeType::Internal => {
                    assert_eq!(node.children_len(), len + 1, "children of level {i}");
                    let minimum = if i == 0 { 1 } else { MIN_ENTRIES };
                    assert!(
                        len >= minimum,
                        "internal level {i} holds only {len} entries"
                    );
                }
                NodeType::Leaf => {
                    assert_eq!(
                        i + 1,
                        self.path.len(),
                        "a leaf above the bottom of the path"
                    );
                    assert_eq!(node.children_len(), 0);
                }
            }

            for j in 0..len {
                let key = node.key(j, memory);
                assert!(
                    level.covers(key),
                    "a key of level {i} is outside its bounds"
                );
                assert!(
                    j == 0 || node.key(j - 1, memory) < key,
                    "keys of level {i} unsorted"
                );
            }
        }
    }
}

/// What a rebalance did to the position of `node`'s children.
struct Moved {
    /// The left sibling was merged in, so `node` now sits one place further left in its
    /// parent.
    merged_left: bool,
    /// How many children were put in front of `node`'s own, shifting each of them right.
    prepended: usize,
}

impl<K, V, M> Drop for BulkRemove<'_, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    fn drop(&mut self) {
        while let Some(level) = self.path.pop() {
            self.release(level);
        }
        if self.header_dirty {
            self.map.save_header();
        }
    }
}
