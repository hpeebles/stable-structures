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
//! empty, while it is held. It is put right once, when the batch is about to leave it, by
//! taking entries from a sibling or merging with one through the parent, which is held too.
//! Taking an entry out of the parent can leave it under the minimum in turn, so the fix
//! cascades up the path, but internal nodes are fixed straight away. So every node off the
//! path, and every internal node on it, is always a valid B-tree node; only the held leaf
//! may not be.
//!
//! The node kept by a rebalance is always the one being fixed, with the sibling merged
//! into it or lending it entries, so the path never has to switch to a different node. The
//! sibling is the one on the side the batch is heading to, where there is one. What comes
//! over from it is then what the batch wants next, and since the leaf is still on the path
//! it goes on serving the batch: a leaf that keeps underflowing swallows its neighbours
//! one after another and is written once, when the batch finally moves beyond it.
//!
//! # Separators
//!
//! A key held by an internal node is replaced by its neighbour in key order, taken from a
//! leaf. Two cases are worth more than that, and both come up once per leaf when a run of
//! consecutive keys is removed.
//!
//! The key may be one of the two separators either side of the path, which the bounds of
//! the bottom of the path are copies of. Its neighbour is then at the near end of the held
//! leaf, and is moved up without leaving the path. Otherwise the leaf would be released
//! and written, the separator found from higher up, and the leaf read back to supply the
//! neighbour: once for every key of the run, since each neighbour moved up is the next key
//! to go.
//!
//! And the held leaf may have been emptied already. It has to go, and removing a child
//! takes one entry out of its parent with it, so the leaf and the separator go together,
//! and no sibling is read or written for either.
//!
//! # Attributing the cost
//!
//! Building with the `bench_scope` feature splits a batch into three canbench scopes:
//!
//! - `bulk_remove_release` — a node leaving the path: written out if dirty, else returned
//!   to the node cache
//! - `bulk_remove_rebalance` — bringing one node back up to the minimum
//! - `bulk_remove_descend` — taking one child out of the cache and recording its bounds

use crate::btreemap::node::{Node, NodeType, B};
use crate::btreemap::path::{PathLevel, Position};
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
        while let Some(bottom) = self.path.last() {
            let toward = match bottom.position(key) {
                Position::Within => break,
                // A separator this node takes a bound from can be removed from where the
                // path already is.
                Position::AtLower => return self.remove_adjacent_separator(Side::Left),
                Position::AtUpper => return self.remove_adjacent_separator(Side::Right),
                Position::Before => Side::Left,
                Position::After => Side::Right,
            };
            if self.path.len() > 1 && bottom.node.entries_len() < MIN_ENTRIES {
                // The leaf has to be put right before it leaves the path. Taking from the
                // sibling the key lies towards often brings the key under the leaf, so
                // the bounds are looked at again before anything is released.
                self.fix_path(toward);
                continue;
            }
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
                    leaf.node.discard_entry(idx);
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
    /// to its right. The path descends to that leaf, and the leaf is left to underflow if
    /// it must, like any other held leaf.
    fn remove_from_internal(&mut self, idx: usize) {
        let holder = self.path.len() - 1;
        self.push_child(idx + 1);
        self.replace_with_neighbour(holder, idx, Side::Left);
    }

    /// Removes the separator on `side` of the bottom of the path, the entry of an ancestor
    /// that the bottom takes its bound on that side from, without leaving the path.
    ///
    /// The separator is held by the parent of the deepest level that has a sibling on that
    /// side: every level below that one is an outermost child, and inherited the bound
    /// unchanged. So the path already runs down the edge of the subtree beside the
    /// separator, which is where the separator's neighbour in key order is.
    fn remove_adjacent_separator(&mut self, side: Side) {
        let turn = (1..self.path.len())
            .rev()
            .find(|&i| match side {
                Side::Left => self.path[i].index_in_parent > 0,
                Side::Right => {
                    self.path[i].index_in_parent + 1 < self.path[i - 1].node.children_len()
                }
            })
            .expect("a bound is a separator held by an ancestor");
        let idx = match side {
            Side::Left => self.path[turn].index_in_parent - 1,
            Side::Right => self.path[turn].index_in_parent,
        };
        self.replace_with_neighbour(turn - 1, idx, side);
    }

    /// Removes the entry at `idx` of the internal node at `path[holder]` by putting its
    /// neighbour in key order in its place: its successor if the entry is on the left of
    /// the path below the holder, its predecessor if it is on the right.
    ///
    /// The path must already lead from the holder into the child beside the entry and from
    /// there through outermost children only, back towards the entry, though it need not
    /// have reached a leaf yet.
    fn replace_with_neighbour(&mut self, holder: usize, idx: usize, side: Side) {
        loop {
            let bottom = self.path.last().expect("the holder");
            if bottom.node.node_type() == NodeType::Leaf {
                break;
            }
            let edge = match side {
                Side::Left => 0,
                Side::Right => bottom.node.children_len() - 1,
            };
            self.push_child(edge);
        }

        let leaf = self.path.last_mut().expect("just reached");
        if leaf.node.entries_len() > 0 {
            leaf.dirty = true;
            let neighbour = match side {
                Side::Left => leaf.node.remove_entry(0, self.map.memory()),
                Side::Right => leaf
                    .node
                    .pop_entry(self.map.memory())
                    .expect("checked above"),
            };
            self.set_separator(holder, idx, neighbour, side);
        } else {
            self.drop_drained_leaf(holder, idx, side);
        }
        self.removed_one();
    }

    /// Puts `entry` in place of the entry at `idx` of the internal node at `path[holder]`,
    /// which is on `side` of the path below it. The entry it replaces is discarded.
    ///
    /// The levels below took the replaced key as their bound on that side, so they take
    /// the new one. Left as it was, a later removal of `entry`'s key would be looked for
    /// in the leaf, which no longer holds it.
    fn set_separator(&mut self, holder: usize, idx: usize, entry: (K, Vec<u8>), side: Side) {
        for level in &mut self.path[holder + 1..] {
            let bound = match side {
                Side::Left => &mut level.lower,
                Side::Right => &mut level.upper,
            };
            *bound = Some(entry.0.clone());
        }
        let level = &mut self.path[holder];
        level.node.set_entry(idx, entry);
        level.dirty = true;
    }

    /// Removes the entry at `idx` of the internal node at `path[holder]` when the leaf its
    /// neighbour was looked for in, at the bottom of the path, has already been emptied by
    /// the batch.
    ///
    /// The leaf has to go in any case, and removing a child takes an entry out of its
    /// parent with it. If the parent is the holder, that entry is the one being removed,
    /// right beside the leaf. Otherwise the leaf is an outermost child of its parent, and
    /// the parent's entry beside it, which is then the neighbour, moves up to the holder.
    /// Either way no sibling is read or written.
    fn drop_drained_leaf(&mut self, holder: usize, idx: usize, side: Side) {
        let leaf = self.path.pop().expect("the leaf was just reached");
        let parent = self.path.len() - 1;
        let level = &mut self.path[parent];
        debug_assert_eq!(
            leaf.index_in_parent,
            match (parent == holder, side) {
                (true, Side::Left) => idx + 1,
                (true, Side::Right) => idx,
                (false, Side::Left) => 0,
                (false, Side::Right) => level.node.children_len() - 1,
            },
            "the leaf is not where the entry's neighbour would be"
        );
        level.node.remove_child(leaf.index_in_parent);
        level.dirty = true;
        if parent == holder {
            level.node.discard_entry(idx);
        } else {
            let neighbour = match side {
                Side::Left => level.node.remove_entry(0, self.map.memory()),
                Side::Right => level
                    .node
                    .pop_entry(self.map.memory())
                    .expect("an internal node below the root holds entries"),
            };
            self.set_separator(holder, idx, neighbour, side);
        }
        self.map.deallocate_node(leaf.node);

        // The parent is an entry down, which may have taken it under the minimum.
        self.fix_path(side);
        if self.path.len() == 1 && self.path[0].node.entries_len() == 0 {
            // It was the root, and that was its last entry: its other child is the root.
            let root = self.path.pop().expect("checked above");
            self.release(root);
        }
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

    /// Releases a node that has left the path: written out if modified, handed back to the
    /// node cache otherwise.
    ///
    /// The only node that leaves the path empty is the root, which has no minimum. An empty
    /// root is dropped instead: an empty leaf leaves the map empty, and an internal root
    /// left with no entries hands the root over to its only child.
    fn release(&mut self, level: PathLevel<K>) {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_remove_release"); // May add significant overhead.

        let mut node = level.node;
        debug_assert!(
            self.path.is_empty() || node.entries_len() >= MIN_ENTRIES,
            "only the root may leave the path under the minimum"
        );

        if node.entries_len() == 0 {
            assert!(self.path.is_empty(), "only the root may be empty");
            self.map.root_addr = match node.node_type() {
                NodeType::Leaf => NULL,
                NodeType::Internal => node.child(0),
            };
            self.map.deallocate_node(node);
            self.header_dirty = true;
        } else if level.dirty {
            self.map.save_node(&mut node);
        } else {
            self.map.return_node(node, level.depth);
        }
    }

    /// Puts right the node at the bottom of the path if it is under the minimum, keeping it
    /// on the path, and then every node above it that the fix took under the minimum in
    /// turn. A root left with no entries is dropped, if its only child is still held.
    ///
    /// Each node is filled from the sibling on the `toward` side, if it has one. That is
    /// the side the batch is heading to, so what the node takes in is what the batch wants
    /// next, and the node goes on serving it instead of being written and read back.
    fn fix_path(&mut self, toward: Side) {
        let mut level = self.path.len().saturating_sub(1);
        while level > 0 && self.path[level].node.entries_len() < MIN_ENTRIES {
            let (above, rest) = self.path.split_at_mut(level);
            let parent = &mut above[level - 1];
            let (current, below) = rest.split_first_mut().expect("level is on the path");

            let moved = Self::rebalance(
                self.map,
                &mut current.node,
                current.index_in_parent,
                &mut parent.node,
                toward,
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
    /// using its sibling on the `toward` side if it has one and its other sibling
    /// otherwise. If the two hold too few entries between them to fill two nodes, the
    /// sibling and their separator are merged into `node` and the sibling is deallocated;
    /// otherwise entries rotate across through the parent until `node` is at the minimum
    /// and the sibling is written out. `node` and `parent` are left for the caller to save.
    fn rebalance(
        map: &mut BTreeMap<K, V, M>,
        node: &mut Node<K>,
        idx: usize,
        parent: &mut Node<K>,
        toward: Side,
    ) -> Moved {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_remove_rebalance"); // May add significant overhead.

        debug_assert!(parent.children_len() >= 2, "a held parent has a sibling");
        let sibling_is_left = match toward {
            Side::Left => idx > 0,
            Side::Right => idx + 1 == parent.children_len(),
        };
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

/// One side of the path: that of the smaller keys, or that of the larger.
#[derive(Clone, Copy)]
enum Side {
    Left,
    Right,
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
        self.fix_path(Side::Right);
        while let Some(level) = self.path.pop() {
            self.release(level);
        }
        if self.header_dirty {
            self.map.save_header();
        }
    }
}
