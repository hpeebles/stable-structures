//! The root-to-leaf path that [`BTreeMap::insert_many`](crate::BTreeMap::insert_many) and
//! [`BTreeMap::remove_many`](crate::BTreeMap::remove_many) hold open across a batch, so
//! that each modified node is written to stable memory once the batch moves past it
//! rather than once per key.
//!
//! [`Batch`] is a map together with the path held open on it, and [`PathLevel`] is one
//! level of that path. This module has what the two kinds of batch share: descending a
//! level, and writing a node back once the batch has left it. Inserting is in
//! `bulk_insert` and removing in `bulk_remove`, each as further methods of [`Batch`].
//!
//! # Attributing the cost
//!
//! Building with the `bench_scope` feature gives the shared steps a canbench scope each:
//!
//! - `bulk_release` — a node leaving the path: written out if dirty, else returned to the
//!   node cache
//! - `bulk_descend` — taking one child out of the cache and recording its bounds
//!
//! `bulk_insert` and `bulk_remove` add scopes of their own. The feature is off by default
//! and adds significant overhead, so the numbers are for comparing parts against each
//! other within one run, not against results measured without it.

use crate::btreemap::node::{Node, NodeType, B};
use crate::types::NULL;
use crate::{BTreeMap, Memory, Storable};

/// The fewest entries a node other than the root may hold.
pub(super) const MIN_ENTRIES: usize = B - 1;

/// One side of the path: that of the smaller keys, or that of the larger.
#[derive(Clone, Copy)]
pub(super) enum Side {
    Left,
    Right,
}

/// One level of the root-to-leaf path a batch holds open.
pub(super) struct PathLevel<K: Storable + Ord + Clone> {
    pub(super) node: Node<K>,

    /// Index of this node among its parent's children. Meaningless for the root.
    pub(super) index_in_parent: usize,

    /// The key range this node is responsible for, taken from the separators either side of
    /// it while descending. Both ends are exclusive, because a separator key is stored in
    /// the parent rather than the child. `None` means unbounded on that side, which is
    /// always the case for the root.
    ///
    /// Both ends are tracked, not just the upper one, so that a key arriving out of order
    /// is detected rather than being dropped into the wrong node. See [`PathLevel::covers`].
    pub(super) lower: Option<K>,
    pub(super) upper: Option<K>,

    /// Distance from the root, needed when handing the node back to the node cache.
    pub(super) depth: u8,

    /// Whether the node has been modified and so must be written out, rather than simply
    /// returned to the cache, when it leaves the path.
    pub(super) dirty: bool,
}

/// Where a key falls relative to the keys a node on the path is responsible for.
pub(super) enum Position {
    /// Under another node, further left.
    Before,
    /// The separator the node takes its lower bound from, which an ancestor holds.
    AtLower,
    /// Somewhere under the node.
    Within,
    /// The separator the node takes its upper bound from, which an ancestor holds.
    AtUpper,
    /// Under another node, further right.
    After,
}

impl<K: Storable + Ord + Clone> PathLevel<K> {
    /// The level a path starts with. Nothing bounds the root.
    pub(super) fn root(node: Node<K>) -> Self {
        Self {
            node,
            index_in_parent: 0,
            lower: None,
            upper: None,
            depth: 0,
            dirty: false,
        }
    }

    /// Where `key` falls relative to this node's bounds. [`PathLevel::covers`] says only
    /// whether it is within them.
    pub(super) fn position(&self, key: &K) -> Position {
        use core::cmp::Ordering;

        if let Some(lower) = &self.lower {
            match key.cmp(lower) {
                Ordering::Less => return Position::Before,
                Ordering::Equal => return Position::AtLower,
                Ordering::Greater => {}
            }
        }
        if let Some(upper) = &self.upper {
            match key.cmp(upper) {
                Ordering::Greater => return Position::After,
                Ordering::Equal => return Position::AtUpper,
                Ordering::Less => {}
            }
        }
        Position::Within
    }

    /// Whether `key` belongs somewhere under this node.
    ///
    /// For an ascending batch only the upper bound can ever fail, and that is the case this
    /// is tuned for. The lower bound is checked too so that an out-of-order key unwinds
    /// past this node — as far as the root if need be — instead of being handled here,
    /// which would break the ordering invariant of the tree.
    pub(super) fn covers(&self, key: &K) -> bool {
        self.lower.as_ref().is_none_or(|lower| lower < key)
            && self.upper.as_ref().is_none_or(|upper| key < upper)
    }

    /// The key range of this node's child at `idx`. The separators either side of the
    /// child bound the keys it may hold; beyond the outermost separator the child inherits
    /// this node's bound.
    pub(super) fn child_bounds<M: Memory>(&self, idx: usize, memory: &M) -> (Option<K>, Option<K>) {
        let lower = if idx > 0 {
            Some(self.node.key(idx - 1, memory).clone())
        } else {
            self.lower.clone()
        };
        let upper = if idx < self.node.entries_len() {
            Some(self.node.key(idx, memory).clone())
        } else {
            self.upper.clone()
        };
        (lower, upper)
    }
}

/// A batch of insertions or removals in progress: the map, and the root-to-leaf path
/// currently held open on it.
///
/// Dropping a `Batch` puts right and writes back whatever it is still holding: the
/// remaining path, and the header if `length` or `root_addr` moved. That is how a batch is
/// completed, so the drop is load-bearing on every call.
pub(super) struct Batch<'a, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    pub(super) map: &'a mut BTreeMap<K, V, M>,

    /// Root first, leaf last. Empty before the first key, and whenever a removal has just
    /// replaced the root.
    pub(super) path: Vec<PathLevel<K>>,

    /// Whether `length` or `root_addr` changed and the header still needs writing.
    pub(super) header_dirty: bool,
}

impl<'a, K, V, M> Batch<'a, K, V, M>
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

    /// Starts a fresh path at the root, which the map must have.
    pub(super) fn push_root(&mut self) {
        let root = self.map.take_or_load_node(self.map.root_addr);
        self.path.push(PathLevel::root(root));
    }

    /// Descends one level, into the child at `idx` of the node currently at the bottom.
    pub(super) fn push_child(&mut self, idx: usize) {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_descend"); // May add significant overhead.

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
    /// Only a removal can empty a node, and the only node it lets leave the path empty is
    /// the root, which has no minimum. An empty root is dropped instead: an empty leaf
    /// leaves the map empty, and an internal root left with no entries hands the root over
    /// to its only child.
    pub(super) fn release(&mut self, level: PathLevel<K>) {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_release"); // May add significant overhead.

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
}

impl<K, V, M> Drop for Batch<'_, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    fn drop(&mut self) {
        // A removal may have left the leaf it was last in under the minimum.
        self.fix_path(Side::Right);
        while let Some(level) = self.path.pop() {
            self.release(level);
        }
        if self.header_dirty {
            self.map.save_header();
        }
    }
}

#[cfg(test)]
impl<K, V, M> Batch<'_, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    /// Checks everything a batch relies on between two keys: the path runs from the root
    /// down through each level's recorded child slot, every level's bounds are the
    /// separators either side of that slot, and nothing but a leaf at the bottom is below
    /// the minimum. Called after every key under test, so that a slip in the bookkeeping
    /// is caught where it happens rather than as a damaged tree several keys later.
    pub(super) fn assert_invariants(&self) {
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
                    "the path does not start at the root"
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
                    "the keys of level {i} are out of order"
                );
            }
        }
    }
}
