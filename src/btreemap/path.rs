//! The root-to-leaf path that [`BTreeMap::insert_many`](crate::BTreeMap::insert_many) and
//! [`BTreeMap::remove_many`](crate::BTreeMap::remove_many) hold open across a batch, so
//! that each modified node is written to stable memory once the batch moves past it
//! rather than once per key. [`PathLevel`] is one level of it.

use crate::btreemap::node::Node;
use crate::{Memory, Storable};

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
