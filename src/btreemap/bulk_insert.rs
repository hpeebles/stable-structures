//! Batched insertion for [`BTreeMap`](crate::BTreeMap).
//!
//! [`BTreeMap::insert_many`](crate::BTreeMap::insert_many) inserts a stream of pairs while
//! holding the current root-to-leaf path in memory, so that each modified node is written
//! to stable memory once the input moves past it rather than once per key. The held path
//! is a [`Batch`], and this module is what a batch does to insert.
//!
//! # Attributing the cost
//!
//! Building with the `bench_scope` feature adds two canbench scopes to the ones every
//! batch has, which are listed with [`Batch`]:
//!
//! - `bulk_insert_leaf` — placing the entry, *including* any split it triggers
//! - `bulk_insert_split` — the split cascade, nested inside the previous one
//!
//! So the cost of placing an entry with no split is `bulk_insert_leaf` minus
//! `bulk_insert_split`.

use crate::btreemap::node::NodeType;
use crate::btreemap::path::{Batch, PathLevel};
use crate::types::NULL;
use crate::{Memory, Storable};

impl<K, V, M> Batch<'_, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    pub(super) fn insert(&mut self, key: K, value: V) {
        // Drop back up the path until we reach a node that still covers this key. Since the
        // batch ascends, a node stops covering once the keys pass its upper bound.
        while self.path.last().is_some_and(|level| !level.covers(&key)) {
            let level = self.path.pop().expect("just checked that one exists");
            self.release(level);
        }

        if self.path.is_empty() {
            if self.map.root_addr == NULL {
                // The first key of an empty map goes into a new leaf root.
                let root = self.map.allocate_node(NodeType::Leaf);
                self.map.root_addr = root.address();
                self.header_dirty = true;
                self.path.push(PathLevel::root(root));
            } else {
                self.push_root();
            }
        }

        // Walk down to a leaf, taking each child out of the node cache as we go.
        loop {
            let bottom = self.path.last().expect("the path always holds a root here");
            if bottom.node.node_type() == NodeType::Leaf {
                break;
            }
            match bottom.node.search(&key, self.map.memory()) {
                Ok(idx) => {
                    // The key lives in this internal node; overwrite it where it sits.
                    let bottom = self.path.last_mut().expect("checked above");
                    bottom.node.set_value(idx, value.into_bytes_checked());
                    bottom.dirty = true;
                    return;
                }
                Err(idx) => self.push_child(idx),
            }
        }

        self.insert_into_leaf(key, value);
    }

    /// Places `key` in the leaf at the bottom of the path, splitting it first if it is
    /// full.
    fn insert_into_leaf(&mut self, key: K, value: V) {
        // Encloses `bulk_insert_split`, so subtract that to get the cost of placing the
        // entry alone.
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_insert_leaf"); // May add significant overhead.

        let search = {
            let leaf = self.path.last().expect("bottom is a leaf here");
            leaf.node.search(&key, self.map.memory())
        };

        let idx = match search {
            Ok(idx) => {
                let leaf = self.path.last_mut().expect("checked above");
                leaf.node.set_value(idx, value.into_bytes_checked());
                leaf.dirty = true;
                return;
            }
            Err(idx) => idx,
        };

        let idx = if self
            .path
            .last()
            .expect("bottom is a leaf here")
            .node
            .is_full()
        {
            self.split_leaf(&key);
            // The split moved half the entries out and may have swapped in the other half
            // entirely, so the position found above no longer means anything. Both halves
            // are at the minimum size, so there is room for the key wherever it now lands.
            self.path
                .last()
                .expect("bottom is a leaf here")
                .node
                .search(&key, self.map.memory())
                .expect_err("the key was absent and a split cannot introduce it")
        } else {
            // Nothing has touched the leaf since the search above, so reuse its answer.
            idx
        };

        let leaf = self.path.last_mut().expect("bottom is a leaf here");
        leaf.node
            .insert_entry(idx, (key, value.into_bytes_checked()));
        leaf.dirty = true;
        self.map.length += 1;
        self.header_dirty = true;
    }

    /// Makes room for `key` in the full leaf at the bottom of the path.
    ///
    /// A leaf split promotes a median into the parent, which may itself be full, so the
    /// split cascades up the path through every full ancestor — growing a new root if the
    /// whole path is full. Afterwards the bottom of the path is the half that owns `key`,
    /// with room for it.
    fn split_leaf(&mut self, key: &K) {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("bulk_insert_split"); // May add significant overhead.

        // Find the topmost level that has to split. Everything from there down to the leaf
        // is full, so none of them has anywhere to promote a median until the level above
        // it has split and made room.
        let mut topmost = self.path.len() - 1;
        while topmost > 0 && self.path[topmost - 1].node.is_full() {
            topmost -= 1;
        }

        if topmost == 0 {
            // The root is full too, so the tree gains a level and everything already on the
            // path moves one deeper.
            self.grow_root();
            topmost = 1;
        }

        // Split downwards. Each split leaves its node half empty, so by the time the level
        // below it splits, there is room for the median it promotes.
        for level in topmost..self.path.len() {
            self.split_level(level, key);
        }
    }

    /// Inserts a fresh internal root above the current one, leaving the tree a level deeper
    /// and the path a level longer. The new root holds the old one as its only child, and
    /// so has room for the median about to be promoted into it.
    fn grow_root(&mut self) {
        let old_root = self.path[0].node.address();
        let mut root = self.map.allocate_node(NodeType::Internal);
        root.push_child(old_root);
        self.map.root_addr = root.address();
        self.header_dirty = true;

        for level in self.path.iter_mut() {
            level.depth = level.depth.saturating_add(1);
        }
        self.path.insert(
            0,
            PathLevel {
                dirty: true,
                ..PathLevel::root(root)
            },
        );
    }

    /// Splits the full node at `path[index]` in two and promotes the median into its
    /// parent, which must have room. The path keeps whichever half owns `key`; the other
    /// half is finished with and is written out.
    fn split_level(&mut self, index: usize, key: &K) {
        debug_assert!(index > 0, "the root splits only after `grow_root`");
        debug_assert!(!self.path[index - 1].node.is_full());
        debug_assert!(self.path[index].node.is_full());

        let mut right = self.map.allocate_node(self.path[index].node.node_type());
        let right_addr = right.address();
        let (median_key, median_value) = self.path[index].node.split(&mut right, self.map.memory());
        let index_in_parent = self.path[index].index_in_parent;

        let parent = &mut self.path[index - 1];
        parent.node.insert_child(index_in_parent + 1, right_addr);
        parent
            .node
            .insert_entry(index_in_parent, (median_key.clone(), median_value));
        parent.dirty = true;

        // Keep whichever half the key belongs to; the other one is finished with. The key
        // cannot equal the median, which was already in the tree while the key was not.
        // The median becomes the boundary between the two halves.
        //
        // The levels below keep the bounds they already have: the separators either side of
        // them survive the split, they just end up in one half or the other.
        if *key < median_key {
            // The left half stays where it is, since `split` left it in place.
            let level = &mut self.path[index];
            level.upper = Some(median_key);
            level.dirty = true;
            self.map.save_node(&mut right);
        } else {
            let (mut left, moved_children) = {
                let level = &mut self.path[index];
                let left = core::mem::replace(&mut level.node, right);
                level.index_in_parent = index_in_parent + 1;
                level.lower = Some(median_key);
                level.dirty = true;
                let moved_children = left.children_len();
                (left, moved_children)
            };
            self.map.save_node(&mut left);

            // The level below travelled with the entries into the right half, so it sits at
            // a lower index among its parent's children than it did before.
            if let Some(below) = self.path.get_mut(index + 1) {
                // It has to have been in the right half: it covers `key`, and `key` is
                // above the median, so its index was at least the number of children the
                // left half kept.
                debug_assert!(below.index_in_parent >= moved_children);
                below.index_in_parent -= moved_children;
            }
        }
    }
}
