//! Batched insertion for [`BTreeMap`].
//!
//! [`BTreeMap::insert_many`] inserts a stream of pairs while holding the current
//! root-to-leaf path in memory, so that each modified node is written to stable memory
//! once the input moves past it rather than once per key. [`BulkInsert`] is that held
//! path, and [`PathLevel`] is one level of it.

use crate::btreemap::node::{Node, NodeType};
use crate::types::NULL;
use crate::{BTreeMap, Memory, Storable};

/// One level of the root-to-leaf path that [`BTreeMap::insert_many`] holds open across a
/// run of inserts.
struct PathLevel<K: Storable + Ord + Clone> {
    node: Node<K>,

    /// Index of this node among its parent's children. Meaningless for the root.
    index_in_parent: usize,

    /// The key range this node is responsible for, taken from the separators either side of
    /// it while descending. Both ends are exclusive, because a separator key is stored in
    /// the parent rather than the child. `None` means unbounded on that side, which is
    /// always the case for the root.
    ///
    /// Both ends are tracked, not just the upper one, so that a key arriving out of order
    /// is detected rather than being dropped into the wrong node. See [`PathLevel::covers`].
    lower: Option<K>,
    upper: Option<K>,

    /// Distance from the root, needed when handing the node back to the node cache.
    depth: u8,

    /// Whether the node has been modified and so must be written out, rather than simply
    /// returned to the cache, when it leaves the path.
    dirty: bool,
}

impl<K: Storable + Ord + Clone> PathLevel<K> {
    /// Whether `key` belongs somewhere under this node.
    ///
    /// For an ascending batch only the upper bound can ever fail, and that is the case this
    /// is tuned for. The lower bound is checked too so that an out-of-order key unwinds
    /// past this node — as far as the root if need be — instead of being inserted here,
    /// which would break the ordering invariant of the tree.
    fn covers(&self, key: &K) -> bool {
        self.lower.as_ref().is_none_or(|lower| lower < key)
            && self.upper.as_ref().is_none_or(|upper| key < upper)
    }
}

/// Inserts a stream of keys while keeping the current root-to-leaf path in
/// memory, so that each modified node is written to stable memory once the batch moves
/// past it rather than once per key.
///
/// Dropping a `BulkInsert` writes back whatever it is still holding: the remaining path,
/// and the header if `length` or `root_addr` moved. That is how a batch is completed, so
/// the drop is load-bearing on every call.
pub(super) struct BulkInsert<'a, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    map: &'a mut BTreeMap<K, V, M>,

    /// Root first, leaf last. Empty before the first insert and after a fallback.
    path: Vec<PathLevel<K>>,

    /// Whether `length` or `root_addr` changed and the header still needs writing.
    header_dirty: bool,
}

impl<'a, K, V, M> BulkInsert<'a, K, V, M>
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

    /// Releases a node that has left the path: written out if modified, handed back to the
    /// node cache otherwise.
    fn release(&mut self, level: PathLevel<K>) {
        let mut node = level.node;
        if level.dirty {
            self.map.save_node(&mut node);
        } else {
            self.map.return_node(node, level.depth);
        }
    }

    /// Releases the whole path, leaving it empty.
    fn release_path(&mut self) {
        while let Some(level) = self.path.pop() {
            self.release(level);
        }
    }

    pub(super) fn insert(&mut self, key: K, value: V) {
        // Drop back up the path until we reach a node that still covers this key. Since the
        // batch ascends, a node stops covering once the keys pass its upper bound.
        while self.path.last().is_some_and(|level| !level.covers(&key)) {
            let level = self.path.pop().expect("just checked that one exists");
            self.release(level);
        }

        if self.path.is_empty() {
            self.push_root();
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

    /// Starts a fresh path at the root, allocating one if the map is empty.
    fn push_root(&mut self) {
        let node = if self.map.root_addr == NULL {
            let node = self.map.allocate_node(NodeType::Leaf);
            self.map.root_addr = node.address();
            self.header_dirty = true;
            node
        } else {
            self.map.take_or_load_node(self.map.root_addr)
        };
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
        let (address, lower, upper, depth) = {
            let bottom = self.path.last().expect("only called while descending");
            // The separators either side of this child bound the keys it may hold; beyond
            // the outermost separator the child inherits its parent's bound.
            let lower = if idx > 0 {
                Some(bottom.node.key(idx - 1, self.map.memory()).clone())
            } else {
                bottom.lower.clone()
            };
            let upper = if idx < bottom.node.entries_len() {
                Some(bottom.node.key(idx, self.map.memory()).clone())
            } else {
                bottom.upper.clone()
            };
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

    /// Places `key` in the leaf at the bottom of the path, splitting it first if it is
    /// full.
    fn insert_into_leaf(&mut self, key: K, value: V) {
        let search = {
            let leaf = self.path.last().expect("bottom is a leaf here");
            leaf.node.search(&key, self.map.memory())
        };

        if let Ok(idx) = search {
            let leaf = self.path.last_mut().expect("checked above");
            leaf.node.set_value(idx, value.into_bytes_checked());
            leaf.dirty = true;
            return;
        }

        if self
            .path
            .last()
            .expect("bottom is a leaf here")
            .node
            .is_full()
            && !self.split_leaf(&key)
        {
            // The split could not be done in place. Write everything back and let the
            // ordinary insert path grow the tree, then start a new path for the next key.
            // The key is absent, so nothing is deserialized to build the discarded return.
            self.release_path();
            self.map.insert(key, value);
            // `insert` saves the header itself on every path that moves `length` or
            // `root_addr`, so this only guards against that ceasing to be true. It costs at
            // most one extra header write for the whole batch.
            self.header_dirty = true;
            return;
        }

        // A split leaves both halves at the minimum size, so there is room now.
        let leaf = self.path.last_mut().expect("bottom is a leaf here");
        let idx = leaf
            .node
            .search(&key, self.map.memory())
            .expect_err("the key was absent and a split cannot introduce it");
        leaf.node
            .insert_entry(idx, (key, value.into_bytes_checked()));
        leaf.dirty = true;
        self.map.length += 1;
        self.header_dirty = true;
    }

    /// Splits the full leaf at the bottom of the path in two and promotes the median into
    /// the parent, leaving the path pointing at whichever half now owns `key`.
    ///
    /// Returns `false` without touching anything when the split cannot be done here — the
    /// leaf is the root, or the parent has no room for the median — leaving the caller to
    /// fall back to [`BTreeMap::insert`], which grows the tree.
    fn split_leaf(&mut self, key: &K) -> bool {
        let depth_of_path = self.path.len();
        if depth_of_path < 2 || self.path[depth_of_path - 2].node.is_full() {
            return false;
        }

        let PathLevel {
            node: mut left,
            index_in_parent,
            lower,
            upper,
            depth,
            ..
        } = self
            .path
            .pop()
            .expect("checked that the path has a parent level");

        let mut right = self.map.allocate_node(NodeType::Leaf);
        let (median_key, median_value) = left.split(&mut right, self.map.memory());

        let parent = self
            .path
            .last_mut()
            .expect("checked that the path has a parent level");
        parent
            .node
            .insert_child(index_in_parent + 1, right.address());
        parent
            .node
            .insert_entry(index_in_parent, (median_key.clone(), median_value));
        parent.dirty = true;

        // Keep whichever half the key belongs to; the other one is finished with. The key
        // cannot equal the median, which was already in the tree while the key was not.
        // The median becomes the boundary between the two halves.
        let (keep, mut done, keep_index, keep_lower, keep_upper) = if *key < median_key {
            (left, right, index_in_parent, lower, Some(median_key))
        } else {
            (right, left, index_in_parent + 1, Some(median_key), upper)
        };
        self.map.save_node(&mut done);
        self.path.push(PathLevel {
            node: keep,
            index_in_parent: keep_index,
            lower: keep_lower,
            upper: keep_upper,
            depth,
            dirty: true,
        });
        true
    }
}

impl<K, V, M> Drop for BulkInsert<'_, K, V, M>
where
    K: Storable + Ord + Clone,
    V: Storable,
    M: Memory,
{
    fn drop(&mut self) {
        self.release_path();
        if self.header_dirty {
            self.map.save_header();
        }
    }
}
