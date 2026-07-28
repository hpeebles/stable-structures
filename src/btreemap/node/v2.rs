//! Node V2
//!
//! A v2 node is an iteration on a v1 node. Compared to a v1 node, it has a
//! smaller memory footprint and support for unbounded types.
//!
//! # Memory Layout
//!
//! To support unbounded types, a v2 node is variable in size and can span multiple
//! pages of memory if needed [^note]. There are two types of pages:
//!
//! 1. Initial Page (the first page of the node)
//! 2. Overflow page (subsequent pages of the node)
//!
//! ## Initial Page Memory Layout
//!
//! ```text
//! ---------------------------------------- <-- Header
//! Magic "BTN"             ↕ 3 bytes
//! ----------------------------------------
//! Layout version (2)      ↕ 1 byte
//! ----------------------------------------
//! Node type               ↕ 1 byte
//! ----------------------------------------
//! # Entries (k)           ↕ 2 bytes
//! ----------------------------------------
//! Overflow address        ↕ 8 bytes
//! ---------------------------------------- <-- Children (Address 15)
//! Child(0) address        ↕ 8 bytes
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Child(k + 1) address    ↕ 8 bytes
//! ---------------------------------------- <-- Keys
//! Key(0)
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Key(k)
//! ---------------------------------------- <-- Values
//! Value(0)
//! ----------------------------------------
//! ...
//! ----------------------------------------
//! Value(k)
//! ----------------------------------------
//! ```
//!
//! ## Overflow Page Memory Layout
//!
//! If the data to be stored in the initial page layout is larger than the page size,
//! then overflow pages are used.
//!
//! ```text
//! ----------------------------------------
//! Magic "NOF"             ↕ 3 bytes
//! ----------------------------------------
//! Next address            ↕ 8 byte
//! ----------------------------------------
//! Data
//! ----------------------------------------
//! ```
//!
//! ## Keys and Values
//! Keys and values are both encoded in memory as blobs.
//!
//! If they are variable in size (i.e. their `IS_FIXED` attribute is set to false),
//! then the size of the blob is encoded before the blob itself. Otherwise, no size
//! information is stored.
//!
//! [^note]: The page here refers to a fixed-size chunk of memory that is provided to the
//! node by the BTreeMap Allocator, and has no connection with OS memory pages or
//! Wasm pages.

use super::*;
use crate::btreemap::Allocator;
use crate::{btreemap::node::io::NodeWriter, types::NULL, WASM_PAGE_SIZE};

// Initial page
pub(super) const OVERFLOW_ADDRESS_OFFSET: Bytes = Bytes::new(7);
const ENTRIES_OFFSET: Bytes = Bytes::new(15);

// Overflow page
pub(super) const OVERFLOW_MAGIC: &[u8; 3] = b"NOF";
pub(super) const PAGE_OVERFLOW_NEXT_OFFSET: Bytes = Bytes::new(3);
pub(super) const PAGE_OVERFLOW_DATA_OFFSET: Bytes = Bytes::new(11);

// The minimum size a page can have.
// Rationale: a page size needs to at least store the header (15 bytes) + all the children
// addresses (88 bytes). We round that up to 128 to get a nice binary number.
const MINIMUM_PAGE_SIZE: u32 = 128;

// How much of the initial page a load reads in one batched read. A memory access costs a
// fixed amount per call plus a fee per byte, so batching many small reads into one pays
// for itself only while the batch stays small; a derived page can be many megabytes, and
// reading all of it to serve a handful of fields is a large regression. 1 KiB covers the
// header, the children and the key area of typical nodes.
const INITIAL_PAGE_READ_CAP: u64 = 1024;

// Keys are included in the batched read only while their bound keeps the read small.
// The walk below needs the size fields, which interleave with the key payloads, so
// covering the sizes means also buying the key bytes between them — bytes the lazy key
// loads may never look at. Up to this bound the size fields the read covers save more
// than the skipped key bytes cost; beyond it the read would mostly buy ignored bytes,
// which measured as a regression on the large-key benchmarks.
const BATCHED_READ_MAX_KEY_SIZE: u32 = 64;

// Slices at least this large are written directly rather than copied into the save
// buffer. Copying a slice into the buffer costs a few instructions per byte, while a
// separate write costs a fixed amount for the call and its page mapping — roughly a
// forty-byte copy. Below the threshold coalescing neighboring fields into one write
// wins; above it the copy costs more than the write it saves.
const DIRECT_WRITE_THRESHOLD: usize = 48;

// The save buffer's initial capacity. Only slices below `DIRECT_WRITE_THRESHOLD` land in
// the buffer, so it stays small; this covers the typical node without regrowing.
const WRITE_BUFFER_INITIAL_CAPACITY: usize = 1024;

impl<K: Storable + Ord + Clone> Node<K> {
    /// Creates a new v2 node at the given address.
    pub fn new_v2(address: Address, node_type: NodeType, page_size: PageSize) -> Node<K> {
        assert!(page_size.get() >= MINIMUM_PAGE_SIZE);

        Node {
            address,
            node_type,
            version: Version::V2(page_size),
            entries: vec![],
            children: vec![],
            overflows: Vec::with_capacity(0),
        }
    }

    /// Loads a v2 node from memory at the given address.
    pub(super) fn load_v2<M: Memory>(
        address: Address,
        page_size: PageSize,
        header: NodeHeader,
        memory: &M,
    ) -> Self {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("node_load_v2"); // May add significant overhead.

        // Load the node type.
        let node_type = match header.node_type {
            LEAF_NODE_TYPE => NodeType::Leaf,
            INTERNAL_NODE_TYPE => NodeType::Internal,
            other => unreachable!("Unknown node type {}", other),
        };

        // Load the number of entries
        let num_entries = header.num_entries as usize;

        // Read the front of the initial page in one go. Almost everything the load needs
        // — the children, the key sizes and small keys, and the value sizes — lives
        // there, so this replaces a separate memory read per field. A memory read costs
        // a fixed amount per call plus a fee per byte, so the read is kept as small as
        // its purpose allows: for key types bounded small enough (see
        // `BATCHED_READ_MAX_KEY_SIZE`) the header pins down the extent of the children
        // and key areas, and the value sizes follow immediately when the values are
        // empty, so that estimate is all that is fetched (the cap cannot bind under the
        // current `CAPACITY` and key-size gate; it guards changes to either). For
        // large or unbounded key types nothing worth reading ahead bounds the entry
        // area, so only the metadata ahead of it is batched and the entry walk reads
        // individually, as it did before batching. Fields beyond the read fall back to
        // individual reads. The read is also clamped to the end of the memory, since a
        // node's tail may extend past the last byte ever written.
        let page = {
            let children_bytes = match node_type {
                NodeType::Internal => (num_entries as u64 + 1) * Address::size().get(),
                NodeType::Leaf => 0,
            };
            let target = match K::BOUND {
                crate::storable::Bound::Bounded {
                    max_size,
                    is_fixed_size,
                } if max_size <= BATCHED_READ_MAX_KEY_SIZE => {
                    let key_size_field = if is_fixed_size { 0 } else { U32_SIZE.get() };
                    let keys_bytes = num_entries as u64 * (max_size as u64 + key_size_field);
                    let value_sizes_bytes = num_entries as u64 * U32_SIZE.get();
                    (ENTRIES_OFFSET.get() + children_bytes + keys_bytes + value_sizes_bytes)
                        .min(INITIAL_PAGE_READ_CAP)
                }
                // Nothing bounds the entry area for large or unbounded key types, so
                // no read pays for itself up front; every field falls back to an
                // individual read, exactly as before batching.
                _ => 0,
            };
            let memory_bytes = memory.size() * WASM_PAGE_SIZE;
            let len = (page_size.get() as u64)
                .min(target)
                .min(memory_bytes.saturating_sub(address.get()));
            let mut page = vec![];
            if len > 0 {
                read_to_vec(memory, address, &mut page, len as usize);
            }
            page
        };

        // Reads a little-endian u32 from the page, falling back to a memory read for the
        // rare node whose fields extend past the initial page.
        let page_u32 = |reader: &NodeReader<M>, offset: Address| -> u32 {
            let start = offset.get() as usize;
            match page.get(start..start + 4) {
                Some(bytes) => u32::from_le_bytes(bytes.try_into().unwrap()),
                None => read_u32(reader, offset),
            }
        };

        // Load the addresses of the node's overflow pages, if any.
        let first_overflow = {
            let start = OVERFLOW_ADDRESS_OFFSET.get() as usize;
            Address::from(match page.get(start..start + 8) {
                Some(bytes) => u64::from_le_bytes(bytes.try_into().unwrap()),
                None => read_u64(memory, address + OVERFLOW_ADDRESS_OFFSET),
            })
        };
        let overflows = read_overflows(first_overflow, memory);

        let reader = NodeReader {
            address,
            overflows: &overflows,
            page_size,
            memory,
        };

        let mut offset = Address::from(0);

        // Load children if this is an internal node.
        offset += ENTRIES_OFFSET;
        let children = if node_type == NodeType::Internal {
            let count = num_entries + 1;
            let byte_len = count * Address::size().get() as usize;
            let start = offset.get() as usize;
            let children = match page.get(start..start + byte_len) {
                Some(bytes) => bytes
                    .chunks_exact(Address::size().get() as usize)
                    .map(|chunk| Address::from(u64::from_le_bytes(chunk.try_into().unwrap())))
                    .collect(),
                None => read_address_vec(&reader, offset, count),
            };
            offset += Bytes::from(byte_len as u64);
            children
        } else {
            vec![]
        };

        // Load the keys (eagerly if small).
        const EAGER_LOAD_KEY_SIZE_THRESHOLD: u32 = 16;
        let mut entries = Vec::with_capacity(num_entries);
        let mut buf = vec![];

        for _ in 0..num_entries {
            let key_offset = Bytes::from(offset.get());

            // Get key size.
            let key_size = if K::BOUND.is_fixed_size() {
                K::BOUND.max_size()
            } else {
                let size = page_u32(&reader, offset);
                offset += U32_SIZE;
                size
            };

            // Eager-load small keys, defer large ones.
            let key = if key_size <= EAGER_LOAD_KEY_SIZE_THRESHOLD {
                let start = offset.get() as usize;
                let key_bytes = match page.get(start..start + key_size as usize) {
                    Some(bytes) => bytes,
                    None => {
                        read_to_vec(
                            &reader,
                            Address::from(offset.get()),
                            &mut buf,
                            key_size as usize,
                        );
                        &buf[..]
                    }
                };
                LazyKey::by_value(K::from_bytes(Cow::Borrowed(key_bytes)))
            } else {
                LazyKey::by_ref(key_offset, key_size)
            };

            offset += Bytes::from(key_size);
            entries.push((key, LazyValue::by_ref(Bytes::from(0_u64), 0)));
        }

        // Load the values
        for (_key, value) in entries.iter_mut() {
            // Load the values lazily.
            let value_size = page_u32(&reader, offset);
            *value = LazyValue::by_ref(Bytes::from(offset.get()), value_size);
            offset += U32_SIZE + Bytes::from(value_size as u64);
        }

        Self {
            address,
            entries,
            children,
            node_type,
            version: Version::V2(page_size),
            overflows,
        }
    }

    // Saves the node to memory.
    pub(super) fn save_v2<M: Memory>(&mut self, allocator: &mut Allocator<M>) {
        #[cfg(feature = "bench_scope")]
        let _p = canbench_rs::bench_scope("node_save_v2"); // May add significant overhead.

        let page_size = self.version.page_size().get();
        assert!(page_size >= MINIMUM_PAGE_SIZE);

        // Load all the entries. One pass is required to load all entries;
        // results are not stored to avoid unnecessary allocations.
        for i in 0..self.entries.len() {
            self.entry(i, allocator.memory());
        }

        // The writer takes care of allocating and deallocating overflow pages as
        // needed; the buffer below batches the many small field writes into few memory
        // writes while passing large slices (big keys and values) straight through,
        // uncopied.
        let mut writer = NodeWriter::new(
            self.address,
            self.overflows.clone(),
            self.page_size(),
            allocator,
        );

        // The node is serialized into `buf` and written out in a handful of writes
        // instead of one write per field. `buf` always holds the pending bytes for the
        // contiguous range starting at `buf_start`; slices of `DIRECT_WRITE_THRESHOLD`
        // bytes or more skip it — flushing what is pending first — so large keys and
        // values go straight to memory without the copy.
        let mut buf = Vec::with_capacity(WRITE_BUFFER_INITIAL_CAPACITY);
        let mut buf_start = Address::from(0);

        // The header.
        buf.extend_from_slice(MAGIC);
        buf.push(LAYOUT_VERSION_2);
        buf.push(match self.node_type {
            NodeType::Leaf => LEAF_NODE_TYPE,
            NodeType::Internal => INTERNAL_NODE_TYPE,
        });
        buf.extend_from_slice(&(self.entries.len() as u16).to_le_bytes());

        // Flush at the overflow address, which lives at [7, 15) and is skipped over: it
        // is owned by the `NodeWriter`, which updates it in place as overflow pages are
        // allocated and released, and a write from here could clobber a chain pointer
        // the writer just installed. Never writing it from here is sufficient because
        // `finish` always leaves the field consistent: it writes the `NULL` terminator
        // into it whenever the node ends up with no overflow pages, so even a fresh
        // node at a reused address cannot retain a stale pointer.
        debug_assert_eq!(buf.len() as u64, OVERFLOW_ADDRESS_OFFSET.get());
        writer.write(buf_start, &buf);
        buf.clear();
        buf_start = Address::from(ENTRIES_OFFSET.get());

        // The children.
        for child in &self.children {
            buf.extend_from_slice(&child.get().to_le_bytes());
        }

        // The keys, sized unless fixed in size.
        for i in 0..self.entries.len() {
            let key_bytes = self.key(i, writer.memory()).to_bytes_checked();
            if !K::BOUND.is_fixed_size() {
                buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            }
            let key_bytes: &[u8] = key_bytes.borrow();
            if key_bytes.len() >= DIRECT_WRITE_THRESHOLD {
                buf_start = write_direct(&mut writer, buf_start, &mut buf, key_bytes);
            } else {
                buf.extend_from_slice(key_bytes);
            }
        }

        // The values, sized.
        for i in 0..self.entries.len() {
            let value = self.value(i, writer.memory());
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
            if value.len() >= DIRECT_WRITE_THRESHOLD {
                buf_start = write_direct(&mut writer, buf_start, &mut buf, value);
            } else {
                buf.extend_from_slice(value);
            }
        }

        if !buf.is_empty() {
            writer.write(buf_start, &buf);
        }

        self.overflows = writer.finish();
    }
}

/// Flushes the save buffer if it holds anything, writes `piece` directly after it, and
/// returns the new buffer start: just past `piece`.
fn write_direct<M: Memory>(
    writer: &mut NodeWriter<'_, M>,
    buf_start: Address,
    buf: &mut Vec<u8>,
    piece: &[u8],
) -> Address {
    let piece_start = buf_start + Bytes::from(buf.len() as u64);
    if !buf.is_empty() {
        writer.write(buf_start, buf);
        buf.clear();
    }
    writer.write(piece_start, piece);
    piece_start + Bytes::from(piece.len() as u64)
}

/// Walks the overflow-page chain starting from `first_overflow`, which the caller has
/// already read out of the node's initial page.
fn read_overflows<M: Memory>(first_overflow: Address, memory: &M) -> Vec<Address> {
    #[repr(C, packed)]
    struct OverflowPageHeader {
        magic: [u8; 3],
        next: Address,
    }

    let mut overflows = vec![];
    let mut overflow = first_overflow;
    while overflow != NULL {
        overflows.push(overflow);

        let header: OverflowPageHeader = read_struct(overflow, memory);
        assert_eq!(&header.magic, OVERFLOW_MAGIC, "Bad overflow magic.");
        overflow = header.next;
    }

    overflows
}
