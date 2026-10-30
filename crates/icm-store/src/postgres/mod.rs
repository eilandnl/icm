//! PostgreSQL storage backend (issue #301, opt-in via `--features postgres`).
//!
//! A node-local SQLite file cannot be shared between several ICM
//! processes or Kubernetes replicas. This backend runs the same memory
//! model over a network-accessible PostgreSQL database so every instance
//! reads and writes one shared store. PostgreSQL serialises concurrent
//! writers, so N replicas can `icm store` into the same memory safely.
//!
//! Design notes:
//!
//! - **Blocking client.** The store traits are synchronous
//!   (`fn store(&self, ...) -> IcmResult<...>`), so we use the blocking
//!   `postgres` crate. No async runtime, no sync-over-async bridge — the
//!   client maps one-to-one onto the trait surface.
//! - **`pgvector` for embeddings.** Memory embeddings live in a
//!   `vector(N)` column; KNN search uses the `<=>` cosine-distance
//!   operator. Similarity is reported as `1 - distance` to match the
//!   SQLite backend.
//! - **PostgreSQL full-text search** replaces SQLite FTS5: a generated
//!   `tsvector` column (config `simple`, no stemming, to mirror FTS5's
//!   unicode61 tokenizer) with a GIN index, queried via
//!   `websearch_to_tsquery` so arbitrary user input is operator-safe.
//! - **Connection string** comes from `ICM_POSTGRES_URL` (or
//!   `DATABASE_URL` as a fallback). The `&Path` arguments that the CLI
//!   passes for the SQLite file are ignored.
//!
//! Scope of this first cut: the full [`MemoryStore`] surface (the core
//! shared-memory use case behind #301) plus the ancillary tables used by
//! the normal store/recall/hook path (hook telemetry, the extraction
//! queue, code areas, the key/value metadata), structured facts, feedback,
//! transcripts, the memoir graph and pattern mining. What stays
//! SQLite-only are the operations on the SQLite file itself (integrity
//! check, search-index rebuild, file backup, vector-index reset).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use chrono::{DateTime, Utc};
use postgres::types::ToSql;
use postgres::{Client, GenericClient, NoTls};

use icm_core::{
    Concept, ConceptLink, Embedder, Fact, FactsStats, FactsStore, Feedback, FeedbackStats,
    FeedbackStore, IcmError, IcmResult, Importance, Label, Memoir, MemoirStats, MemoirStore,
    Memory, MemorySource, MemoryStore, Message, PatternCluster, Relation, Role, Session,
    StoreStats, TopicHealth, TranscriptHit, TranscriptStats, TranscriptStore,
};

// Shared public row types live in `crate::common` (issue #301) so every
// backend can be compiled into one binary without colliding definitions.
pub use crate::common::{
    CodeArea, ConsolidationJob, HookEvent, HookEventInsert, HookStatsRow, PendingRow,
};

// Helpers (mirrored from the SQLite backend so behaviour matches)

// PostgresStore

/// PostgreSQL-backed store. See the module docs.
pub struct PostgresStore {
    client: Mutex<Client>,
    /// Dimension of the `vector(N)` columns. Changes when the vector index
    /// is reset for another embedding model.
    embedding_dims: AtomicUsize,
    readonly: bool,
}

mod connection;
mod facts;
mod feedback;
mod hooks;
mod maintenance;
mod memoir;
mod memory;
mod patterns;
mod ranked;
mod rows;
#[cfg(test)]
mod tests;
mod transcript;

pub(crate) use rows::*;
