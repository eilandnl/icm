# PostgreSQL backend (opt-in)

By default ICM stores memory in a node-local SQLite file. That is perfect
for a single machine, but a local file cannot be shared between several
ICM processes or between Kubernetes replicas, so memory can't be shared
across instances.

The **PostgreSQL backend** (issue #301) runs the same memory model over a
network-accessible PostgreSQL database. Every instance reads and writes
one shared store, and PostgreSQL serialises concurrent writers, so N
replicas can `icm store` into the same memory with no lost rows.

It is **opt-in** and selected at build time via a Cargo feature. The
default build is unchanged: SQLite only, single binary, zero external
services.

## What it covers

This first cut implements the **core memory surface** — exactly the part
that benefits from being shared:

- `store` (with dedup + metadata merge), `get`, `update`, `forget`
- keyword search, full-text search (PostgreSQL `tsvector` + GIN),
  vector KNN (`pgvector`, cosine), and the hybrid blend (30% FTS / 70%
  vector)
- ranked recall (`search_ranked`): a lexical and a vector arm fetched deep,
  filtered, fused by Reciprocal Rank Fusion, with the optional time
  window. The lexical arm ranks by BM25 computed in SQL (k1 1.2, b 0.75,
  as FTS5 does), because PostgreSQL's `ts_rank_cd` does not weigh how rare
  a term is; the scores follow the same order as SQLite's but are not the
  same numbers. The vector arm reaches at most 1000 candidates (pgvector's
  `hnsw.ef_search` ceiling), and an HNSW scan is approximate
- full-text indexing splits words on punctuation, as SQLite's does:
  PostgreSQL's parser keeps `src/lib.rs`, `file_name.py` and
  `state-of-the-art` as one token each, which left about one memory in six
  unfindable by two of its own words. The `memories` column is rebuilt once
  on the first start after upgrading (4 s for 8,000 memories); other
  tables are created with it. The statistics BM25 needs are computed per
  query. On a large table the average length is sampled, and a word that is
  in more than 5% of the memories (and in more than 1,000) is left out of
  the candidates as a stop word would be; a query of only such words is
  ranked by `ts_rank_cd` and has no BM25 score
- `list`, `topics`, `stats`, `health`
- temporal `decay` (access-aware) and `prune`
- auto-consolidation (`[memory] auto_consolidate_enabled`): the same
  lossy rollup as SQLite, safe for concurrent writers because it replaces
  only the memories it read
- the ancillary tables used by the normal store/recall/hook path:
  hook telemetry, the async extraction queue, code areas, and the
  key/value metadata.

- structured facts (`icm facts …`): one active value per `(entity, key)`
  with the superseded values kept as history; writers on the same slot
  take turns, so replicas can set facts concurrently
- feedback (`icm feedback …`): corrections of past predictions, searched by
  a 30% full-text / 70% semantic blend. The semantic part is an exact
  scan with no vector index, as feedback is low-cardinality

- verbatim transcripts (`icm transcript …`): sessions and messages, with
  the same size caps as SQLite. Search is PostgreSQL full-text search read
  as `word word` (all must match), `word or word`, `"exact phrase"` and
  `-word`; it is not FTS5 syntax, so `AND`, `NOT`, `NEAR`, `prefix*` and
  parentheses are not understood, and a malformed query matches loosely
  instead of returning nothing. Results rank by BM25, as above

- memoirs (`icm memoir …`): named memoirs of concepts with typed links that
  must stay acyclic, concept refinement, and search by words or by label.
  A neighbourhood is walked in one recursive query instead of one query per
  concept, link inserts take turns so two replicas cannot each add half of
  a cycle, and a label search matches one whole label, so `domain` of one
  label and `rust` of another never combine as they can on SQLite

- pattern mining (`icm patterns …`): clusters of memories with overlapping
  keywords, and extracting a cluster as a concept
- the key/value metadata (`icm_metadata`), which the hooks use to remember
  how far into a conversation they have already read

## What stays SQLite-only

The file backup and its rotation copy the SQLite file, so on PostgreSQL the
automatic backup does not run and `icm repair` does not take a copy. Back
the database up with `pg_dump`, point-in-time recovery or your platform's
snapshots.

## Integrity check and repair

`icm repair` checks what ICM depends on: that its tables and indexes exist
and are valid (an index left invalid by a failed `CREATE INDEX
CONCURRENTLY` is reported), and that the recorded `embedding_dims` is the
size the vector columns have. It does not look for damage in PostgreSQL's
own pages, which its write-ahead log and checksums guard. The repair
recreates whatever is missing and runs `REINDEX` on the tables with a
search index; it does not touch the data. Opening for the check does not
run the usual schema migration, since that would recreate the things the
check is looking for.

## Changing the embedding model

The `vector(N)` columns are sized when the database is created, and the
stored size wins over what a later run asks for. To move to a model with
another dimension, configure the new model and run `icm embed --migrate`.
It clears every stored vector (they are derived data), resizes the vector
columns of `memories` and `feedback`, recreates the HNSW index, records the
new size and model, all in one transaction, and then computes the vectors
again; an interrupted run is resumed by a plain `icm embed`. It takes no
file backup on PostgreSQL, so take a database snapshot first (`pg_dump`,
or your platform's) if you want to keep the old vectors.

Other ICM processes keep checking against the old size until they restart,
and PostgreSQL rejects a vector of the wrong size with its own error.

## TLS

The connection follows the `sslmode` of the connection string: `disable`
never uses TLS, the default (`prefer`) tries TLS and falls back to plain text
when the server has none, and `require` fails when the server offers none.
Whenever TLS is used the server certificate and its host name are verified,
so `require` is stricter than libpq's (which skips verification) and there is
no way to turn verification off. Roots are the system store plus, optionally,
the PEM file in `ICM_POSTGRES_CA_FILE`, for a private CA or a provider's own
bundle (for example RDS):

```sh
export ICM_POSTGRES_URL="postgres://user:pass@db.example.com:5432/icm?sslmode=require"
export ICM_POSTGRES_CA_FILE=/etc/ssl/db-ca.pem
```

Client certificates are not supported. If a host name resolves to both IPv6
and IPv4 and only one of them works, the error shown is the last attempt's
(often "Connection refused"), which can hide a certificate failure; connect
by IP or fix the DNS to see it.

## Known limits

- **TLS is verified, not configurable beyond the CA.** See "TLS" below.
- **Every process runs the schema setup when it starts**: 27 round trips,
  about 30 ms on localhost. Hooks start a process on each tool call, so at
  a 20 ms round trip to the database that is about half a second per hook.
  Keep the database close, or use the MCP server, which stays running.
- **One connection per process**, without a pool.
- **Recall time grows with the table.** On 101,000 memories (debug build,
  localhost, everything above included): a rare word 85 ms, five ordinary
  words 145 ms, a 15-word sentence 245 ms. Without leaving out common words
  the same five words took 0.9 s.
- **Recall quality is measured, not assumed.** On a copy of a real database
  of 7,788 memories, a query made of a memory's five rarest words found
  that memory first in 94% of cases against 96% on SQLite, with the same top
  result in 97% of queries; a query made of its keywords found it first in
  70% against 63%. With embeddings (a small model, 40 memories, 20
  paraphrase queries) both backends returned the same top result for all 20.
  It was not measured on other corpora, or with a vector index larger than a
  few thousand rows.
- **Versions.** The Postgres tests pass on PostgreSQL 15, 16 and 17 with
  pgvector 0.6.0 and 0.8.7; CI runs `pgvector/pgvector:pg16`. `icm embed
  --migrate` was run end to end: reset, re-embed, and recall unchanged.

## Requirements

PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector)
extension available (the backend runs `CREATE EXTENSION IF NOT EXISTS
vector`). The `pgvector/pgvector` images and Azure Database for
PostgreSQL Flexible Server (with `vector` allow-listed) both work.

## Build

The **default `icm` binary already includes this backend** (backends are
additive and selected at runtime, like SurrealDB's `Surreal<Any>`). Nothing
to build — just configure it. For a lean SQLite-only binary, build with
`--no-default-features --features "embeddings,tui,http-api,backend-sqlite"`.

## Configure

Select the backend at runtime and give it a connection string:

```sh
export ICM_DB_BACKEND=postgres
export ICM_POSTGRES_URL="postgres://user:pass@host:5432/icm"
# DATABASE_URL is accepted as a fallback for the URL.
```

With `ICM_DB_BACKEND` unset (or `sqlite`) the binary uses the local SQLite
file as before. The `--db` flag (a SQLite file path) is ignored by this
backend.

The schema — including the `vector(N)` embedding column whose dimension
`N` matches your embedder — is created on first connect. The stored
dimension is authoritative afterwards, so always initialise a database
with the embedder you intend to use (or `--no-embeddings` for
keyword/FTS-only).

## Verify

```sh
docker run -d --name icm-pg \
    -e POSTGRES_PASSWORD=icm -e POSTGRES_USER=icm -e POSTGRES_DB=icm \
    -p 55432:5432 pgvector/pgvector:pg16

export ICM_DB_BACKEND=postgres
export ICM_POSTGRES_URL="postgres://icm:icm@127.0.0.1:55432/icm"

icm store -t demo -c "PostgreSQL backend shares memory across replicas" -i high
icm recall "shared memory"
icm stats

# Integration tests (skipped automatically when the env var is unset):
cargo test -p icm-store --no-default-features --features postgres -- --test-threads=1

# Store tests (ranked search, auto-consolidation): each one creates its own
# database on this server, so the role needs CREATEDB and a separate
# variable guards against pointing them at a database you care about.
ICM_POSTGRES_TEST_URL="postgres://icm:icm@127.0.0.1:55432/icm" \
    cargo test -p icm-store --no-default-features --features postgres \
    --lib postgres:: -- --test-threads=1
```

## Kubernetes

A network backend is what makes a horizontally-scaled deployment share
memory. Point every replica's `ICM_POSTGRES_URL` at the same PostgreSQL
service and they read/write one store. See
[`deploy/k8s`](../deploy/k8s) for a minimal manifest set used to validate
this on a real cluster.
