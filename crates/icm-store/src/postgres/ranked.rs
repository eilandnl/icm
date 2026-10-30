//! Ranked (v2) search for the PostgreSQL backend: deep candidates per arm,
//! rank fusion (RRF), caller filter applied before the cut.
//!
//! Same contract as `SqliteStore::search_ranked`; see there for the full
//! description. What differs on PostgreSQL:
//!
//! - the lexical arm orders by `ts_rank_cd` over the generated `tsvector`
//!   column, not BM25, so `RawRelevance::bm25` is always `None` here;
//! - the vector arm is capped at [`VEC_MAX_DEPTH`]: an HNSW index scan
//!   returns at most `hnsw.ef_search` rows, whose ceiling is 1000;
//! - neither arm reads the embedding column. Final hits are hydrated in
//!   one extra query, so `keep` never sees an embedding.
//!
//! Read-only: no access bookkeeping here, the caller decides.

use super::*;

use icm_core::fusion::{RRF_K, RankedHit, RankedList, RankedQuery, RawRelevance, rrf_fuse};

/// Floor on candidates per arm, as on SQLite.
const MIN_DEPTH: usize = 20;
/// Ceiling on lexical candidates per arm.
const MAX_DEPTH: usize = 2000;
/// Ceiling on vector candidates: pgvector's `hnsw.ef_search` maximum.
const VEC_MAX_DEPTH: usize = 1000;
/// Each widening pass multiplies the depth by this much.
const WIDEN_FACTOR: usize = 4;
/// Widening passes after the first fetch.
const MAX_WIDENINGS: usize = 3;

/// [`SELECT_COLS`] with the embedding replaced by NULL: at depth 1000+
/// reading the vectors would only be to rank ids.
const ARM_COLS: &str = "id, created_at, updated_at, last_accessed, access_count, weight, topic, summary, raw_excerpt, keywords, importance, source_type, source_data, related_ids, NULL::vector AS embedding";

/// One arm's candidates that passed the caller's filter, best first.
struct Arm {
    kept: Vec<Memory>,
    /// Cosine similarity per kept id (vector arm only).
    similarity: HashMap<String, f32>,
    /// The arm returned as many rows as it was asked for: more may exist
    /// beyond `depth`.
    saturated: bool,
}

impl Arm {
    fn empty() -> Self {
        Self {
            kept: Vec::new(),
            similarity: HashMap::new(),
            saturated: false,
        }
    }
}

/// OR-joined `websearch_to_tsquery` text, so a natural-language question
/// matches on any of its words (the AND default would match nothing).
/// A bare `or` token is dropped: it would be read as the operator.
fn lexical_query(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if matches!(
                c,
                '-' | '*' | '"' | '(' | ')' | '{' | '}' | ':' | '^' | '+' | '~' | '\\'
            ) {
                ' '
            } else {
                c
            }
        })
        .collect();
    cleaned
        .split_whitespace()
        .filter(|w| !w.eq_ignore_ascii_case("or"))
        .take(100)
        .collect::<Vec<_>>()
        .join(" or ")
}

impl PostgresStore {
    /// Ranked search: lexical and vector arms fetched `q.depth` deep,
    /// filtered by `keep`, fused by Reciprocal Rank Fusion and cut to
    /// `q.limit`.
    ///
    /// - `q.depth` is clamped to `[20, 2000]` (vector arm: 1000).
    /// - Empty `q.text` gives the vector ranking; `q.embedding = None`, or
    ///   one whose length is not the store's dimension, gives the lexical
    ///   ranking.
    /// - `q.window` adds a third list, as on SQLite.
    /// - `keep` sees memories without their embedding: it must not test it.
    ///
    /// Writes nothing.
    pub fn search_ranked(
        &self,
        q: &RankedQuery<'_>,
        keep: &dyn Fn(&Memory) -> bool,
    ) -> IcmResult<Vec<RankedHit>> {
        if q.limit == 0 {
            return Ok(Vec::new());
        }

        let match_expr = lexical_query(q.text);
        let embedding = q.embedding.filter(|e| e.len() == self.embedding_dims);

        let mut depth = q.depth.clamp(MIN_DEPTH, MAX_DEPTH);
        let mut fts = self.ranked_fts_arm(&match_expr, depth, keep)?;
        let mut vec = self.ranked_vec_arm(embedding, depth, keep)?;

        // A filter can discard most of a saturated arm: fetch deeper rather
        // than return a short list while matching rows sit past `depth`.
        for _ in 0..MAX_WIDENINGS {
            if depth >= MAX_DEPTH
                || !(fts.saturated || vec.saturated)
                || union_len(&fts.kept, &vec.kept) >= q.limit
            {
                break;
            }
            depth = depth.saturating_mul(WIDEN_FACTOR).min(MAX_DEPTH);
            if fts.saturated {
                fts = self.ranked_fts_arm(&match_expr, depth, keep)?;
            }
            if vec.saturated {
                vec = self.ranked_vec_arm(embedding, depth, keep)?;
            }
        }

        let fts_ids: Vec<String> = fts.kept.iter().map(|m| m.id.clone()).collect();
        let vec_ids: Vec<String> = vec.kept.iter().map(|m| m.id.clone()).collect();

        let mut by_id: HashMap<String, Memory> =
            HashMap::with_capacity(fts_ids.len() + vec_ids.len());
        for memory in vec.kept {
            by_id.insert(memory.id.clone(), memory);
        }
        for memory in fts.kept {
            by_id.entry(memory.id.clone()).or_insert(memory);
        }

        let time_ids: Vec<String> = match q.window {
            Some(window) => {
                let vec_pos = positions(&vec_ids);
                let fts_pos = positions(&fts_ids);
                let mut inside: Vec<(usize, usize, &str)> = by_id
                    .values()
                    .filter(|m| window.contains(m.created_at))
                    .map(|m| {
                        (
                            vec_pos.get(m.id.as_str()).copied().unwrap_or(usize::MAX),
                            fts_pos.get(m.id.as_str()).copied().unwrap_or(usize::MAX),
                            m.id.as_str(),
                        )
                    })
                    .collect();
                inside.sort_unstable();
                inside
                    .into_iter()
                    .map(|(_, _, id)| id.to_string())
                    .collect()
            }
            None => Vec::new(),
        };

        let fused = rrf_fuse(
            &[
                RankedList {
                    name: "fts",
                    ids: &fts_ids,
                    weight: 1.0,
                },
                RankedList {
                    name: "vec",
                    ids: &vec_ids,
                    weight: 1.0,
                },
                RankedList {
                    name: "time",
                    ids: &time_ids,
                    weight: 1.0,
                },
            ],
            RRF_K,
        );

        let mut hits: Vec<RankedHit> = Vec::with_capacity(q.limit.min(fused.len()));
        for hit in fused.into_iter().take(q.limit) {
            let rank = |arm: usize| hit.ranks.get(arm).copied().flatten();
            if let Some(memory) = by_id.remove(&hit.id) {
                let raw = RawRelevance {
                    bm25: None,
                    similarity: vec.similarity.get(&hit.id).copied(),
                };
                hits.push(RankedHit {
                    memory,
                    score: hit.score,
                    fts_rank: rank(0),
                    vec_rank: rank(1),
                    time_rank: rank(2),
                    raw,
                });
            }
        }

        self.hydrate_embeddings(&mut hits)?;
        Ok(hits)
    }

    /// Lexical arm: `ts_rank_cd` order, at most `depth` rows, filtered by
    /// `keep`.
    fn ranked_fts_arm(
        &self,
        match_expr: &str,
        depth: usize,
        keep: &dyn Fn(&Memory) -> bool,
    ) -> IcmResult<Arm> {
        if match_expr.is_empty() {
            return Ok(Arm::empty());
        }

        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {ARM_COLS} \
                     FROM memories, websearch_to_tsquery('simple', $1) AS q \
                     WHERE fts @@ q \
                     ORDER BY ts_rank_cd(fts, q) DESC, id ASC \
                     LIMIT $2"
                ),
                &[&match_expr, &(depth as i64)],
            )
            .map_err(pg_err)?;
        let saturated = rows.len() >= depth;
        Ok(Arm {
            kept: rows
                .iter()
                .map(row_to_memory)
                .filter(|memory| keep(memory))
                .collect(),
            similarity: HashMap::new(),
            saturated,
        })
    }

    /// Vector arm: KNN order, at most `depth` rows (at most
    /// [`VEC_MAX_DEPTH`]), filtered by `keep`. No embedding = empty arm.
    fn ranked_vec_arm(
        &self,
        embedding: Option<&[f32]>,
        depth: usize,
        keep: &dyn Fn(&Memory) -> bool,
    ) -> IcmResult<Arm> {
        let Some(embedding) = embedding else {
            return Ok(Arm::empty());
        };

        let depth = depth.min(VEC_MAX_DEPTH);
        let qv = pgvector::Vector::from(embedding.to_vec());
        let mut c = self.conn()?;
        // An HNSW index scan stops at `ef_search` (default 40) rows whatever
        // the LIMIT says: raise it for this transaction only.
        let mut tx = c.transaction().map_err(pg_err)?;
        tx.batch_execute(&format!("SET LOCAL hnsw.ef_search = {VEC_MAX_DEPTH}"))
            .map_err(pg_err)?;
        let rows = tx
            .query(
                &format!(
                    "SELECT {ARM_COLS}, (embedding <=> $1)::float8 AS distance FROM memories \
                     WHERE embedding IS NOT NULL ORDER BY embedding <=> $1 LIMIT $2"
                ),
                &[&qv, &(depth as i64)],
            )
            .map_err(pg_err)?;
        tx.commit().map_err(pg_err)?;
        drop(c);

        // Nothing wider can be asked for once at the ceiling.
        let saturated = rows.len() >= depth && depth < VEC_MAX_DEPTH;

        let mut scored: Vec<(Memory, f32)> = rows
            .iter()
            .map(|row| {
                let distance: f64 = row.get(15);
                (row_to_memory(row), 1.0 - distance as f32)
            })
            .collect();
        // Similarity descending, NaN (zero-norm vector) last, ties on id.
        let key = |similarity: f32| {
            if similarity.is_nan() {
                f32::NEG_INFINITY
            } else {
                similarity
            }
        };
        scored.sort_by(|a, b| {
            key(b.1)
                .total_cmp(&key(a.1))
                .then_with(|| a.0.id.cmp(&b.0.id))
        });

        let similarity = scored
            .iter()
            .filter(|(_, similarity)| !similarity.is_nan())
            .map(|(memory, similarity)| (memory.id.clone(), *similarity))
            .collect();
        Ok(Arm {
            kept: scored
                .into_iter()
                .map(|(memory, _)| memory)
                .filter(|memory| keep(memory))
                .collect(),
            similarity,
            saturated,
        })
    }

    /// Load the stored embedding of the final hits, so a returned `Memory`
    /// equals what `get()` gives. `update()` writes `embedding` back as-is:
    /// a hit handed to it with a stripped `None` would erase the vector.
    fn hydrate_embeddings(&self, hits: &mut [RankedHit]) -> IcmResult<()> {
        if hits.is_empty() {
            return Ok(());
        }
        let ids: Vec<String> = hits.iter().map(|hit| hit.memory.id.clone()).collect();
        let mut c = self.conn()?;
        let rows = c
            .query(
                "SELECT id, embedding FROM memories \
                 WHERE id = ANY($1::text[]) AND embedding IS NOT NULL",
                &[&ids],
            )
            .map_err(pg_err)?;
        let mut stored: HashMap<String, Vec<f32>> = rows
            .iter()
            .map(|row| {
                let vector: pgvector::Vector = row.get(1);
                (row.get::<_, String>(0), vector.as_slice().to_vec())
            })
            .collect();
        for hit in hits {
            if let Some(embedding) = stored.remove(&hit.memory.id) {
                hit.memory.embedding = Some(embedding);
            }
        }
        Ok(())
    }
}

/// Position of each id in a ranked list.
fn positions(ids: &[String]) -> HashMap<&str, usize> {
    ids.iter()
        .enumerate()
        .map(|(pos, id)| (id.as_str(), pos))
        .collect()
}

/// Distinct memories across both arms.
fn union_len(fts: &[Memory], vec: &[Memory]) -> usize {
    let seen: HashSet<&str> = fts.iter().map(|m| m.id.as_str()).collect();
    seen.len() + vec.iter().filter(|m| !seen.contains(m.id.as_str())).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use icm_core::temporal::TimeWindow;

    const DIMS: usize = 64;

    #[test]
    fn arm_cols_is_select_cols_without_the_blob() {
        assert_eq!(
            ARM_COLS.replace("NULL::vector AS embedding", "embedding"),
            SELECT_COLS
        );
    }

    #[test]
    fn lexical_query_is_or_joined_and_operator_free() {
        assert_eq!(lexical_query("rust or go"), "rust or go");
        assert_eq!(
            lexical_query("sqlite-vec \"x\" (y)"),
            "sqlite or vec or x or y"
        );
        assert_eq!(lexical_query("  -- ** "), "");
        assert_eq!(lexical_query(""), "");
    }

    /// A store in a fresh database of the server named by
    /// `ICM_POSTGRES_TEST_URL` (`postgres://user@host:port/db`, a role
    /// that may create databases and the pgvector extension), or `None` to
    /// skip when it is not set. One database per test: the schema setup
    /// looks indexes up by name, so tests cannot share a database.
    fn store(name: &str) -> Option<PostgresStore> {
        let base = std::env::var("ICM_POSTGRES_TEST_URL").ok()?;
        let (server, _) = base.rsplit_once('/').expect("URL ends in /<database>");
        let database = format!("icm_test_{name}");
        let mut admin = Client::connect(&base, NoTls).expect("connect to test server");
        for statement in [
            format!("DROP DATABASE IF EXISTS {database} WITH (FORCE)"),
            format!("CREATE DATABASE {database}"),
        ] {
            admin
                .batch_execute(&statement)
                .expect("reset test database");
        }
        let url = format!("{server}/{database}");
        Some(PostgresStore::connect_url(&url, DIMS, false).expect("open test store"))
    }

    /// Unit vector on `axis`.
    fn axis(axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; DIMS];
        v[axis % DIMS] = 1.0;
        v
    }

    /// Vector whose cosine similarity to `axis(0)` strictly decreases as
    /// `i` grows.
    fn spread(i: usize) -> Vec<f32> {
        let mut v = axis(0);
        v[1 + i % (DIMS - 1)] = (i + 1) as f32 / 50.0;
        v
    }

    fn put(store: &PostgresStore, id: &str, topic: &str, summary: &str, emb: Option<Vec<f32>>) {
        let mut m = Memory::new(topic.into(), summary.into(), Importance::Medium);
        m.id = id.into();
        m.embedding = emb;
        store.store(m).expect("store memory");
    }

    fn query<'a>(
        text: &'a str,
        embedding: Option<&'a [f32]>,
        limit: usize,
        depth: usize,
    ) -> RankedQuery<'a> {
        RankedQuery {
            text,
            embedding,
            limit,
            depth,
            window: None,
        }
    }

    fn all(_: &Memory) -> bool {
        true
    }

    fn ids(hits: &[RankedHit]) -> Vec<&str> {
        hits.iter().map(|h| h.memory.id.as_str()).collect()
    }

    fn seed_150(store: &PostgresStore) {
        for i in 0..150 {
            put(
                store,
                &format!("m-{i:03}"),
                "notes",
                &format!("alpha entry {i}"),
                Some(spread(i)),
            );
        }
    }

    #[test]
    fn deep_limit_returns_more_than_forty() {
        let Some(store) = store("deep") else { return };
        seed_150(&store);
        let q_emb = axis(0);

        let hits = store
            .search_ranked(&query("alpha", Some(&q_emb), 100, 100), &all)
            .unwrap();
        assert_eq!(hits.len(), 100);
        assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));

        // Asked too shallow for the limit: widening fills it anyway.
        let hits = store
            .search_ranked(&query("alpha", Some(&q_emb), 100, 20), &all)
            .unwrap();
        assert_eq!(hits.len(), 100);

        // The whole base is reachable.
        let hits = store
            .search_ranked(&query("alpha", Some(&q_emb), 500, 2000), &all)
            .unwrap();
        assert_eq!(hits.len(), 150);
        assert_eq!(hits[0].memory.id, "m-000");
        assert_eq!(hits[0].score, 1.0);
    }

    /// An HNSW index scan stops at `hnsw.ef_search` (default 40) rows
    /// whatever the LIMIT says. Forcing the index shows the vector arm
    /// reaches past that. The scan is approximate, so the count is a lower
    /// bound, not the exact limit.
    #[test]
    fn vector_arm_is_deep_when_the_hnsw_index_is_used() {
        let Some(store) = store("hnsw") else { return };
        seed_150(&store);
        store
            .conn()
            .unwrap()
            .batch_execute("SET enable_seqscan = off")
            .unwrap();
        let q_emb = axis(0);

        let hits = store
            .search_ranked(&query("", Some(&q_emb), 100, 100), &all)
            .unwrap();
        assert!(hits.len() > 40, "got {} hits", hits.len());
        assert!(hits.iter().all(|h| h.vec_rank.is_some()));
    }

    #[test]
    fn memory_without_embedding_can_rank_first() {
        let Some(store) = store("noembed") else {
            return;
        };
        put(
            &store,
            "a-target",
            "notes",
            "zebra quartz migration checklist",
            None,
        );
        for i in 0..30 {
            put(
                &store,
                &format!("d-{i:02}"),
                "notes",
                &format!("grocery list item {i}"),
                Some(spread(i)),
            );
        }
        let q_emb = axis(0);

        let hits = store
            .search_ranked(&query("zebra quartz", Some(&q_emb), 10, 50), &all)
            .unwrap();

        assert_eq!(hits[0].memory.id, "a-target");
        assert_eq!(hits[0].fts_rank, Some(1));
        assert_eq!(hits[0].vec_rank, None);
        assert!(hits[0].memory.embedding.is_none());
        assert_eq!(hits[1].memory.id, "d-00");
        assert_eq!(hits[0].score, hits[1].score);
        assert!(hits[2..].iter().all(|h| h.score < hits[0].score));
    }

    #[test]
    fn filter_applies_before_the_cut() {
        let Some(store) = store("filter") else { return };
        for i in 0..60 {
            put(
                &store,
                &format!("o-{i:02}"),
                "other",
                &format!("deploy pipeline deploy pipeline rollout {i}"),
                Some(spread(i)),
            );
        }
        for i in 0..5 {
            put(
                &store,
                &format!("t-{i}"),
                "wanted",
                &format!("pipeline cleanup task {i} with a longer unrelated tail of words"),
                Some(spread(100 + i)),
            );
        }
        for i in 0..100 {
            put(
                &store,
                &format!("z-{i:03}"),
                "misc",
                &format!("unrelated background fact {i}"),
                None,
            );
        }
        let q_emb = axis(0);
        let wanted = |m: &Memory| m.topic == "wanted";

        // At the starting depth neither arm holds a memory of the topic.
        let expr = lexical_query("deploy pipeline");
        let fts = store.ranked_fts_arm(&expr, 20, &wanted).unwrap();
        let vec = store.ranked_vec_arm(Some(&q_emb), 20, &wanted).unwrap();
        assert!(fts.kept.is_empty() && fts.saturated);
        assert!(vec.kept.is_empty() && vec.saturated);

        let hits = store
            .search_ranked(&query("deploy pipeline", Some(&q_emb), 5, 20), &wanted)
            .unwrap();
        let mut got = ids(&hits);
        got.sort_unstable();
        assert_eq!(got, vec!["t-0", "t-1", "t-2", "t-3", "t-4"]);

        let none = store
            .search_ranked(&query("deploy pipeline", Some(&q_emb), 5, 20), &|_| false)
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn no_embedding_gives_lexical_order() {
        let Some(store) = store("lexical") else {
            return;
        };
        put(&store, "b-3", "notes", "rust filler filler", Some(axis(1)));
        put(&store, "b-1", "notes", "rust rust rust", None);
        put(&store, "b-2", "notes", "rust rust filler", Some(axis(2)));
        for i in 0..7 {
            put(
                &store,
                &format!("f-{i}"),
                "notes",
                &format!("filler filler other{i}"),
                Some(axis(3 + i)),
            );
        }

        let hits = store
            .search_ranked(&query("rust", None, 10, 50), &all)
            .unwrap();
        assert_eq!(ids(&hits), vec!["b-1", "b-2", "b-3"]);
        assert_eq!(hits[0].score, 1.0);
        for (pos, h) in hits.iter().enumerate() {
            assert_eq!(h.fts_rank, Some(pos as u32 + 1));
            assert_eq!(h.vec_rank, None);
            assert_eq!(h.raw.bm25, None);
        }

        // An empty slice is "no embedding".
        let empty: [f32; 0] = [];
        let same = store
            .search_ranked(&query("rust", Some(&empty), 10, 50), &all)
            .unwrap();
        assert_eq!(ids(&same), ids(&hits));
    }

    #[test]
    fn empty_text_gives_vector_order() {
        let Some(store) = store("vector") else { return };
        for i in [3_usize, 0, 4, 1, 2] {
            put(
                &store,
                &format!("v-{i}"),
                "notes",
                &format!("note number {i}"),
                Some(spread(i * 10)),
            );
        }
        put(&store, "v-none", "notes", "note without vector", None);
        let q_emb = axis(0);

        for text in ["", "   ", "-- ** \"\" ()"] {
            let hits = store
                .search_ranked(&query(text, Some(&q_emb), 10, 50), &all)
                .unwrap();
            assert_eq!(
                ids(&hits),
                vec!["v-0", "v-1", "v-2", "v-3", "v-4"],
                "text = {text:?}"
            );
            assert_eq!(hits[0].score, 1.0);
            assert!(hits.iter().all(|h| h.fts_rank.is_none()));
            assert!(hits.iter().all(|h| h.raw.similarity.is_some()));
        }

        let nothing = store.search_ranked(&query("", None, 10, 50), &all).unwrap();
        assert!(nothing.is_empty());
    }

    #[test]
    fn window_lifts_a_memory_dated_inside() {
        let Some(store) = store("window") else { return };
        let january = chrono::Utc.with_ymd_and_hms(2024, 1, 5, 9, 0, 0).unwrap();
        for i in 0..6_usize {
            let created_at = match i {
                3 => chrono::Utc.with_ymd_and_hms(2024, 3, 10, 12, 0, 0).unwrap(),
                5 => chrono::Utc.with_ymd_and_hms(2024, 3, 20, 12, 0, 0).unwrap(),
                _ => january,
            };
            let mut m = Memory::new(
                "notes".into(),
                format!("sprint review notes {i}"),
                Importance::Medium,
            );
            m.id = format!("w-{i}");
            m.embedding = Some(spread(i * 10));
            m.created_at = created_at;
            store.store(m).unwrap();
        }
        let q_emb = axis(0);
        let base = query("sprint review", Some(&q_emb), 10, 50);

        let plain = store.search_ranked(&base, &all).unwrap();
        assert_eq!(ids(&plain), vec!["w-0", "w-1", "w-2", "w-3", "w-4", "w-5"]);
        assert!(plain.iter().all(|h| h.time_rank.is_none()));

        let march = TimeWindow {
            start: chrono::Utc.with_ymd_and_hms(2024, 3, 1, 0, 0, 0).unwrap(),
            end: chrono::Utc.with_ymd_and_hms(2024, 4, 1, 0, 0, 0).unwrap(),
        };
        let windowed = store
            .search_ranked(
                &RankedQuery {
                    window: Some(march),
                    ..base
                },
                &all,
            )
            .unwrap();
        assert_eq!(
            ids(&windowed),
            vec!["w-3", "w-5", "w-0", "w-1", "w-2", "w-4"]
        );
        assert_eq!(windowed[0].time_rank, Some(1));
        assert_eq!(windowed[1].time_rank, Some(2));
        assert!(windowed[2..].iter().all(|h| h.time_rank.is_none()));
    }

    #[test]
    fn identical_calls_give_identical_order() {
        let Some(store) = store("determinism") else {
            return;
        };
        seed_150(&store);
        let q_emb = axis(0);
        let q = query("alpha entry", Some(&q_emb), 100, 60);

        let first = store.search_ranked(&q, &all).unwrap();
        let second = store.search_ranked(&q, &all).unwrap();
        assert_eq!(first.len(), 100);
        assert_eq!(ids(&first), ids(&second));
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.score, b.score);
        }
    }

    /// A query vector of the wrong dimension must degrade to the lexical
    /// ranking, not fail the search on pgvector's dimension error.
    #[test]
    fn wrong_dimension_query_vector_falls_back_to_the_lexical_ranking() {
        let Some(store) = store("wrongdim") else {
            return;
        };
        seed_150(&store);
        let wrong = vec![0.5_f32; DIMS / 2];

        let hits = store
            .search_ranked(&query("alpha", Some(&wrong), 10, 50), &all)
            .expect("a mismatched query vector must not fail the search");
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|h| h.vec_rank.is_none()));

        let lexical = store
            .search_ranked(&query("alpha", None, 10, 50), &all)
            .unwrap();
        assert_eq!(ids(&hits), ids(&lexical));
    }

    #[test]
    fn depth_is_clamped_and_zero_limit_is_empty() {
        let Some(store) = store("depth") else { return };
        seed_150(&store);
        let q_emb = axis(0);

        for depth in [0, 1, usize::MAX] {
            let hits = store
                .search_ranked(&query("alpha", Some(&q_emb), 10, depth), &all)
                .unwrap();
            assert_eq!(hits.len(), 10, "depth = {depth}");
            assert!(hits.iter().any(|h| h.vec_rank.is_some()));
            assert!(hits.iter().any(|h| h.fts_rank.is_some()));
        }

        let none = store
            .search_ranked(&query("alpha", Some(&q_emb), 0, 50), &all)
            .unwrap();
        assert!(none.is_empty());
    }

    /// `update()` writes `embedding` back as-is, so a hit must carry the
    /// stored vector even though neither arm reads it.
    #[test]
    fn hits_keep_their_stored_embedding() {
        let Some(store) = store("hydrate") else {
            return;
        };
        put(&store, "h-1", "notes", "kiwi harvest plan", Some(spread(1)));
        put(&store, "h-2", "notes", "kiwi storage plan", None);

        let hits = store
            .search_ranked(&query("kiwi", None, 10, 50), &all)
            .unwrap();
        assert_eq!(hits.len(), 2);
        for h in &hits {
            let stored = store.get(&h.memory.id).unwrap().unwrap();
            assert_eq!(h.memory.embedding, stored.embedding, "{}", h.memory.id);
        }
        let with_vector = hits.iter().find(|h| h.memory.id == "h-1").unwrap();
        assert_eq!(
            with_vector.memory.embedding.as_deref(),
            Some(&spread(1)[..])
        );
    }

    #[test]
    fn search_writes_nothing() {
        let Some(store) = store("readonly") else {
            return;
        };
        put(
            &store,
            "r-1",
            "notes",
            "mango ripening log",
            Some(spread(1)),
        );
        let before = store.get("r-1").unwrap().unwrap();

        let q_emb = axis(0);
        let hits = store
            .search_ranked(&query("mango", Some(&q_emb), 5, 50), &all)
            .unwrap();
        assert_eq!(ids(&hits), vec!["r-1"]);

        let after = store.get("r-1").unwrap().unwrap();
        assert_eq!(after.access_count, before.access_count);
        assert_eq!(after.last_accessed, before.last_accessed);
    }

    #[test]
    fn query_with_operators_is_not_an_error() {
        let Some(store) = store("operators") else {
            return;
        };
        put(&store, "s-1", "notes", "sqlite vec extension notes", None);

        for text in [
            "sqlite-vec",
            "\"sqlite",
            "vec AND (",
            "notes: ^vec*",
            "or",
            "vec or",
        ] {
            let hits = store
                .search_ranked(&query(text, None, 5, 50), &all)
                .unwrap();
            let expect_hit = text != "or";
            assert_eq!(hits.len(), usize::from(expect_hit), "text = {text:?}");
        }
    }
}
