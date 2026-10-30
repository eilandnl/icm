//! PostgreSQL backend: feedback (corrections of past AI predictions).
//!
//! Same model as the SQLite backend. Search blends full-text (30%) and
//! semantic similarity (70%) like `search_hybrid`; the semantic arm is an
//! exact scan, no vector index: feedback is low-cardinality.

use super::*;

pub(super) fn schema(dims: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            context TEXT NOT NULL,
            predicted TEXT NOT NULL,
            corrected TEXT NOT NULL,
            reason TEXT,
            source TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL,
            applied_count INTEGER NOT NULL DEFAULT 0,
            embedding vector({dims}),
            fts tsvector GENERATED ALWAYS AS (
                to_tsvector('simple',
                    topic || ' ' || context || ' ' || predicted || ' ' || corrected
                    || ' ' || coalesce(reason, ''))
            ) STORED
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_topic ON feedback(topic);
        CREATE INDEX IF NOT EXISTS idx_feedback_fts ON feedback USING GIN (fts);"
    )
}

const FEEDBACK_COLS: &str = "id, topic, context, predicted, corrected, reason, source, \
     created_at, applied_count, embedding";

fn row_to_feedback(row: &postgres::Row) -> Feedback {
    let applied_count: i32 = row.get(8);
    Feedback {
        id: row.get(0),
        topic: row.get(1),
        context: row.get(2),
        predicted: row.get(3),
        corrected: row.get(4),
        reason: row.get(5),
        source: row.get(6),
        created_at: row.get(7),
        applied_count: applied_count.max(0) as u32,
        embedding: row
            .get::<_, Option<pgvector::Vector>>(9)
            .map(|v| v.as_slice().to_vec()),
    }
}

/// AND-joined words of `text`, as `websearch_to_tsquery` text: every word
/// must match, like the SQLite backend. Anything but letters and digits
/// separates words, and a bare `or` is dropped (it would be the operator).
fn and_query(text: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !w.eq_ignore_ascii_case("or"))
        .take(100)
        .collect::<Vec<_>>()
        .join(" ")
}

impl FeedbackStore for PostgresStore {
    fn store_feedback(&self, feedback: Feedback) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("store_feedback".into()));
        }
        if let Some(embedding) = &feedback.embedding
            && embedding.len() != self.embedding_dims
        {
            return Err(IcmError::InvalidInput(format!(
                "embedding has {} dimensions, but this store uses {}",
                embedding.len(),
                self.embedding_dims
            )));
        }
        let embedding = feedback
            .embedding
            .as_ref()
            .map(|e| pgvector::Vector::from(e.clone()));
        let applied = feedback.applied_count as i32;
        let mut c = self.conn()?;
        c.execute(
            "INSERT INTO feedback (id, topic, context, predicted, corrected, reason, source, \
                                   created_at, applied_count, embedding) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            &[
                &feedback.id,
                &feedback.topic,
                &feedback.context,
                &feedback.predicted,
                &feedback.corrected,
                &feedback.reason,
                &feedback.source,
                &feedback.created_at,
                &applied,
                &embedding,
            ],
        )
        .map_err(pg_err)?;
        Ok(feedback.id)
    }

    fn search_feedback(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
        topic: Option<&str>,
        limit: usize,
    ) -> IcmResult<Vec<Feedback>> {
        let text = and_query(query);
        if text.is_empty() {
            return self.list_feedback(topic, limit);
        }

        let pool = limit.saturating_mul(4).max(limit).min(1000) as i64;
        let mut c = self.conn()?;

        let mut all: HashMap<String, Feedback> = HashMap::new();
        let mut fts_scores: HashMap<String, f32> = HashMap::new();
        let rows = c
            .query(
                &format!(
                    "SELECT {FEEDBACK_COLS}, ts_rank_cd(fts, q)::float8 \
                     FROM feedback, websearch_to_tsquery('simple', $1) AS q \
                     WHERE fts @@ q AND ($2::text IS NULL OR topic = $2) \
                     ORDER BY ts_rank_cd(fts, q) DESC, id ASC LIMIT $3"
                ),
                &[&text, &topic, &pool],
            )
            .map_err(pg_err)?;
        for row in &rows {
            let rank: f64 = row.get(10);
            // Monotonic in relevance, in [0, 1): the same transform the
            // SQLite backend applies to its BM25 rank.
            let rank = rank.abs() as f32;
            let feedback = row_to_feedback(row);
            fts_scores.insert(feedback.id.clone(), rank / (1.0 + rank));
            all.insert(feedback.id.clone(), feedback);
        }

        let mut vec_scores: HashMap<String, f32> = HashMap::new();
        if let Some(embedding) = query_embedding.filter(|e| e.len() == self.embedding_dims) {
            let qv = pgvector::Vector::from(embedding.to_vec());
            let rows = c
                .query(
                    &format!(
                        "SELECT {FEEDBACK_COLS}, (embedding <=> $1)::float8 FROM feedback \
                         WHERE embedding IS NOT NULL AND ($2::text IS NULL OR topic = $2) \
                         ORDER BY embedding <=> $1, id ASC LIMIT $3"
                    ),
                    &[&qv, &topic, &pool],
                )
                .map_err(pg_err)?;
            for row in &rows {
                let similarity = 1.0 - row.get::<_, f64>(10) as f32;
                if similarity > 0.0 {
                    let feedback = row_to_feedback(row);
                    vec_scores.insert(feedback.id.clone(), similarity);
                    all.entry(feedback.id.clone()).or_insert(feedback);
                }
            }
        }

        let mut scored: Vec<(String, f32)> = if query_embedding.is_none() {
            fts_scores.iter().map(|(id, s)| (id.clone(), *s)).collect()
        } else {
            all.keys()
                .map(|id| {
                    let fts = fts_scores.get(id).copied().unwrap_or(0.0);
                    let vec = vec_scores.get(id).copied().unwrap_or(0.0);
                    (id.clone(), 0.3 * fts + 0.7 * vec)
                })
                .collect()
        };
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(limit);
        Ok(scored
            .into_iter()
            .filter_map(|(id, _)| all.remove(&id))
            .collect())
    }

    fn list_feedback(&self, topic: Option<&str>, limit: usize) -> IcmResult<Vec<Feedback>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {FEEDBACK_COLS} FROM feedback \
                     WHERE ($1::text IS NULL OR topic = $1) \
                     ORDER BY created_at DESC, id DESC LIMIT $2"
                ),
                &[&topic, &sql_bound(limit)],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_feedback).collect())
    }

    fn increment_applied(&self, id: &str) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("increment_applied".into()));
        }
        let mut c = self.conn()?;
        let changed = c
            .execute(
                "UPDATE feedback SET applied_count = applied_count + 1 WHERE id = $1",
                &[&id],
            )
            .map_err(pg_err)?;
        if changed == 0 {
            return Err(IcmError::NotFound(id.to_string()));
        }
        Ok(())
    }

    fn delete_feedback(&self, id: &str) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("delete_feedback".into()));
        }
        let mut c = self.conn()?;
        let changed = c
            .execute("DELETE FROM feedback WHERE id = $1", &[&id])
            .map_err(pg_err)?;
        if changed == 0 {
            return Err(IcmError::NotFound(id.to_string()));
        }
        Ok(())
    }

    fn feedback_stats(&self) -> IcmResult<FeedbackStats> {
        let mut c = self.conn()?;
        let total: i64 = c
            .query_one("SELECT COUNT(*)::bigint FROM feedback", &[])
            .map_err(pg_err)?
            .get(0);
        let by_topic = c
            .query(
                "SELECT topic, COUNT(*)::bigint AS cnt FROM feedback \
                 GROUP BY topic ORDER BY cnt DESC, topic COLLATE \"C\" ASC",
                &[],
            )
            .map_err(pg_err)?;
        let most_applied = c
            .query(
                "SELECT id, applied_count FROM feedback WHERE applied_count > 0 \
                 ORDER BY applied_count DESC, id ASC LIMIT 10",
                &[],
            )
            .map_err(pg_err)?;
        Ok(FeedbackStats {
            total: total.max(0) as usize,
            by_topic: by_topic
                .iter()
                .map(|row| (row.get(0), row.get::<_, i64>(1).max(0) as usize))
                .collect(),
            most_applied: most_applied
                .iter()
                .map(|row| (row.get(0), row.get::<_, i32>(1).max(0) as u32))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::tests::live_store;

    const DIMS: usize = 64;

    fn feedback(topic: &str, context: &str, corrected: &str) -> Feedback {
        Feedback::new(
            topic.into(),
            context.into(),
            "wrong answer".into(),
            corrected.into(),
            None,
            "test".into(),
        )
    }

    fn axis(axis: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; DIMS];
        v[axis % DIMS] = 1.0;
        v
    }

    #[test]
    fn and_query_keeps_words_and_drops_operators() {
        assert_eq!(
            and_query("deploy-pipeline \"x\" or (y)"),
            "deploy pipeline x y"
        );
        assert_eq!(and_query("  -- ** "), "");
    }

    #[test]
    fn store_list_and_delete_round_trip() {
        let Some(store) = live_store("fb_crud", DIMS) else {
            return;
        };
        let mut first = feedback("ops", "deploy failed", "use the staging key");
        first.reason = Some("key was rotated".into());
        first.embedding = Some(axis(0));
        let id = store.store_feedback(first.clone()).unwrap();
        store
            .store_feedback(feedback("code", "lint failed", "run fmt first"))
            .unwrap();

        let ops = store.list_feedback(Some("ops"), 10).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].id, id);
        assert_eq!(ops[0].reason.as_deref(), Some("key was rotated"));
        assert_eq!(ops[0].embedding.as_deref(), Some(&axis(0)[..]));
        assert_eq!(store.list_feedback(None, 10).unwrap().len(), 2);
        assert_eq!(store.list_feedback(None, 1).unwrap().len(), 1);

        store.delete_feedback(&id).unwrap();
        assert!(store.list_feedback(Some("ops"), 10).unwrap().is_empty());
        assert!(matches!(
            store.delete_feedback(&id),
            Err(IcmError::NotFound(_))
        ));
    }

    #[test]
    fn a_wrong_embedding_dimension_is_rejected() {
        let Some(store) = live_store("fb_dims", DIMS) else {
            return;
        };
        let mut f = feedback("ops", "c", "x");
        f.embedding = Some(vec![0.1; DIMS / 2]);
        assert!(matches!(
            store.store_feedback(f),
            Err(IcmError::InvalidInput(_))
        ));
    }

    #[test]
    fn increment_applied_counts_and_feeds_stats() {
        let Some(store) = live_store("fb_stats", DIMS) else {
            return;
        };
        let a = store.store_feedback(feedback("ops", "a", "x")).unwrap();
        let b = store.store_feedback(feedback("ops", "b", "y")).unwrap();
        store.store_feedback(feedback("code", "c", "z")).unwrap();
        for _ in 0..3 {
            store.increment_applied(&a).unwrap();
        }
        store.increment_applied(&b).unwrap();
        assert!(matches!(
            store.increment_applied("missing"),
            Err(IcmError::NotFound(_))
        ));

        let stats = store.feedback_stats().unwrap();
        assert_eq!(stats.total, 3);
        assert_eq!(
            stats.by_topic,
            vec![("ops".to_string(), 2), ("code".to_string(), 1)]
        );
        assert_eq!(stats.most_applied, vec![(a, 3), (b, 1)]);
    }

    #[test]
    fn search_is_and_joined_topic_scoped_and_ranked_by_relevance() {
        let Some(store) = live_store("fb_search", DIMS) else {
            return;
        };
        let strong = store
            .store_feedback(feedback(
                "ops",
                "kubernetes deploy failed",
                "kubernetes deploy needs the staging key",
            ))
            .unwrap();
        store
            .store_feedback(feedback("ops", "kubernetes upgrade", "check the changelog"))
            .unwrap();
        store
            .store_feedback(feedback("code", "kubernetes deploy lint", "run fmt"))
            .unwrap();

        // Every word must match: only the two "deploy" entries.
        let both = store
            .search_feedback("kubernetes deploy", None, None, 10)
            .unwrap();
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].id, strong, "more relevant first");

        let scoped = store
            .search_feedback("kubernetes deploy", None, Some("code"), 10)
            .unwrap();
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].topic, "code");

        // A query with no searchable word lists instead.
        assert_eq!(
            store
                .search_feedback("-- **", None, None, 10)
                .unwrap()
                .len(),
            3
        );
    }

    /// An entry missing the query's exact words is still found through its
    /// embedding, and a wrong-sized query vector is ignored, not an error.
    #[test]
    fn search_falls_back_to_semantic_similarity() {
        let Some(store) = live_store("fb_semantic", DIMS) else {
            return;
        };
        let mut near = feedback("ops", "rotating credentials", "use the vault");
        near.embedding = Some(axis(0));
        let near = store.store_feedback(near).unwrap();
        let mut far = feedback("ops", "rotating credentials too", "use the vault again");
        far.embedding = Some(axis(5));
        store.store_feedback(far).unwrap();

        let hits = store
            .search_feedback("secrets", Some(&axis(0)), None, 5)
            .unwrap();
        assert_eq!(hits.len(), 1, "orthogonal vector scores 0 and is dropped");
        assert_eq!(hits[0].id, near);

        assert!(
            store
                .search_feedback("secrets", Some(&[0.5; DIMS / 2]), None, 5)
                .unwrap()
                .is_empty()
        );
    }
}
