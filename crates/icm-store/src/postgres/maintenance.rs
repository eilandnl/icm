//! PostgreSQL backend -- split out of the former monolithic postgres.rs.
//!
//! Decay/auto-consolidate maintenance methods.

use super::*;

impl PostgresStore {
    /// Read a raw string value from `icm_metadata` by key.
    /// Returns `Ok(None)` when the key is absent.
    pub fn get_metadata_str(&self, key: &str) -> IcmResult<Option<String>> {
        let mut c = self.conn()?;
        let row = c
            .query_opt("SELECT value FROM icm_metadata WHERE key = $1", &[&key])
            .map_err(pg_err)?;
        Ok(row.and_then(|row| row.get(0)))
    }

    /// Upsert a raw string value into `icm_metadata`.
    pub fn set_metadata_str(&self, key: &str, value: &str) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("set_metadata_str".into()));
        }
        let mut c = self.conn()?;
        c.execute(
            "INSERT INTO icm_metadata (key, value) VALUES ($1, $2) \
             ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            &[&key, &value],
        )
        .map_err(pg_err)?;
        Ok(())
    }

    /// Drop every stored vector and resize the vector columns and their
    /// index to `new_dims` (the explicit `icm embed --migrate` path).
    /// Returns the number of memories whose embedding was cleared.
    pub fn reset_vector_index(&self, new_dims: usize) -> IcmResult<usize> {
        self.reset_vectors(new_dims, None)
    }

    /// [`Self::reset_vector_index`] that also records `model` as the new
    /// owner of the index, in the same transaction.
    pub fn reset_vector_index_for_model(&self, new_dims: usize, model: &str) -> IcmResult<usize> {
        self.reset_vectors(new_dims, Some(model))
    }

    fn reset_vectors(&self, new_dims: usize, model: Option<&str>) -> IcmResult<usize> {
        if self.readonly {
            return Err(IcmError::ReadOnly("reset_vector_index".into()));
        }
        if !(64..=4096).contains(&new_dims) {
            return Err(IcmError::InvalidInput(format!(
                "embedding_dims must be between 64 and 4096, got {new_dims}"
            )));
        }
        let mut c = self.conn()?;
        let mut tx = c.transaction().map_err(pg_err)?;
        // Two migrations at once would each resize the columns under the other.
        tx.query(
            "SELECT pg_advisory_xact_lock(hashtext('icm-vector-reset'))",
            &[],
        )
        .map_err(pg_err)?;
        let discarded: i64 = tx
            .query_one("SELECT COUNT(embedding)::bigint FROM memories", &[])
            .map_err(pg_err)?
            .get(0);
        // The vectors are derived data: re-embedding rebuilds them. A column
        // cannot keep values of another size, so they are cleared.
        tx.batch_execute(&format!(
            "DROP INDEX IF EXISTS idx_memories_embedding; \
             ALTER TABLE memories ALTER COLUMN embedding TYPE vector({new_dims}) USING NULL; \
             ALTER TABLE feedback ALTER COLUMN embedding TYPE vector({new_dims}) USING NULL; \
             CREATE INDEX idx_memories_embedding \
                 ON memories USING hnsw (embedding vector_cosine_ops)"
        ))
        .map_err(pg_err)?;
        let mut record = vec![("embedding_dims", new_dims.to_string())];
        if let Some(model) = model {
            record.push((icm_core::META_EMBEDDING_MODEL, model.to_string()));
        }
        for (key, value) in &record {
            tx.execute(
                "INSERT INTO icm_metadata (key, value) VALUES ($1, $2) \
                 ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                &[key, value],
            )
            .map_err(pg_err)?;
        }
        tx.commit().map_err(pg_err)?;
        self.embedding_dims.store(new_dims, Ordering::Relaxed);
        Ok(discarded.max(0) as usize)
    }

    /// Apply decay if more than 24 hours since the last run. Mirrors the
    /// SQLite backend's atomic check-and-claim via `icm_metadata`.
    pub fn maybe_auto_decay(&self) -> IcmResult<()> {
        if self.readonly {
            return Ok(());
        }
        let now = Utc::now();
        let claimed = {
            let mut c = self.conn()?;
            c.execute(
                "INSERT INTO icm_metadata (key, value) VALUES ('last_decay_at', $1)
                 ON CONFLICT (key) DO UPDATE SET value = $1
                 WHERE icm_metadata.value IS NULL
                    OR ($1::timestamptz - icm_metadata.value::timestamptz) >= interval '1 day'",
                &[&now.to_rfc3339()],
            )
            .map_err(pg_err)?
        };
        if claimed > 0 {
            self.apply_decay(0.95)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::tests::{live_database, live_store};

    const DIMS: usize = 64;

    #[test]
    fn metadata_values_round_trip_and_upsert() {
        let Some(store) = live_store("meta_roundtrip", DIMS) else {
            return;
        };
        assert_eq!(store.get_metadata_str("conversation:s1").unwrap(), None);
        store.set_metadata_str("conversation:s1", "120").unwrap();
        assert_eq!(
            store
                .get_metadata_str("conversation:s1")
                .unwrap()
                .as_deref(),
            Some("120")
        );
        store.set_metadata_str("conversation:s1", "480").unwrap();
        assert_eq!(
            store
                .get_metadata_str("conversation:s1")
                .unwrap()
                .as_deref(),
            Some("480")
        );
        // The schema's own keys are readable too.
        assert_eq!(
            store.get_metadata_str("embedding_dims").unwrap().as_deref(),
            Some("64")
        );
    }

    fn embedded(topic: &str, summary: &str, dims: usize) -> Memory {
        let mut m = Memory::new(topic.into(), summary.into(), Importance::Medium);
        let mut v = vec![0.0_f32; dims];
        v[0] = 1.0;
        m.embedding = Some(v);
        m
    }

    #[test]
    fn resetting_the_vector_index_resizes_everything_and_records_the_model() {
        let Some(url) = live_database("reset_vectors") else {
            return;
        };
        let store = PostgresStore::connect_url(&url, DIMS, false).unwrap();
        for i in 0..3 {
            store
                .store(embedded("t", &format!("with vector {i}"), DIMS))
                .unwrap();
        }
        store
            .store(Memory::new(
                "t".into(),
                "without".into(),
                Importance::Medium,
            ))
            .unwrap();
        let mut fb = Feedback::new(
            "t".into(),
            "c".into(),
            "p".into(),
            "x".into(),
            None,
            "s".into(),
        );
        fb.embedding = Some(vec![0.5; DIMS]);
        store.store_feedback(fb).unwrap();

        assert_eq!(
            store
                .reset_vector_index_for_model(128, "new/model")
                .unwrap(),
            3
        );

        // The same store now works at the new size and refuses the old one.
        assert_eq!(store.dims(), 128);
        assert!(
            store
                .list_all()
                .unwrap()
                .iter()
                .all(|m| m.embedding.is_none())
        );
        assert!(matches!(
            store.store(embedded("t", "old size", DIMS)),
            Err(IcmError::InvalidInput(_))
        ));
        store.store(embedded("t", "new size", 128)).unwrap();
        let mut fb = Feedback::new(
            "t".into(),
            "c2".into(),
            "p".into(),
            "x".into(),
            None,
            "s".into(),
        );
        fb.embedding = Some(vec![0.5; 128]);
        store.store_feedback(fb).unwrap();
        let mut query = vec![0.0_f32; 128];
        query[0] = 1.0;
        let hits = store.search_by_embedding(&query, 5).unwrap();
        assert_eq!(hits.len(), 1);

        // What is recorded is what a later open uses, whatever it asks for.
        assert_eq!(
            store.get_metadata_str("embedding_dims").unwrap().as_deref(),
            Some("128")
        );
        assert_eq!(
            store
                .get_metadata_str(icm_core::META_EMBEDDING_MODEL)
                .unwrap()
                .as_deref(),
            Some("new/model")
        );
        let reopened = PostgresStore::connect_url(&url, DIMS, false).unwrap();
        assert_eq!(reopened.dims(), 128);

        // The HNSW index was recreated.
        let mut c = store.conn().unwrap();
        let indexes = c
            .query(
                "SELECT indexname FROM pg_indexes WHERE indexname = 'idx_memories_embedding'",
                &[],
            )
            .unwrap();
        assert_eq!(indexes.len(), 1);
    }

    #[test]
    fn a_reset_rejects_a_bad_size_and_a_read_only_store() {
        let Some(url) = live_database("reset_guard") else {
            return;
        };
        let store = PostgresStore::connect_url(&url, DIMS, false).unwrap();
        assert!(matches!(
            store.reset_vector_index(8),
            Err(IcmError::InvalidInput(_))
        ));
        assert!(matches!(
            store.reset_vector_index(5000),
            Err(IcmError::InvalidInput(_))
        ));
        assert_eq!(store.dims(), DIMS, "a rejected reset changes nothing");

        let reader = PostgresStore::connect_url(&url, DIMS, true).unwrap();
        assert!(matches!(
            reader.reset_vector_index(128),
            Err(IcmError::ReadOnly(_))
        ));
    }

    #[test]
    fn a_read_only_store_reads_but_does_not_write_metadata() {
        let Some(url) = live_database("meta_readonly") else {
            return;
        };
        let writer = PostgresStore::connect_url(&url, DIMS, false).unwrap();
        writer.set_metadata_str("k", "v").unwrap();

        let reader = PostgresStore::connect_url(&url, DIMS, true).unwrap();
        assert_eq!(reader.get_metadata_str("k").unwrap().as_deref(), Some("v"));
        assert!(matches!(
            reader.set_metadata_str("k", "other"),
            Err(IcmError::ReadOnly(_))
        ));
        assert_eq!(reader.get_metadata_str("k").unwrap().as_deref(), Some("v"));
    }
}
