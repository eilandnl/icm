//! PostgreSQL backend: integrity check and index rebuild (`icm repair`).
//!
//! The SQLite backend checks its file with `PRAGMA integrity_check`.
//! PostgreSQL protects its own pages with the write-ahead log, so what is
//! checked here is what ICM depends on: that its tables and indexes exist
//! and are valid, and that the stored embedding size is the one the vector
//! columns have. The repair recreates what is missing and rebuilds the
//! indexes.

use super::*;

/// Tables the schema creates.
const TABLES: &[&str] = &[
    "icm_metadata",
    "memories",
    "pending_extractions",
    "pending_consolidations",
    "code_areas",
    "hook_events",
    "facts",
    "feedback",
    "sessions",
    "messages",
    "memoirs",
    "concepts",
    "concept_links",
];

/// Indexes the search and uniqueness rules rely on.
const INDEXES: &[&str] = &[
    "idx_memories_fts",
    "idx_memories_embedding",
    "idx_memories_topic_hash",
    "idx_feedback_fts",
    "idx_messages_fts",
    "idx_concepts_fts",
    "idx_facts_active_unique",
];

/// Tables with a search index, rebuilt by [`PostgresStore::rebuild_search_indexes`].
const SEARCHED_TABLES: &[&str] = &["memories", "feedback", "messages", "concepts"];

fn names(of: &[&str]) -> Vec<String> {
    of.iter().map(|n| (*n).to_string()).collect()
}

impl PostgresStore {
    /// Open the database for inspection and repair. Unlike a normal open
    /// this does not run the schema migration, which would quietly recreate
    /// the very tables and indexes the check is meant to find missing.
    pub fn open_maintenance(_path: &Path) -> IcmResult<Self> {
        Self::connect_maintenance(&Self::conn_string()?)
    }

    pub(crate) fn connect_maintenance(url: &str) -> IcmResult<Self> {
        let mut client = Client::connect(url, tls::connector(url)?).map_err(|e| {
            IcmError::Database(format!("cannot connect to PostgreSQL: {}", describe(&e)))
        })?;
        let has_metadata: bool = client
            .query_one("SELECT to_regclass('icm_metadata') IS NOT NULL", &[])
            .map_err(pg_err)?
            .get(0);
        let stored: Option<i64> = if has_metadata {
            client
                .query_opt(
                    "SELECT value::bigint FROM icm_metadata WHERE key = 'embedding_dims'",
                    &[],
                )
                .map_err(pg_err)?
                .map(|row| row.get(0))
        } else {
            None
        };
        let dims = stored
            .map(|d| d as usize)
            .unwrap_or(icm_core::DEFAULT_EMBEDDING_DIMS);
        Ok(Self {
            client: Mutex::new(client),
            embedding_dims: AtomicUsize::new(dims),
            readonly: false,
        })
    }

    /// Problems found with the schema, or `["ok"]` when there are none.
    pub fn integrity_check(&self) -> IcmResult<Vec<String>> {
        let mut c = self.conn()?;
        let mut problems = Vec::new();

        let missing = |c: &mut Client, wanted: &[&str]| -> IcmResult<Vec<String>> {
            let rows = c
                .query(
                    "SELECT n FROM unnest($1::text[]) AS n WHERE to_regclass(n) IS NULL",
                    &[&names(wanted)],
                )
                .map_err(pg_err)?;
            Ok(rows.iter().map(|row| row.get(0)).collect())
        };
        let missing_tables = missing(&mut c, TABLES)?;
        problems.extend(missing_tables.iter().map(|t| format!("missing table {t}")));
        problems.extend(
            missing(&mut c, INDEXES)?
                .iter()
                .map(|i| format!("missing index {i}")),
        );

        let invalid = c
            .query(
                "SELECT cls.relname FROM pg_index i \
                 JOIN pg_class cls ON cls.oid = i.indexrelid \
                 WHERE NOT i.indisvalid \
                   AND i.indrelid = ANY(ARRAY( \
                       SELECT to_regclass(t)::oid FROM unnest($1::text[]) AS t \
                        WHERE to_regclass(t) IS NOT NULL)) \
                 ORDER BY 1",
                &[&names(TABLES)],
            )
            .map_err(pg_err)?;
        problems.extend(
            invalid
                .iter()
                .map(|row| format!("invalid index {}", row.get::<_, String>(0))),
        );

        // The recorded size must be the size of the vector columns, or every
        // write of a vector fails.
        if !missing_tables.iter().any(|t| t == "icm_metadata") {
            let stored: Option<String> = c
                .query_opt(
                    "SELECT value FROM icm_metadata WHERE key = 'embedding_dims'",
                    &[],
                )
                .map_err(pg_err)?
                .and_then(|row| row.get(0));
            for table in ["memories", "feedback"] {
                if missing_tables.iter().any(|t| t == table) {
                    continue;
                }
                let typmod: Option<i32> = c
                    .query_opt(
                        "SELECT atttypmod FROM pg_attribute \
                         WHERE attrelid = to_regclass($1) AND attname = 'embedding'",
                        &[&table],
                    )
                    .map_err(pg_err)?
                    .map(|row| row.get(0));
                if let (Some(stored), Some(typmod)) = (&stored, typmod)
                    && stored.parse::<i64>().ok() != Some(i64::from(typmod))
                {
                    problems.push(format!(
                        "embedding_dims is recorded as {stored} but {table}.embedding is \
                         vector({typmod})"
                    ));
                }
            }
        }

        if problems.is_empty() {
            problems.push("ok".to_string());
        }
        Ok(problems)
    }

    /// Same as [`Self::integrity_check`]: every query is read-only.
    pub fn integrity_check_structural(&self) -> IcmResult<Vec<String>> {
        self.integrity_check()
    }

    /// Recreate any missing table or index, then rebuild the indexes of the
    /// tables that have a search index. Returns those tables.
    pub fn rebuild_search_indexes(&self) -> IcmResult<Vec<String>> {
        if self.readonly {
            return Err(IcmError::ReadOnly("rebuild_search_indexes".into()));
        }
        let mut c = self.conn()?;
        // The schema is written to be run again: it only adds what is absent.
        connection::init_schema_serialized(&mut c, self.dims())?;
        for table in SEARCHED_TABLES {
            c.batch_execute(&format!("REINDEX TABLE {table}"))
                .map_err(pg_err)?;
        }
        Ok(names(SEARCHED_TABLES))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::tests::{live_database, live_store};

    const DIMS: usize = 64;

    fn exec(store: &PostgresStore, sql: &str) {
        store.conn().unwrap().batch_execute(sql).unwrap();
    }

    fn ok() -> Vec<String> {
        vec!["ok".to_string()]
    }

    #[test]
    fn a_healthy_database_checks_ok() {
        let Some(store) = live_store("ic_ok", DIMS) else {
            return;
        };
        assert_eq!(store.integrity_check().unwrap(), ok());
        assert_eq!(store.integrity_check_structural().unwrap(), ok());
    }

    #[test]
    fn missing_tables_and_indexes_are_reported_and_recreated() {
        let Some(store) = live_store("ic_missing", DIMS) else {
            return;
        };
        exec(&store, "DROP TABLE concept_links");
        exec(&store, "DROP INDEX idx_feedback_fts");

        let problems = store.integrity_check().unwrap();
        assert_eq!(
            problems,
            vec![
                "missing table concept_links".to_string(),
                "missing index idx_feedback_fts".to_string()
            ]
        );

        let rebuilt = store.rebuild_search_indexes().unwrap();
        assert_eq!(
            rebuilt,
            vec!["memories", "feedback", "messages", "concepts"]
        );
        assert_eq!(store.integrity_check().unwrap(), ok());
    }

    #[test]
    fn an_invalid_index_is_reported_and_rebuilt() {
        let Some(store) = live_store("ic_invalid", DIMS) else {
            return;
        };
        // What a failed `CREATE INDEX CONCURRENTLY` leaves behind.
        exec(
            &store,
            "UPDATE pg_index SET indisvalid = false WHERE indexrelid = 'idx_messages_fts'::regclass",
        );
        assert_eq!(
            store.integrity_check().unwrap(),
            vec!["invalid index idx_messages_fts".to_string()]
        );

        store.rebuild_search_indexes().unwrap();
        assert_eq!(store.integrity_check().unwrap(), ok());
    }

    #[test]
    fn a_recorded_size_that_differs_from_the_columns_is_reported() {
        let Some(store) = live_store("ic_dims", DIMS) else {
            return;
        };
        exec(
            &store,
            "UPDATE icm_metadata SET value = '999' WHERE key = 'embedding_dims'",
        );
        let problems = store.integrity_check().unwrap();
        assert_eq!(problems.len(), 2, "{problems:?}");
        assert!(problems[0].contains("recorded as 999 but memories.embedding is vector(64)"));
        assert!(problems[1].contains("feedback.embedding"));
    }

    #[test]
    fn a_read_only_store_does_not_rebuild() {
        let Some(url) = live_database("ic_readonly") else {
            return;
        };
        PostgresStore::connect_url(&url, DIMS, false).unwrap();
        let reader = PostgresStore::connect_url(&url, DIMS, true).unwrap();
        assert!(matches!(
            reader.rebuild_search_indexes(),
            Err(IcmError::ReadOnly(_))
        ));
        assert_eq!(reader.integrity_check().unwrap(), ok());
    }

    /// The maintenance open must not migrate, or it would repair what it is
    /// asked to inspect.
    #[test]
    fn opening_for_maintenance_does_not_create_the_schema() {
        let Some(url) = live_database("ic_maintenance") else {
            return;
        };
        let store = PostgresStore::connect_maintenance(&url).unwrap();
        let problems = store.integrity_check().unwrap();
        assert!(problems.contains(&"missing table memories".to_string()));
        assert!(problems.contains(&"missing index idx_memories_embedding".to_string()));
        let created: i64 = store
            .conn()
            .unwrap()
            .query_one(
                "SELECT COUNT(*) FROM pg_tables WHERE tablename = 'memories'",
                &[],
            )
            .unwrap()
            .get(0);
        assert_eq!(created, 0);

        // The repair then builds it.
        store.rebuild_search_indexes().unwrap();
        assert_eq!(store.integrity_check().unwrap(), ok());
    }
}
