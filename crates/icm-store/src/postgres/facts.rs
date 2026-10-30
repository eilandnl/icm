//! PostgreSQL backend: structured facts.
//!
//! Same model as the SQLite backend: one active row per `(entity, key)`,
//! older values kept as superseded history.

use super::*;

pub(super) const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS facts (
        id TEXT PRIMARY KEY,
        entity TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL,
        superseded_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_facts_entity_key ON facts(entity, key);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_active_unique
        ON facts(entity, key) WHERE superseded_at IS NULL;";

const FACT_COLS: &str = "id, entity, key, value, source, created_at, superseded_at";

fn row_to_fact(row: &postgres::Row) -> Fact {
    Fact {
        id: row.get(0),
        entity: row.get(1),
        key: row.get(2),
        value: row.get(3),
        source: row.get(4),
        created_at: row.get(5),
        superseded_at: row.get(6),
    }
}

impl PostgresStore {
    /// List every **active** fact across all entities (for `icm export`).
    pub fn list_all_facts(&self) -> IcmResult<Vec<Fact>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {FACT_COLS} FROM facts WHERE superseded_at IS NULL \
                     ORDER BY entity COLLATE \"C\" ASC, key COLLATE \"C\" ASC"
                ),
                &[],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_fact).collect())
    }
}

impl FactsStore for PostgresStore {
    fn set_fact(&self, entity: &str, key: &str, value: &str, source: &str) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("set_fact".into()));
        }
        if entity.is_empty() || key.is_empty() {
            return Err(IcmError::InvalidInput(
                "entity and key must be non-empty".into(),
            ));
        }
        let mut c = self.conn()?;
        let mut tx = c.transaction().map_err(pg_err)?;
        // Two writers on a slot with no active row would both insert one and
        // the unique index would fail the second. The lock makes them take
        // turns, so the second sees the first's row and supersedes it.
        tx.query(
            "SELECT pg_advisory_xact_lock(hashtext($1))",
            &[&format!("icm-fact\u{1f}{entity}\u{1f}{key}")],
        )
        .map_err(pg_err)?;

        let existing = tx
            .query_opt(
                "SELECT id, value FROM facts \
                 WHERE entity = $1 AND key = $2 AND superseded_at IS NULL",
                &[&entity, &key],
            )
            .map_err(pg_err)?;
        if let Some(row) = existing {
            let id: String = row.get(0);
            let current: String = row.get(1);
            if current == value {
                return Ok(id);
            }
            tx.execute(
                "UPDATE facts SET superseded_at = $1 WHERE id = $2",
                &[&Utc::now(), &id],
            )
            .map_err(pg_err)?;
        }

        let new = Fact::new(
            entity.to_string(),
            key.to_string(),
            value.to_string(),
            source.to_string(),
        );
        tx.execute(
            "INSERT INTO facts (id, entity, key, value, source, created_at, superseded_at) \
             VALUES ($1, $2, $3, $4, $5, $6, NULL)",
            &[
                &new.id,
                &new.entity,
                &new.key,
                &new.value,
                &new.source,
                &new.created_at,
            ],
        )
        .map_err(pg_err)?;
        tx.commit().map_err(pg_err)?;
        Ok(new.id)
    }

    fn get_fact(&self, entity: &str, key: &str) -> IcmResult<Option<Fact>> {
        let mut c = self.conn()?;
        let row = c
            .query_opt(
                &format!(
                    "SELECT {FACT_COLS} FROM facts \
                     WHERE entity = $1 AND key = $2 AND superseded_at IS NULL"
                ),
                &[&entity, &key],
            )
            .map_err(pg_err)?;
        Ok(row.as_ref().map(row_to_fact))
    }

    fn list_facts(&self, entity: &str, key_prefix: Option<&str>) -> IcmResult<Vec<Fact>> {
        let mut c = self.conn()?;
        let rows = match key_prefix {
            Some(prefix) if !prefix.is_empty() => {
                let pattern = format!("{}%", escape_like_wildcards(prefix));
                c.query(
                    &format!(
                        "SELECT {FACT_COLS} FROM facts \
                         WHERE entity = $1 AND key LIKE $2 ESCAPE '\\' \
                           AND superseded_at IS NULL \
                         ORDER BY key COLLATE \"C\" ASC"
                    ),
                    &[&entity, &pattern],
                )
            }
            _ => c.query(
                &format!(
                    "SELECT {FACT_COLS} FROM facts \
                     WHERE entity = $1 AND superseded_at IS NULL \
                     ORDER BY key COLLATE \"C\" ASC"
                ),
                &[&entity],
            ),
        }
        .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_fact).collect())
    }

    fn history(&self, entity: &str, key: &str) -> IcmResult<Vec<Fact>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {FACT_COLS} FROM facts WHERE entity = $1 AND key = $2 \
                     ORDER BY created_at DESC, id DESC"
                ),
                &[&entity, &key],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_fact).collect())
    }

    fn forget_fact(&self, entity: &str, key: &str) -> IcmResult<usize> {
        if self.readonly {
            return Err(IcmError::ReadOnly("forget_fact".into()));
        }
        let mut c = self.conn()?;
        let removed = c
            .execute(
                "DELETE FROM facts WHERE entity = $1 AND key = $2",
                &[&entity, &key],
            )
            .map_err(pg_err)?;
        Ok(removed as usize)
    }

    fn facts_stats(&self) -> IcmResult<FactsStats> {
        let mut c = self.conn()?;
        let counts = c
            .query_one(
                "SELECT COUNT(*) FILTER (WHERE superseded_at IS NULL)::bigint, \
                        COUNT(*)::bigint, \
                        COUNT(DISTINCT entity) FILTER (WHERE superseded_at IS NULL)::bigint \
                 FROM facts",
                &[],
            )
            .map_err(pg_err)?;
        let top = c
            .query(
                "SELECT entity, COUNT(*)::bigint AS n FROM facts \
                 WHERE superseded_at IS NULL \
                 GROUP BY entity ORDER BY n DESC, entity COLLATE \"C\" ASC LIMIT 10",
                &[],
            )
            .map_err(pg_err)?;
        let count = |idx: usize| counts.get::<_, i64>(idx).max(0) as usize;
        Ok(FactsStats {
            active_count: count(0),
            total_count: count(1),
            distinct_entities: count(2),
            top_entities: top
                .iter()
                .map(|row| (row.get(0), row.get::<_, i64>(1).max(0) as usize))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::tests::{live_database, live_store};

    const DIMS: usize = 64;

    #[test]
    fn a_new_value_supersedes_the_old_one_and_keeps_it_as_history() {
        let Some(store) = live_store("facts_supersede", DIMS) else {
            return;
        };
        let first = store.set_fact("svc", "host", "a.example", "user").unwrap();
        // The same value again is a no-op on the same row.
        assert_eq!(
            store.set_fact("svc", "host", "a.example", "other").unwrap(),
            first
        );
        let second = store.set_fact("svc", "host", "b.example", "user").unwrap();
        assert_ne!(first, second);

        let active = store.get_fact("svc", "host").unwrap().unwrap();
        assert_eq!(
            (active.id.as_str(), active.value.as_str()),
            (second.as_str(), "b.example")
        );
        assert!(active.is_active());

        let history = store.history("svc", "host").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].id, second);
        assert!(!history[1].is_active());
        assert_eq!(history[1].value, "a.example");
    }

    #[test]
    fn list_returns_active_facts_in_key_order_and_prefix_is_literal() {
        let Some(store) = live_store("facts_list", DIMS) else {
            return;
        };
        for (key, value) in [("b", "2"), ("a_1", "1"), ("a%2", "x"), ("ab", "y")] {
            store.set_fact("e", key, value, "t").unwrap();
        }
        store.set_fact("e", "b", "3", "t").unwrap();
        store.set_fact("other", "b", "z", "t").unwrap();

        let all: Vec<(String, String)> = store
            .list_facts("e", None)
            .unwrap()
            .into_iter()
            .map(|f| (f.key, f.value))
            .collect();
        assert_eq!(
            all,
            vec![
                ("a%2".to_string(), "x".to_string()),
                ("a_1".to_string(), "1".to_string()),
                ("ab".to_string(), "y".to_string()),
                ("b".to_string(), "3".to_string()),
            ]
        );

        // `_` and `%` in a prefix match themselves, not "any character".
        let keys = |prefix: &str| -> Vec<String> {
            store
                .list_facts("e", Some(prefix))
                .unwrap()
                .into_iter()
                .map(|f| f.key)
                .collect()
        };
        assert_eq!(keys("a_"), vec!["a_1"]);
        assert_eq!(keys("a%"), vec!["a%2"]);
        assert_eq!(
            keys(""),
            keys("a")
                .into_iter()
                .chain(["b".to_string()])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn forget_removes_history_too_and_stats_count_active_rows() {
        let Some(store) = live_store("facts_stats", DIMS) else {
            return;
        };
        store.set_fact("a", "k1", "1", "t").unwrap();
        store.set_fact("a", "k1", "2", "t").unwrap();
        store.set_fact("a", "k2", "1", "t").unwrap();
        store.set_fact("b", "k1", "1", "t").unwrap();

        let stats = store.facts_stats().unwrap();
        assert_eq!(stats.active_count, 3);
        assert_eq!(stats.total_count, 4);
        assert_eq!(stats.distinct_entities, 2);
        assert_eq!(
            stats.top_entities,
            vec![("a".to_string(), 2), ("b".to_string(), 1)]
        );
        assert_eq!(store.list_all_facts().unwrap().len(), 3);

        assert_eq!(store.forget_fact("a", "k1").unwrap(), 2);
        assert!(store.get_fact("a", "k1").unwrap().is_none());
        assert!(store.history("a", "k1").unwrap().is_empty());
        assert_eq!(store.forget_fact("a", "k1").unwrap(), 0);
    }

    #[test]
    fn empty_entity_or_key_is_rejected() {
        let Some(store) = live_store("facts_empty", DIMS) else {
            return;
        };
        assert!(store.set_fact("", "k", "v", "t").is_err());
        assert!(store.set_fact("e", "", "v", "t").is_err());
    }

    /// Two writers racing on a slot with no active row must both succeed and
    /// leave exactly one active value, not fail the unique index.
    #[test]
    fn concurrent_writers_leave_one_active_row() {
        let Some(url) = live_database("facts_race") else {
            return;
        };
        // Create the schema once: concurrent first-time migrations race on
        // `CREATE EXTENSION`, which is not what this test is about.
        let store = PostgresStore::connect_url(&url, DIMS, false).unwrap();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let url = url.clone();
                std::thread::spawn(move || {
                    let store = PostgresStore::connect_url(&url, DIMS, false).unwrap();
                    store.set_fact("e", "k", &format!("v{i}"), "t").unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }

        let history = store.history("e", "k").unwrap();
        assert_eq!(history.len(), 8);
        assert_eq!(history.iter().filter(|f| f.is_active()).count(), 1);
    }
}
