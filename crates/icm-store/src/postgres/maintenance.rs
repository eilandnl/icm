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
