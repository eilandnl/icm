//! Test suite for the PostgreSQL backend (`postgres::tests`).

use super::*;

/// Audit regression: a keyword containing `%`/`_` was interpolated
/// straight into an ILIKE pattern unescaped, turning it into an
/// unintended wildcard.
#[test]
fn test_escape_like_wildcards() {
    assert_eq!(escape_like_wildcards("100%"), "100\\%");
    assert_eq!(escape_like_wildcards("snake_case"), "snake\\_case");
    assert_eq!(escape_like_wildcards("back\\slash"), "back\\\\slash");
    assert_eq!(escape_like_wildcards("plain"), "plain");
}

/// Audit regression: `apply_decay`'s raw multiplier goes negative for
/// `low` importance + low access count at factor < 0.5 (still inside
/// the CLI's own validated [0.0, 1.0) range). This reproduces the exact
/// arithmetic the SQL `GREATEST(0.0, ...)` clamp now guards, as a plain
/// Rust assertion (no live Postgres needed to prove the formula itself
/// would go negative without the clamp).
#[test]
fn test_apply_decay_formula_would_go_negative_without_clamp() {
    let factor: f64 = 0.4;
    let mult: f64 = 2.0; // low importance
    let access: f64 = 0.0;
    let raw = 1.0 - (1.0 - factor) * mult / (1.0 + access * 0.1);
    assert!(
        raw < 0.0,
        "expected the pre-clamp formula to go negative, got {raw}"
    );
    assert_eq!(
        raw.max(0.0),
        0.0,
        "GREATEST(0.0, ...) must clamp this to 0.0"
    );
}

/// A store in a fresh database of the server named by
/// `ICM_POSTGRES_TEST_URL` (`postgres://user@host:port/db`, a role that may
/// create databases and the pgvector extension), or `None` to skip when it
/// is not set. One database per test: the schema setup looks indexes up by
/// name, so tests cannot share a database.
pub(crate) fn live_store(name: &str, dims: usize) -> Option<PostgresStore> {
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
    Some(PostgresStore::connect_url(&url, dims, false).expect("open test store"))
}

fn memory(topic: &str, summary: &str, importance: Importance) -> Memory {
    Memory::new(topic.into(), summary.into(), importance)
}

#[test]
fn auto_consolidate_below_threshold_does_nothing() {
    let Some(store) = live_store("ac_below", icm_core::DEFAULT_EMBEDDING_DIMS) else {
        return;
    };
    store.store(memory("t", "one", Importance::Medium)).unwrap();
    store.store(memory("t", "two", Importance::Medium)).unwrap();

    assert!(!store.auto_consolidate("t", 10).unwrap());
    assert_eq!(store.count_by_topic("t").unwrap(), 2);
}

#[test]
fn auto_consolidate_rolls_a_topic_up_into_one_memory() {
    let Some(store) = live_store("ac_above", icm_core::DEFAULT_EMBEDDING_DIMS) else {
        return;
    };
    for i in 0..12 {
        let mut m = memory("bulk", &format!("entry {i}"), Importance::Medium);
        m.keywords = vec![format!("kw{}", i % 3), "Shared".into(), "shared".into()];
        store.store(m).unwrap();
    }

    assert!(store.auto_consolidate("bulk", 10).unwrap());

    let rolled = store.get_by_topic("bulk").unwrap();
    assert_eq!(rolled.len(), 1);
    assert_eq!(rolled[0].importance, Importance::High);
    assert_eq!(
        rolled[0].raw_excerpt.as_deref(),
        Some("auto-consolidated from 12 memories")
    );
    // Keywords are the union, case-insensitively de-duplicated.
    let mut keywords: Vec<String> = rolled[0]
        .keywords
        .iter()
        .map(|k| k.to_lowercase())
        .collect();
    keywords.sort();
    assert_eq!(keywords, vec!["kw0", "kw1", "kw2", "shared"]);
}

/// The store reads a topic 500 memories at a time. The rollup replaces the
/// memories that were read; the ones past the read must survive.
#[test]
fn auto_consolidate_only_removes_the_memories_it_read() {
    let Some(store) = live_store("ac_read", icm_core::DEFAULT_EMBEDDING_DIMS) else {
        return;
    };
    for i in 0..520 {
        store
            .store(memory("many", &format!("fact {i:03}"), Importance::Medium))
            .unwrap();
    }

    assert!(store.auto_consolidate("many", 10).unwrap());
    assert_eq!(store.count_by_topic("many").unwrap(), 1 + 20);
}

/// Critical memories are exempt from consolidation, so they must not count
/// toward the threshold, or a topic of criticals would churn a rollup on
/// every store.
#[test]
fn auto_consolidate_ignores_critical_memories() {
    let Some(store) = live_store("ac_critical", icm_core::DEFAULT_EMBEDDING_DIMS) else {
        return;
    };
    for i in 0..3 {
        store
            .store(memory("t", &format!("critical {i}"), Importance::Critical))
            .unwrap();
    }
    store
        .store(memory("t", "one expendable", Importance::Medium))
        .unwrap();

    assert!(!store.auto_consolidate("t", 3).unwrap());
    assert_eq!(store.get_by_topic("t").unwrap().len(), 4);

    store
        .store(memory("t", "expendable 2", Importance::Medium))
        .unwrap();
    store
        .store(memory("t", "expendable 3", Importance::Medium))
        .unwrap();

    assert!(store.auto_consolidate("t", 3).unwrap());
    let after = store.get_by_topic("t").unwrap();
    let criticals = after
        .iter()
        .filter(|m| matches!(m.importance, Importance::Critical))
        .count();
    assert_eq!(criticals, 3, "all criticals must survive the rollup");
    assert_eq!(after.len(), 4, "3 criticals + 1 consolidated");
}

#[test]
fn auto_consolidate_with_embedder_attaches_an_embedding() {
    struct StubEmbedder;
    impl icm_core::Embedder for StubEmbedder {
        fn embed(&self, _text: &str) -> IcmResult<Vec<f32>> {
            Ok(vec![0.42; icm_core::DEFAULT_EMBEDDING_DIMS])
        }
        fn embed_batch(&self, texts: &[&str]) -> IcmResult<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|_| vec![0.42; icm_core::DEFAULT_EMBEDDING_DIMS])
                .collect())
        }
        fn dimensions(&self) -> usize {
            icm_core::DEFAULT_EMBEDDING_DIMS
        }
    }
    let Some(store) = live_store("ac_embed", icm_core::DEFAULT_EMBEDDING_DIMS) else {
        return;
    };
    for i in 0..11 {
        store
            .store(memory("rolled", &format!("fact {i}"), Importance::Medium))
            .unwrap();
    }

    assert!(
        store
            .auto_consolidate_with_embedder("rolled", 10, Some(&StubEmbedder))
            .unwrap()
    );
    let rolled = store.get_by_topic("rolled").unwrap();
    assert_eq!(rolled.len(), 1);
    let embedding = rolled[0].embedding.as_ref().expect("rollup is embedded");
    assert_eq!(embedding.len(), icm_core::DEFAULT_EMBEDDING_DIMS);
    assert!((embedding[0] - 0.42).abs() < 1e-6);
}
