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

/// The URL of a fresh database on the server named by
/// `ICM_POSTGRES_TEST_URL` (`postgres://user@host:port/db`, a role that may
/// create databases and the pgvector extension), or `None` to skip when it
/// is not set. One database per test: the schema setup looks indexes up by
/// name, so tests cannot share a database.
pub(crate) fn live_database(name: &str) -> Option<String> {
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
    Some(format!("{server}/{database}"))
}

/// A store in a fresh database, see [`live_database`].
pub(crate) fn live_store(name: &str, dims: usize) -> Option<PostgresStore> {
    let url = live_database(name)?;
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

#[test]
fn sql_bound_never_goes_negative() {
    assert_eq!(sql_bound(0), 0);
    assert_eq!(sql_bound(42), 42);
    assert_eq!(sql_bound(usize::MAX), i64::MAX);
}

/// Callers pass `usize::MAX` for "no limit" (`icm export` does). Cast with
/// `as i64` that is -1, which SQLite accepts and PostgreSQL rejects.
#[test]
fn a_usize_max_limit_means_no_limit() {
    let Some(store) = live_store("limits_max", 64) else {
        return;
    };
    let sid = store.create_session("cli", Some("p"), None).unwrap();
    store
        .record_message(&sid, Role::User, "hello world", None, None, None)
        .unwrap();
    store
        .store_feedback(Feedback::new(
            "t".into(),
            "c".into(),
            "p".into(),
            "x".into(),
            None,
            "s".into(),
        ))
        .unwrap();
    let memoir = store
        .create_memoir(Memoir::new("m".into(), String::new()))
        .unwrap();
    let concept = store
        .add_concept(Concept::new(
            memoir.clone(),
            "alpha".into(),
            "alpha concept".into(),
        ))
        .unwrap();

    assert_eq!(store.list_feedback(None, usize::MAX).unwrap().len(), 1);
    assert_eq!(store.list_sessions(None, usize::MAX).unwrap().len(), 1);
    assert_eq!(
        store
            .list_session_messages(&sid, usize::MAX, 0)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .search_transcripts("hello", None, None, usize::MAX)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .search_concepts_fts(&memoir, "alpha", usize::MAX)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        store
            .search_all_concepts_fts("alpha", usize::MAX)
            .unwrap()
            .len(),
        1
    );
    assert!(
        store
            .search_concepts_by_label(&memoir, &Label::new("a", "b"), usize::MAX)
            .unwrap()
            .is_empty()
    );
    let (concepts, links) = store.get_neighborhood(&concept, usize::MAX).unwrap();
    assert_eq!((concepts.len(), links.len()), (1, 0));
}

/// A rollout of replicas starts several processes at once on an empty
/// database. Without the setup being serialized, all but one fail on
/// `CREATE EXTENSION`.
#[test]
fn many_first_starts_at_once_all_succeed() {
    let Some(url) = live_database("first_starts") else {
        return;
    };
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(16));
    let handles: Vec<_> = (0..16)
        .map(|_| {
            let (url, barrier) = (url.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                PostgresStore::connect_url(&url, 64, false).map(|_| ())
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap().unwrap();
    }
}

#[test]
fn punctuation_list_has_one_blank_per_character() {
    assert_eq!(PUNCTUATION.replace("''", "'").chars().count(), 32);
    assert_eq!(
        PUNCTUATION.replace("''", "'"),
        (0u8..128)
            .map(char::from)
            .filter(char::is_ascii_punctuation)
            .collect::<String>()
    );
}

#[test]
fn queries_are_split_like_the_indexed_text_but_keep_their_syntax() {
    assert_eq!(tokenize_query("src/lib.rs"), "src lib rs");
    assert_eq!(tokenize_query("state-of-the-art"), "state of the art");
    assert_eq!(tokenize_query("(hello"), " hello");
    assert_eq!(tokenize_query("notes: ^vec*"), "notes   vec ");
    // What websearch_to_tsquery reads as syntax stays.
    assert_eq!(tokenize_query("\"BRIN indexes\""), "\"BRIN indexes\"");
    assert_eq!(tokenize_query("postgres -jsonb"), "postgres -jsonb");
    assert_eq!(tokenize_query("a - b"), "a   b");
    assert_eq!(tokenize_query("postgres or mysql"), "postgres or mysql");
}

/// Words inside paths, filenames and hyphenated terms must be found, as they
/// are on SQLite. PostgreSQL's parser keeps each of those as one token.
#[test]
fn words_inside_paths_and_hyphenated_terms_are_found_everywhere() {
    use icm_core::fusion::RankedQuery;
    let Some(store) = live_store("tokens", 64) else {
        return;
    };
    let mut m = Memory::new(
        "notes".into(),
        "Edit src/lib.rs and file_name.py, follow state-of-the-art rules".into(),
        Importance::Medium,
    );
    m.keywords = vec!["my-tool".into()];
    let id = store.store(m).unwrap();
    store
        .store(Memory::new(
            "notes".into(),
            "unrelated filler".into(),
            Importance::Medium,
        ))
        .unwrap();

    let ranked = |text: &str| -> Vec<String> {
        let q = RankedQuery {
            text,
            embedding: None,
            limit: 5,
            depth: 50,
            window: None,
        };
        store
            .search_ranked(&q, &|_| true)
            .unwrap()
            .into_iter()
            .map(|h| h.memory.id)
            .collect()
    };
    for word in [
        "lib",
        "rs",
        "name",
        "py",
        "art",
        "state",
        "tool",
        "src/lib.rs",
        "file_name.py",
    ] {
        assert_eq!(ranked(word), vec![id.clone()], "ranked search for {word:?}");
    }
    for word in ["lib", "name", "art"] {
        let hits = store.search_fts(word, 5).unwrap();
        assert_eq!(hits.len(), 1, "search_fts for {word:?}");
    }

    let sid = store.create_session("cli", None, None).unwrap();
    store
        .record_message(
            &sid,
            Role::User,
            "see src/lib.rs and file_name.py",
            None,
            None,
            None,
        )
        .unwrap();
    for word in ["lib", "name", "src/lib.rs"] {
        assert_eq!(
            store.search_transcripts(word, None, None, 5).unwrap().len(),
            1,
            "transcript search for {word:?}"
        );
    }

    store
        .store_feedback(Feedback::new(
            "t".into(),
            "edited src/lib.rs".into(),
            "p".into(),
            "use file_name.py".into(),
            None,
            "s".into(),
        ))
        .unwrap();
    assert_eq!(
        store
            .search_feedback("lib name", None, None, 5)
            .unwrap()
            .len(),
        1
    );

    let memoir = store
        .create_memoir(Memoir::new("m".into(), String::new()))
        .unwrap();
    store
        .add_concept(Concept::new(
            memoir.clone(),
            "my-tool".into(),
            "wraps src/lib.rs".into(),
        ))
        .unwrap();
    assert_eq!(
        store
            .search_concepts_fts(&memoir, "tool lib", 5)
            .unwrap()
            .len(),
        1
    );
}

/// A `memories` table made with the old column definition is rebuilt on the
/// next start, and its memories become searchable by their words.
#[test]
fn an_old_full_text_column_is_rebuilt_on_the_next_start() {
    let Some(url) = live_database("fts_migration") else {
        return;
    };
    let store = PostgresStore::connect_url(&url, 64, false).unwrap();
    store
        .store(Memory::new(
            "notes".into(),
            "Edit src/lib.rs today".into(),
            Importance::Medium,
        ))
        .unwrap();
    // The column as it was before words were split on punctuation.
    store
        .conn()
        .unwrap()
        .batch_execute(
            "ALTER TABLE memories DROP COLUMN fts; \
             ALTER TABLE memories ADD COLUMN fts tsvector GENERATED ALWAYS AS ( \
                 to_tsvector('simple', coalesce(topic, '') || ' ' || coalesce(summary, '') \
                     || ' ' || coalesce(keywords, ''))) STORED; \
             CREATE INDEX idx_memories_fts ON memories USING GIN (fts)",
        )
        .unwrap();
    assert!(store.search_fts("lib", 5).unwrap().is_empty());

    let restarted = PostgresStore::connect_url(&url, 64, false).unwrap();
    assert_eq!(restarted.search_fts("lib", 5).unwrap().len(), 1);
    assert_eq!(restarted.count().unwrap(), 1, "no memory was lost");
    let index_valid: bool = restarted
        .conn()
        .unwrap()
        .query_one(
            "SELECT indisvalid FROM pg_index WHERE indexrelid = 'idx_memories_fts'::regclass",
            &[],
        )
        .unwrap()
        .get(0);
    assert!(index_valid);
}

/// `postgres::Error` prints only "db error" for what the server refused. A
/// failed command must say why.
#[test]
fn a_refused_statement_shows_the_servers_message() {
    let Some(store) = live_store("error_text", 64) else {
        return;
    };
    store
        .create_memoir(Memoir::new("same".into(), String::new()))
        .unwrap();
    let err = store
        .create_memoir(Memoir::new("same".into(), String::new()))
        .unwrap_err()
        .to_string();
    assert!(err.contains("duplicate key value"), "{err}");
    assert!(err.contains("SQLSTATE 23505"), "{err}");
}
