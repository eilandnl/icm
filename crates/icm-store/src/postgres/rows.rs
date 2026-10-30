//! PostgreSQL backend -- split out of the former monolithic postgres.rs.
//!
//! Row/parse helpers shared by every trait-impl submodule here.

use super::*;

pub(crate) fn pg_err(e: postgres::Error) -> IcmError {
    IcmError::Database(describe(&e))
}

/// `e` with the server's own message and its SQLSTATE. `postgres::Error`
/// prints only "db error" for anything the server refused, which left a
/// failed `icm export` or a rejected connection undiagnosable.
pub(crate) fn describe(e: &postgres::Error) -> String {
    match e.as_db_error() {
        Some(db) => format!("{} (SQLSTATE {})", db.message(), db.code().code()),
        None => match std::error::Error::source(e) {
            Some(cause) => format!("{e}: {cause}"),
            None => e.to_string(),
        },
    }
}

/// The 32 ASCII punctuation characters, as the text of a SQL string literal
/// (so the quote is doubled).
pub(crate) const PUNCTUATION: &str = r##"!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~"##;

/// SQL that turns every ASCII punctuation character of `expr` into a space,
/// for `to_tsvector`. PostgreSQL's parser keeps `file_name.py`, `src/lib.rs`
/// and `state-of-the-art` as one token each, so a search for `lib` or `art`
/// found nothing: about one memory in six could not be found by two of its
/// own words. SQLite's FTS5 splits on everything that is not a letter or a
/// digit, and so does this.
pub(crate) fn tokenizable(expr: &str) -> String {
    format!(
        "translate({expr}, '{PUNCTUATION}', '{}')",
        " ".repeat(PUNCTUATION.replace("''", "'").chars().count())
    )
}

/// Query text the way the indexed text is tokenized: ASCII punctuation
/// separates words, except what `websearch_to_tsquery` reads as syntax, a `"`
/// and a `-` that starts a word.
pub(crate) fn tokenize_query(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous: Option<char> = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let starts_word = previous.is_none_or(|p| p.is_whitespace() || p == '"');
        let syntax = c == '"'
            || (c == '-' && starts_word && chars.peek().is_some_and(|next| !next.is_whitespace()));
        out.push(if c.is_ascii_punctuation() && !syntax {
            ' '
        } else {
            c
        });
        previous = Some(c);
    }
    out
}

/// The `WITH` clauses of a BM25 query over the `fts` column of `table`, which
/// the query must alias `m`. `words` and `query` are the placeholders of the
/// words to weigh (plain text, no operators) and of the text for
/// `websearch_to_tsquery` that selects the rows. PostgreSQL has no ranking
/// function that weighs how rare a term is, and `ts_rank_cd` alone put the
/// right memory first in 60% of queries on a real corpus, against 92% for this
/// and 96% for SQLite's FTS5. The statistics come from the table itself: its
/// size and average length (a scan, about 5 ms per 10k rows) and one document
/// count per term from the GIN index.
pub(crate) fn bm25_with(table: &str, words: &str, query: &str) -> String {
    format!(
        "WITH terms AS MATERIALIZED ( \
             SELECT DISTINCT lexeme FROM unnest(to_tsvector('simple', {words}))), \
         dfs AS MATERIALIZED ( \
             SELECT t.lexeme, (SELECT count(*) FROM {table} \
                  WHERE fts @@ plainto_tsquery('simple', t.lexeme))::float8 AS df \
             FROM terms t), \
         bs AS MATERIALIZED ( \
             SELECT count(*)::float8 AS n, avg(length(fts))::float8 AS avgdl FROM {table}), \
         bq AS MATERIALIZED (SELECT websearch_to_tsquery('simple', {query}) AS tsq) "
    )
}

/// The joins that go with [`bm25_with`], after `FROM <table> m`: they bring
/// in the statistics, the query (`bq.tsq`, for the `WHERE`) and each matching
/// row's term frequencies. The caller groups by the primary key of `m`.
pub(crate) const BM25_JOINS: &str = "CROSS JOIN bs CROSS JOIN bq \
     CROSS JOIN LATERAL ( \
         SELECT d.df, array_length(u.positions, 1)::float8 AS tf \
         FROM unnest(m.fts) u JOIN dfs d ON d.lexeme = u.lexeme) bx";

/// BM25 of a row (k1 1.2, b 0.75, as FTS5 does), summed over the query
/// terms. The IDF is the Lucene form `ln(1 + (N - n + 0.5) / (n + 0.5))`, so
/// scores are not comparable with FTS5's, only their order is.
pub(crate) const BM25_SCORE: &str = "SUM(ln(1 + (bs.n - bx.df + 0.5) / (bx.df + 0.5)) \
     * (bx.tf * 2.2) / (bx.tf + 1.2 * (0.25 + 0.75 * length(m.fts) / bs.avgdl)))::float8";

/// What a BM25 query needs to know about `table`.
pub(crate) struct Bm25Stats {
    /// Each distinct word of the query as the index holds it, and the number
    /// of rows that contain it, rarest first.
    pub terms: Vec<(String, f64)>,
    /// Rows in the table.
    pub n: f64,
    /// Average length of the `fts` column, in words.
    pub avgdl: f64,
}

/// The statistics of `words` over the `fts` column of `table`. The average
/// length is an exact scan on a small table and a 5% block sample on a large
/// one: 4 ms instead of 45 ms on 100k rows, and BM25 does not need it exact.
pub(crate) fn bm25_stats(c: &mut Client, table: &str, words: &str) -> IcmResult<Bm25Stats> {
    let rows = c
        .query(
            &format!(
                "WITH t AS MATERIALIZED ( \
                     SELECT DISTINCT lexeme FROM unnest(to_tsvector('simple', $1))) \
                 SELECT lexeme, (SELECT count(*) FROM {table} \
                          WHERE fts @@ plainto_tsquery('simple', lexeme))::float8 AS df, \
                        (SELECT count(*) FROM {table})::float8 \
                 FROM t ORDER BY df, lexeme"
            ),
            &[&words],
        )
        .map_err(pg_err)?;
    let n = rows.first().map_or(0.0, |row| row.get::<_, f64>(2));
    let terms = rows
        .iter()
        .map(|row| (row.get::<_, String>(0), row.get::<_, f64>(1)))
        .collect();
    let sampled: Option<f64> = if n > 20_000.0 {
        c.query_one(
            &format!("SELECT avg(length(fts))::float8 FROM {table} TABLESAMPLE SYSTEM (5)"),
            &[],
        )
        .map_err(pg_err)?
        .get(0)
    } else {
        None
    };
    let avgdl = match sampled {
        Some(avgdl) => avgdl,
        None => c
            .query_one(
                &format!("SELECT coalesce(avg(length(fts)), 1)::float8 FROM {table}"),
                &[],
            )
            .map_err(pg_err)?
            .get(0),
    };
    Ok(Bm25Stats { terms, n, avgdl })
}

/// The words of `text` to weigh in a BM25 query: letters and digits, the
/// `or` between them dropped.
pub(crate) fn bm25_words(text: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty() && !w.eq_ignore_ascii_case("or"))
        .take(100)
        .collect::<Vec<_>>()
        .join(" ")
}

/// A `usize` as a SQL `LIMIT`/`OFFSET` bound. Callers pass `usize::MAX` for
/// "no limit", and `as i64` wraps that to -1, which SQLite reads as no limit
/// but PostgreSQL rejects as negative.
pub(crate) fn sql_bound(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

pub(crate) fn lock_err() -> IcmError {
    IcmError::Database("postgres client mutex poisoned".into())
}

pub(crate) fn source_type(source: &MemorySource) -> &'static str {
    match source {
        MemorySource::ClaudeCode { .. } => "claude_code",
        MemorySource::Conversation { .. } => "conversation",
        MemorySource::Extracted { .. } => "extracted",
        MemorySource::Manual => "manual",
    }
}

pub(crate) fn source_data(source: &MemorySource) -> Option<String> {
    match source {
        MemorySource::Manual => None,
        other => serde_json::to_string(other).ok(),
    }
}

pub(crate) fn parse_source(source_type_str: &str, source_data_str: Option<String>) -> MemorySource {
    match source_type_str {
        "manual" => MemorySource::Manual,
        _ => source_data_str
            .and_then(|d| serde_json::from_str(&d).ok())
            .unwrap_or(MemorySource::Manual),
    }
}

pub(crate) fn importance_rank(i: Importance) -> u8 {
    match i {
        Importance::Critical => 4,
        Importance::High => 3,
        Importance::Medium => 2,
        Importance::Low => 1,
    }
}

pub(crate) fn max_importance(a: Importance, b: Importance) -> Importance {
    if importance_rank(a) >= importance_rank(b) {
        a
    } else {
        b
    }
}

/// SHA-256 over the normalized `(topic, summary)` pair, hex-encoded.
/// Identical normalization to the SQLite backend so dedup hashes match.
pub(crate) fn summary_hash(topic: &str, summary: &str) -> String {
    use sha2::{Digest, Sha256};
    let topic_n = topic.trim().to_lowercase();
    let summary_n: String = summary
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
        .to_lowercase();
    let mut h = Sha256::new();
    h.update(topic_n.as_bytes());
    h.update(b"\0");
    h.update(summary_n.as_bytes());
    icm_core::to_hex(&h.finalize())
}

pub(crate) const MAX_SUMMARY_BYTES: usize = 64 * 1024;
pub(crate) const MAX_TOPIC_BYTES: usize = 256;

/// Escape `%`, `_`, and the escape character itself so a value can be
/// safely wrapped in a LIKE/ILIKE pattern. Pair with `ESCAPE '\'` in the
/// SQL. Mirrors the SQLite backend's `escape_like_wildcards`.
pub(crate) fn escape_like_wildcards(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Validate and normalize a `Memory` before insertion. Mirrors the
/// SQLite backend's `validate_and_normalize`.
pub(crate) fn validate_and_normalize(mut memory: Memory) -> IcmResult<Memory> {
    memory.topic = memory.topic.trim().to_string();

    if memory.topic.is_empty() {
        return Err(IcmError::InvalidInput("topic cannot be empty".into()));
    }
    if memory.summary.trim().is_empty() {
        return Err(IcmError::InvalidInput("summary cannot be empty".into()));
    }
    if memory.topic.contains('\0') {
        return Err(IcmError::InvalidInput(
            "topic must not contain NUL bytes".into(),
        ));
    }
    if memory.summary.contains('\0') {
        return Err(IcmError::InvalidInput(
            "summary must not contain NUL bytes".into(),
        ));
    }
    if memory.topic.contains(['\n', '\r', '\t']) {
        return Err(IcmError::InvalidInput(
            "topic must not contain newline / CR / tab characters".into(),
        ));
    }
    if memory.topic.len() > MAX_TOPIC_BYTES {
        return Err(IcmError::InvalidInput(format!(
            "topic exceeds {MAX_TOPIC_BYTES} bytes"
        )));
    }
    if memory.summary.len() > MAX_SUMMARY_BYTES {
        return Err(IcmError::InvalidInput(format!(
            "summary exceeds {MAX_SUMMARY_BYTES} bytes"
        )));
    }
    Ok(memory)
}

pub(crate) const SELECT_COLS: &str = "id, created_at, updated_at, last_accessed, access_count, weight, \
                           topic, summary, raw_excerpt, keywords, \
                           importance, source_type, source_data, related_ids, embedding";

/// Map a `memories` row (selected via [`SELECT_COLS`]) to a [`Memory`].
pub(crate) fn row_to_memory(row: &postgres::Row) -> Memory {
    let keywords_json: Option<String> = row.get(9);
    let keywords: Vec<String> = keywords_json
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let importance_str: String = row.get(10);
    let importance = importance_str.parse().unwrap_or(Importance::Medium);

    let source_type_str: String = row.get(11);
    let source_data_str: Option<String> = row.get(12);
    let source = parse_source(&source_type_str, source_data_str);

    let related_json: Option<String> = row.get(13);
    let related_ids: Vec<String> = related_json
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    let embedding: Option<Vec<f32>> = row
        .get::<_, Option<pgvector::Vector>>(14)
        .map(|v| v.as_slice().to_vec());

    let access_count: i32 = row.get(4);

    Memory {
        id: row.get(0),
        created_at: row.get(1),
        updated_at: row.get(2),
        last_accessed: row.get(3),
        access_count: access_count.max(0) as u32,
        weight: row.get(5),
        topic: row.get(6),
        summary: row.get(7),
        raw_excerpt: row.get(8),
        keywords,
        importance,
        source,
        related_ids,
        embedding,
        scope: icm_core::Scope::User,
    }
}

/// Insert a memory, or merge metadata into an existing duplicate.
///
/// Dedup contract identical to the SQLite backend: a collision on
/// `summary_hash` alone (which already encodes the topic, Rust-side,
/// Unicode-correct) is ignored and the existing row's id is returned, after
/// merging the caller's importance (take max), keywords (union), and
/// `raw_excerpt` (prefer new) into it.
pub(crate) fn insert_or_merge_memory<C: GenericClient>(
    c: &mut C,
    memory: &Memory,
) -> IcmResult<String> {
    let keywords_json = serde_json::to_string(&memory.keywords)?;
    let related_json = serde_json::to_string(&memory.related_ids)?;
    let st = source_type(&memory.source);
    let sd = source_data(&memory.source);
    let hash = summary_hash(&memory.topic, &memory.summary);
    let importance = memory.importance.to_string();
    let access = memory.access_count as i32;
    let emb: Option<pgvector::Vector> = memory
        .embedding
        .as_ref()
        .map(|e| pgvector::Vector::from(e.clone()));

    let inserted = c
        .query_opt(
            "INSERT INTO memories
             (id, created_at, updated_at, last_accessed, access_count, weight,
              topic, summary, raw_excerpt, keywords, importance,
              source_type, source_data, related_ids, summary_hash, embedding)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
             ON CONFLICT (summary_hash) WHERE summary_hash IS NOT NULL
             DO NOTHING
             RETURNING id",
            &[
                &memory.id,
                &memory.created_at,
                &memory.updated_at,
                &memory.last_accessed,
                &access,
                &memory.weight,
                &memory.topic,
                &memory.summary,
                &memory.raw_excerpt,
                &keywords_json,
                &importance,
                &st,
                &sd,
                &related_json,
                &hash,
                &emb,
            ],
        )
        .map_err(pg_err)?;

    if let Some(row) = inserted {
        return Ok(row.get::<_, String>(0));
    }

    // Dedup hit: merge metadata into the existing row (mirrors SQLite).
    let existing = c
        .query_one(
            "SELECT id, importance, keywords, raw_excerpt FROM memories
             WHERE summary_hash = $1",
            &[&hash],
        )
        .map_err(pg_err)?;

    let existing_id: String = existing.get(0);
    let existing_importance_str: String = existing.get(1);
    let existing_keywords_json: Option<String> = existing.get(2);
    let existing_raw: Option<String> = existing.get(3);

    let existing_importance: Importance = existing_importance_str
        .parse()
        .unwrap_or(Importance::Medium);
    let merged_importance = max_importance(existing_importance, memory.importance);

    let existing_keywords: Vec<String> = existing_keywords_json
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    let mut merged_keywords = existing_keywords.clone();
    for kw in &memory.keywords {
        if !merged_keywords.contains(kw) {
            merged_keywords.push(kw.clone());
        }
    }

    let merged_raw = memory.raw_excerpt.clone().or_else(|| existing_raw.clone());

    let importance_changed = merged_importance != existing_importance;
    let keywords_changed = merged_keywords != existing_keywords;
    let raw_changed = merged_raw != existing_raw;
    if importance_changed || keywords_changed || raw_changed {
        let merged_keywords_json = serde_json::to_string(&merged_keywords)?;
        c.execute(
            "UPDATE memories
             SET importance = $1, keywords = $2, raw_excerpt = $3, updated_at = $4
             WHERE id = $5",
            &[
                &merged_importance.to_string(),
                &merged_keywords_json,
                &merged_raw,
                &Utc::now(),
                &existing_id,
            ],
        )
        .map_err(pg_err)?;
    }

    Ok(existing_id)
}
