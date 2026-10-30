//! PostgreSQL backend: verbatim transcripts (sessions + messages).
//!
//! Same model as the SQLite backend. Search is PostgreSQL full-text search
//! through `websearch_to_tsquery`, which reads the query text as `word word`
//! (all must match), `word or word`, `"exact phrase"` and `-word`, and never
//! rejects a query: a malformed one matches loosely or not at all.

use super::*;

/// The tables and indexes, with the full-text vector of a message: its role,
/// content and tool name, split into words on punctuation.
pub(super) fn schema() -> String {
    SCHEMA_TEMPLATE.replace(
        "@FTS@",
        &tokenizable("role || ' ' || content || ' ' || coalesce(tool_name, '')"),
    )
}

const SCHEMA_TEMPLATE: &str = "
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        agent TEXT NOT NULL DEFAULT '',
        project TEXT,
        started_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
    CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);

    CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_name TEXT,
        tokens BIGINT,
        ts TIMESTAMPTZ NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}',
        fts tsvector GENERATED ALWAYS AS (to_tsvector('simple', @FTS@)) STORED
    );
    CREATE INDEX IF NOT EXISTS idx_messages_session_ts ON messages(session_id, ts);
    CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
    CREATE INDEX IF NOT EXISTS idx_messages_fts ON messages USING GIN (fts);";

/// Cap on a single message: transcripts are best-effort logs, so an
/// oversized one is truncated rather than lost.
const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const MAX_TOOL_NAME_BYTES: usize = 512;
/// Cap on the auxiliary metadata of sessions and messages.
const MAX_METADATA_BYTES: usize = 8 * 1024;

const SESSION_COLS: &str = "id, agent, project, started_at, updated_at, metadata";

/// `s` cut to at most `max` bytes on a char boundary, without NUL bytes,
/// which PostgreSQL text cannot hold.
fn clean(s: &str, max: usize) -> String {
    let mut cut = s.len().min(max);
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    s[..cut].replace('\0', "")
}

fn row_to_session(row: &postgres::Row) -> Session {
    session_at(row, 0)
}

/// A [`Session`] from the six [`SESSION_COLS`] columns starting at `from`.
fn session_at(row: &postgres::Row, from: usize) -> Session {
    Session {
        id: row.get(from),
        agent: row.get(from + 1),
        project: row.get(from + 2),
        started_at: row.get(from + 3),
        updated_at: row.get(from + 4),
        metadata: row.get(from + 5),
    }
}

/// A [`Message`] from the eight columns `id, session_id, role, content,
/// tool_name, tokens, ts, metadata` at the start of the row.
fn row_to_message(row: &postgres::Row) -> Message {
    let role: String = row.get(2);
    Message {
        id: row.get(0),
        session_id: row.get(1),
        role: Role::parse(&role).unwrap_or(Role::Tool),
        content: row.get(3),
        tool_name: row.get(4),
        tokens: row.get(5),
        ts: row.get(6),
        metadata: row.get(7),
    }
}

const MESSAGE_COLS: &str = "id, session_id, role, content, tool_name, tokens, ts, metadata";

impl TranscriptStore for PostgresStore {
    fn create_session(
        &self,
        agent: &str,
        project: Option<&str>,
        metadata: Option<&str>,
    ) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("create_session".into()));
        }
        let session = Session::new(
            agent.replace('\0', ""),
            project.map(|p| p.replace('\0', "")),
            metadata.map(|m| clean(m, MAX_METADATA_BYTES)),
        );
        let mut c = self.conn()?;
        c.execute(
            "INSERT INTO sessions (id, agent, project, started_at, updated_at, metadata) \
             VALUES ($1, $2, $3, $4, $5, $6)",
            &[
                &session.id,
                &session.agent,
                &session.project,
                &session.started_at,
                &session.updated_at,
                &session.metadata,
            ],
        )
        .map_err(pg_err)?;
        Ok(session.id)
    }

    fn ensure_session(
        &self,
        id: &str,
        agent: &str,
        project: Option<&str>,
        metadata: Option<&str>,
    ) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("ensure_session".into()));
        }
        let now = Utc::now();
        let metadata = metadata
            .map(|m| clean(m, MAX_METADATA_BYTES))
            .unwrap_or_else(|| "{}".into());
        let mut c = self.conn()?;
        // The first row stays as it is across re-fires: the id is the host
        // agent's own session id, so repeated hook posts land on one row.
        c.execute(
            "INSERT INTO sessions (id, agent, project, started_at, updated_at, metadata) \
             VALUES ($1, $2, $3, $4, $4, $5) ON CONFLICT (id) DO NOTHING",
            &[
                &id,
                &agent.replace('\0', ""),
                &project.map(|p| p.replace('\0', "")),
                &now,
                &metadata,
            ],
        )
        .map_err(pg_err)?;
        Ok(id.to_string())
    }

    fn get_session(&self, id: &str) -> IcmResult<Option<Session>> {
        let mut c = self.conn()?;
        let row = c
            .query_opt(
                &format!("SELECT {SESSION_COLS} FROM sessions WHERE id = $1"),
                &[&id],
            )
            .map_err(pg_err)?;
        Ok(row.as_ref().map(row_to_session))
    }

    fn list_sessions(&self, project: Option<&str>, limit: usize) -> IcmResult<Vec<Session>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {SESSION_COLS} FROM sessions \
                     WHERE ($1::text IS NULL OR project = $1) \
                     ORDER BY updated_at DESC, id DESC LIMIT $2"
                ),
                &[&project, &sql_bound(limit)],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_session).collect())
    }

    fn record_message(
        &self,
        session_id: &str,
        role: Role,
        content: &str,
        tool_name: Option<&str>,
        tokens: Option<i64>,
        metadata: Option<&str>,
    ) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("record_message".into()));
        }
        let message = Message::new(
            session_id.to_string(),
            role,
            clean(content, MAX_MESSAGE_BYTES),
            tool_name.map(|t| clean(t, MAX_TOOL_NAME_BYTES)),
            tokens,
            metadata.map(|m| clean(m, MAX_METADATA_BYTES)),
        );
        let mut c = self.conn()?;
        let mut tx = c.transaction().map_err(pg_err)?;
        // A missing session is friendlier than the raw foreign-key error.
        let updated = tx
            .execute(
                "UPDATE sessions SET updated_at = $1 WHERE id = $2",
                &[&message.ts, &session_id],
            )
            .map_err(pg_err)?;
        if updated == 0 {
            return Err(IcmError::NotFound(format!(
                "session {session_id} does not exist"
            )));
        }
        tx.execute(
            "INSERT INTO messages (id, session_id, role, content, tool_name, tokens, ts, metadata) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            &[
                &message.id,
                &message.session_id,
                &message.role.as_str(),
                &message.content,
                &message.tool_name,
                &message.tokens,
                &message.ts,
                &message.metadata,
            ],
        )
        .map_err(pg_err)?;
        tx.commit().map_err(pg_err)?;
        Ok(message.id)
    }

    fn list_session_messages(
        &self,
        session_id: &str,
        limit: usize,
        offset: usize,
    ) -> IcmResult<Vec<Message>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {MESSAGE_COLS} FROM messages WHERE session_id = $1 \
                     ORDER BY ts ASC, id ASC LIMIT $2 OFFSET $3"
                ),
                &[&session_id, &sql_bound(limit), &sql_bound(offset)],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_message).collect())
    }

    fn search_transcripts(
        &self,
        query: &str,
        session_id: Option<&str>,
        project: Option<&str>,
        limit: usize,
    ) -> IcmResult<Vec<TranscriptHit>> {
        let query = tokenize_query(&query.replace('\0', ""));
        let words = bm25_words(&query);
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "{with}SELECT m.id, m.session_id, m.role, m.content, m.tool_name, m.tokens, \
                            m.ts, m.metadata, \
                            s.id, s.agent, s.project, s.started_at, s.updated_at, s.metadata, \
                            {BM25_SCORE} AS score \
                     FROM messages m JOIN sessions s ON s.id = m.session_id {BM25_JOINS} \
                     WHERE m.fts @@ bq.tsq \
                       AND ($3::text IS NULL OR m.session_id = $3) \
                       AND ($4::text IS NULL OR s.project = $4) \
                     GROUP BY m.id, s.id, bs.n, bs.avgdl \
                     ORDER BY score DESC, m.ts DESC, m.id ASC LIMIT $5",
                    with = bm25_with("messages", "$1", "$2")
                ),
                &[&words, &query, &session_id, &project, &sql_bound(limit)],
            )
            .map_err(pg_err)?;
        Ok(rows
            .iter()
            .map(|row| TranscriptHit {
                message: row_to_message(row),
                session: session_at(row, 8),
                score: row.get(14),
            })
            .collect())
    }

    fn forget_session(&self, id: &str) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("forget_session".into()));
        }
        // The foreign key cascades to the session's messages.
        let mut c = self.conn()?;
        c.execute("DELETE FROM sessions WHERE id = $1", &[&id])
            .map_err(pg_err)?;
        Ok(())
    }

    fn transcript_stats(&self) -> IcmResult<TranscriptStats> {
        let mut c = self.conn()?;
        let totals = c
            .query_one(
                "SELECT (SELECT COUNT(*) FROM sessions)::bigint, \
                        COUNT(*)::bigint, \
                        COALESCE(SUM(octet_length(content)), 0)::bigint, \
                        MIN(ts), MAX(ts) \
                 FROM messages",
                &[],
            )
            .map_err(pg_err)?;
        let counted = |rows: Vec<postgres::Row>| -> Vec<(String, usize)> {
            rows.iter()
                .map(|row| (row.get(0), row.get::<_, i64>(1).max(0) as usize))
                .collect()
        };
        let by_role = counted(
            c.query(
                "SELECT role, COUNT(*)::bigint FROM messages \
                 GROUP BY role ORDER BY 2 DESC, role COLLATE \"C\" ASC",
                &[],
            )
            .map_err(pg_err)?,
        );
        let by_agent = counted(
            c.query(
                "SELECT agent, COUNT(*)::bigint FROM sessions \
                 GROUP BY agent ORDER BY 2 DESC, agent COLLATE \"C\" ASC",
                &[],
            )
            .map_err(pg_err)?,
        );
        let top_sessions = counted(
            c.query(
                "SELECT session_id, COUNT(*)::bigint FROM messages \
                 GROUP BY session_id ORDER BY 2 DESC, session_id COLLATE \"C\" ASC LIMIT 10",
                &[],
            )
            .map_err(pg_err)?,
        );
        let count = |idx: usize| totals.get::<_, i64>(idx).max(0) as usize;
        Ok(TranscriptStats {
            total_sessions: count(0),
            total_messages: count(1),
            total_bytes: totals.get::<_, i64>(2).max(0) as u64,
            by_role,
            by_agent,
            top_sessions,
            oldest: totals.get(3),
            newest: totals.get(4),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::tests::live_store;

    const DIMS: usize = 64;

    fn say(store: &PostgresStore, session: &str, role: Role, content: &str) -> String {
        store
            .record_message(session, role, content, None, None, None)
            .unwrap()
    }

    #[test]
    fn clean_cuts_on_a_char_boundary_and_drops_nul() {
        assert_eq!(clean("abcdef", 3), "abc");
        assert_eq!(clean("héllo", 2), "h", "does not split the two-byte é");
        assert_eq!(clean("a\0b", 10), "ab");
    }

    #[test]
    fn ensure_session_is_idempotent_and_keeps_the_first_row() {
        let Some(store) = live_store("tr_ensure", DIMS) else {
            return;
        };
        let id = store
            .ensure_session("host-session-1", "claude", Some("proj"), Some("{\"a\":1}"))
            .unwrap();
        assert_eq!(id, "host-session-1");
        store
            .ensure_session("host-session-1", "other", None, None)
            .unwrap();

        let session = store.get_session("host-session-1").unwrap().unwrap();
        assert_eq!(session.agent, "claude");
        assert_eq!(session.project.as_deref(), Some("proj"));
        assert_eq!(session.metadata, "{\"a\":1}");
        assert!(store.get_session("missing").unwrap().is_none());
    }

    #[test]
    fn sessions_list_newest_activity_first_and_filter_by_project() {
        let Some(store) = live_store("tr_list", DIMS) else {
            return;
        };
        let a = store.create_session("cli", Some("p1"), None).unwrap();
        let b = store.create_session("cli", Some("p2"), None).unwrap();
        let c = store.create_session("cli", Some("p1"), None).unwrap();
        // A message bumps its session to the front.
        say(&store, &a, Role::User, "hi");

        let all: Vec<String> = store
            .list_sessions(None, 10)
            .unwrap()
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(all, vec![a.clone(), c.clone(), b]);
        let p1: Vec<String> = store
            .list_sessions(Some("p1"), 10)
            .unwrap()
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(p1, vec![a, c]);
        assert_eq!(store.list_sessions(None, 1).unwrap().len(), 1);
    }

    #[test]
    fn messages_come_back_in_order_with_paging_and_need_a_session() {
        let Some(store) = live_store("tr_messages", DIMS) else {
            return;
        };
        let sid = store.create_session("cli", None, None).unwrap();
        for i in 0..5 {
            store
                .record_message(
                    &sid,
                    Role::User,
                    &format!("m{i}"),
                    Some("tool"),
                    Some(7),
                    None,
                )
                .unwrap();
        }
        let contents = |limit, offset| -> Vec<String> {
            store
                .list_session_messages(&sid, limit, offset)
                .unwrap()
                .into_iter()
                .map(|m| m.content)
                .collect()
        };
        assert_eq!(contents(10, 0), vec!["m0", "m1", "m2", "m3", "m4"]);
        assert_eq!(contents(2, 2), vec!["m2", "m3"]);

        let first = &store.list_session_messages(&sid, 1, 0).unwrap()[0];
        assert_eq!(first.tool_name.as_deref(), Some("tool"));
        assert_eq!(first.tokens, Some(7));
        assert_eq!(first.role, Role::User);

        assert!(matches!(
            store.record_message("missing", Role::User, "x", None, None, None),
            Err(IcmError::NotFound(_))
        ));
    }

    #[test]
    fn oversized_and_nul_input_is_cut_not_rejected() {
        let Some(store) = live_store("tr_limits", DIMS) else {
            return;
        };
        let sid = store.create_session("cli", None, None).unwrap();
        let long = format!("start\0{}", "x ".repeat(MAX_MESSAGE_BYTES));
        store
            .record_message(
                &sid,
                Role::Tool,
                &long,
                None,
                None,
                Some(&"m".repeat(20_000)),
            )
            .unwrap();

        let message = &store.list_session_messages(&sid, 1, 0).unwrap()[0];
        assert_eq!(message.content.len(), MAX_MESSAGE_BYTES - 1);
        assert!(message.content.starts_with("start"));
        assert_eq!(message.metadata.len(), MAX_METADATA_BYTES);
    }

    #[test]
    fn search_reads_or_and_phrases_and_scopes_to_session_and_project() {
        let Some(store) = live_store("tr_search", DIMS) else {
            return;
        };
        let db = store
            .create_session("cli", Some("db-debate"), None)
            .unwrap();
        let other = store
            .create_session("cli", Some("elsewhere"), None)
            .unwrap();
        say(
            &store,
            &db,
            Role::Assistant,
            "Postgres 16 supports JSONB and BRIN indexes natively.",
        );
        say(
            &store,
            &db,
            Role::Assistant,
            "MySQL lacks BRIN; its JSON type is stored differently.",
        );
        say(&store, &db, Role::User, "Et SQLite ?");
        say(
            &store,
            &other,
            Role::User,
            "postgres is also discussed here",
        );

        let hits = |query: &str, session: Option<&str>, project: Option<&str>| {
            store
                .search_transcripts(query, session, project, 10)
                .unwrap()
        };
        assert_eq!(hits("postgres OR mysql", None, None).len(), 3);
        // All words must match without an `or`.
        assert_eq!(hits("postgres mysql", None, None).len(), 0);

        let phrase = hits("\"BRIN indexes\"", None, None);
        assert_eq!(phrase.len(), 1);
        assert!(phrase[0].message.content.contains("Postgres"));
        assert_eq!(phrase[0].session.id, db);
        assert!(phrase[0].score > 0.0);

        assert_eq!(hits("postgres", Some(&other), None).len(), 1);
        assert_eq!(hits("postgres", None, Some("db-debate")).len(), 1);
        assert_eq!(hits("postgres -jsonb", None, None).len(), 1);
        // The tool name and role are searched too.
        assert_eq!(hits("user", None, Some("db-debate")).len(), 1);
    }

    #[test]
    fn a_malformed_query_does_not_error() {
        let Some(store) = live_store("tr_malformed", DIMS) else {
            return;
        };
        let sid = store.create_session("cli", None, None).unwrap();
        say(&store, &sid, Role::User, "hello world");
        for query in [
            "hello AND",
            "(hello",
            "hello OR OR",
            "\"unclosed",
            "",
            "  ",
            "-",
        ] {
            assert!(
                store.search_transcripts(query, None, None, 10).is_ok(),
                "query {query:?} must not error"
            );
        }
    }

    #[test]
    fn forgetting_a_session_removes_its_messages_and_stats_count_the_rest() {
        let Some(store) = live_store("tr_stats", DIMS) else {
            return;
        };
        let a = store.create_session("claude", None, None).unwrap();
        let b = store.create_session("codex", None, None).unwrap();
        say(&store, &a, Role::User, "one");
        say(&store, &a, Role::Assistant, "two");
        say(&store, &a, Role::Assistant, "three");
        say(&store, &b, Role::User, "four");

        let stats = store.transcript_stats().unwrap();
        assert_eq!((stats.total_sessions, stats.total_messages), (2, 4));
        assert_eq!(
            stats.total_bytes,
            ("one".len() + "two".len() + "three".len() + "four".len()) as u64
        );
        assert_eq!(
            stats.by_role,
            vec![("assistant".to_string(), 2), ("user".to_string(), 2)]
        );
        assert_eq!(stats.top_sessions[0], (a.clone(), 3));
        assert_eq!(stats.by_agent.len(), 2);
        assert!(stats.oldest <= stats.newest && stats.oldest.is_some());

        store.forget_session(&a).unwrap();
        assert!(store.get_session(&a).unwrap().is_none());
        assert!(store.list_session_messages(&a, 10, 0).unwrap().is_empty());
        let stats = store.transcript_stats().unwrap();
        assert_eq!((stats.total_sessions, stats.total_messages), (1, 1));
        // Forgetting an unknown session is not an error, as on SQLite.
        store.forget_session(&a).unwrap();
    }
}
