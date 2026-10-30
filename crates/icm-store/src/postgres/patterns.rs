//! PostgreSQL backend -- split out of the former monolithic postgres.rs.
//!
//! Neighbor expansion and pattern-mining helpers.

use super::*;

impl PostgresStore {
    // Memory reads used by recall expansion

    /// Fetch many memories by id in one round-trip, deduplicated by id.
    pub fn get_many(&self, ids: &[&str]) -> IcmResult<HashMap<String, Memory>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let id_vec: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!("SELECT {SELECT_COLS} FROM memories WHERE id = ANY($1)"),
                &[&id_vec],
            )
            .map_err(pg_err)?;
        let mut map = HashMap::with_capacity(rows.len());
        for row in &rows {
            let m = row_to_memory(row);
            map.insert(m.id.clone(), m);
        }
        Ok(map)
    }

    /// Expand a scored result set with one hop of related memories.
    /// Backend-agnostic logic mirrored from the SQLite store.
    pub fn expand_with_neighbors(
        &self,
        initial: &[(Memory, f32)],
        max_neighbors: usize,
        hop_discount: f32,
        max_total: usize,
    ) -> IcmResult<Vec<(Memory, f32)>> {
        if max_neighbors == 0 || initial.is_empty() {
            let mut out = initial.to_vec();
            out.truncate(max_total);
            return Ok(out);
        }

        let initial_ids: HashSet<String> = initial.iter().map(|(m, _)| m.id.clone()).collect();

        let mut candidates: Vec<(String, f32)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        'outer: for (mem, score) in initial {
            for neighbor_id in &mem.related_ids {
                if candidates.len() >= max_neighbors {
                    break 'outer;
                }
                if initial_ids.contains(neighbor_id) || !seen.insert(neighbor_id.clone()) {
                    continue;
                }
                candidates.push((neighbor_id.clone(), *score));
            }
        }

        let mut neighbors: Vec<(Memory, f32)> = Vec::new();
        if !candidates.is_empty() {
            let ids: Vec<&str> = candidates.iter().map(|(id, _)| id.as_str()).collect();
            let fetched = self.get_many(&ids)?;
            for (id, parent_score) in candidates {
                if let Some(m) = fetched.get(&id) {
                    neighbors.push((m.clone(), parent_score * hop_discount));
                }
            }
        }

        let mut combined: Vec<(Memory, f32)> = initial.to_vec();
        combined.extend(neighbors);
        combined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        combined.truncate(max_total);
        Ok(combined)
    }

    /// Memories whose topic starts with `topic` (prefix match).
    pub fn get_by_topic_prefix(&self, topic: &str) -> IcmResult<Vec<Memory>> {
        // Audit finding: `topic` (the literal prefix to match) was
        // interpolated unescaped — a topic containing `%`/`_` turned into
        // unintended wildcards within what's meant to be a literal prefix.
        // The trailing `%` is the deliberate "starts with" wildcard and
        // stays outside the escaped portion.
        let pattern = format!("{}%", escape_like_wildcards(topic));
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {SELECT_COLS} FROM memories WHERE topic LIKE $1 ESCAPE '\\' \
                     ORDER BY weight DESC LIMIT 500"
                ),
                &[&pattern],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_memory).collect())
    }

    /// Distinct topics (optionally prefix-filtered) with their counts.
    pub fn list_topics_with_prefix(&self, prefix: Option<&str>) -> IcmResult<Vec<(String, usize)>> {
        let mut c = self.conn()?;
        let rows = match prefix {
            Some(p) => {
                let pattern = format!("{p}%");
                c.query(
                    "SELECT topic, COUNT(*)::bigint FROM memories WHERE topic LIKE $1 \
                     GROUP BY topic ORDER BY topic",
                    &[&pattern],
                )
            }
            None => c.query(
                "SELECT topic, COUNT(*)::bigint FROM memories GROUP BY topic ORDER BY topic",
                &[],
            ),
        }
        .map_err(pg_err)?;
        Ok(rows
            .iter()
            .map(|row| {
                let n: i64 = row.get(1);
                (row.get(0), n.max(0) as usize)
            })
            .collect())
    }

    // Consolidation / patterns

    /// Consolidate a topic once it holds `threshold` non-`critical`
    /// memories: keep the 3 heaviest summaries, merge every keyword, and
    /// replace the memories that were read with that one rollup. Returns
    /// `true` if it consolidated. Mirrors `SqliteStore::auto_consolidate`,
    /// including being lossy by design (off by default).
    pub fn auto_consolidate(&self, topic: &str, threshold: usize) -> IcmResult<bool> {
        self.auto_consolidate_with_embedder(topic, threshold, None)
    }

    /// Same as [`Self::auto_consolidate`], and embeds the rollup when an
    /// embedder is available so vector recall finds it right away.
    pub fn auto_consolidate_with_embedder(
        &self,
        topic: &str,
        threshold: usize,
        embedder: Option<&dyn Embedder>,
    ) -> IcmResult<bool> {
        if self.count_by_topic(topic)? < threshold {
            return Ok(false);
        }

        // `critical` memories are exempt from consolidation, so they
        // neither count toward the threshold nor feed the rollup.
        let mut memories = self.get_by_topic(topic)?;
        memories.retain(|m| !matches!(m.importance, Importance::Critical));
        if memories.is_empty() || memories.len() < threshold {
            return Ok(false);
        }
        memories.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut summary = memories
            .iter()
            .take(3)
            .map(|m| m.summary.as_str())
            .collect::<Vec<_>>()
            .join(" | ");
        // The rollup goes through the same size check as any write.
        if summary.len() > MAX_SUMMARY_BYTES {
            let mut cut = MAX_SUMMARY_BYTES;
            while !summary.is_char_boundary(cut) {
                cut -= 1;
            }
            summary.truncate(cut);
        }

        let mut keywords: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for keyword in memories.iter().flat_map(|m| &m.keywords) {
            if seen.insert(keyword.to_lowercase()) {
                keywords.push(keyword.clone());
            }
        }

        // The rollup stands for the memories read here and only those: one
        // stored since, or past the 500-row read, must not go with them.
        let replaced: Vec<icm_core::ReadMemory> =
            memories.iter().map(icm_core::ReadMemory::from).collect();

        let mut consolidated = Memory::new(topic.into(), summary, Importance::High);
        consolidated.keywords = keywords;
        consolidated.raw_excerpt = Some(format!(
            "auto-consolidated from {} memories",
            memories.len()
        ));
        consolidated.weight = 1.0;

        // An embedder failure costs the rollup its vector, not the rollup.
        if let Some(embedder) = embedder {
            match embedder.embed(&consolidated.embed_text()) {
                Ok(embedding) => consolidated.embedding = Some(embedding),
                Err(e) => tracing::warn!(
                    "auto-consolidate: embedding failed for topic '{topic}': {e}; \
                     consolidated memory will lack vector representation"
                ),
            }
        }

        // `Stale` (a memory changed since the read) writes nothing; the
        // next store tries again.
        let outcome = self.consolidate_ids(topic, &replaced, consolidated)?;
        Ok(matches!(outcome, icm_core::Consolidated::Replaced { .. }))
    }

    /// Pattern mining is not yet available on the PostgreSQL backend.
    pub fn detect_patterns(
        &self,
        _topic: &str,
        _min_cluster_size: usize,
    ) -> IcmResult<Vec<PatternCluster>> {
        Err(IcmError::Unsupported(
            "detect_patterns (use the default SQLite backend)".into(),
        ))
    }

    /// Pattern mining is not yet available on the PostgreSQL backend.
    pub fn extract_pattern_as_concept(
        &self,
        _cluster: &PatternCluster,
        _memoir_id: &str,
    ) -> IcmResult<String> {
        Err(IcmError::Unsupported(
            "extract_pattern_as_concept (use the default SQLite backend)".into(),
        ))
    }
}
