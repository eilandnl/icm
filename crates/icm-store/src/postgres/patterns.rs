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

    /// Recurring patterns in a topic: memories whose keyword sets overlap by
    /// more than 0.5 (Jaccard similarity) are grouped, and the groups of at
    /// least `min_cluster_size` are returned, largest first. Mirrors
    /// `SqliteStore::detect_patterns`.
    pub fn detect_patterns(
        &self,
        topic: &str,
        min_cluster_size: usize,
    ) -> IcmResult<Vec<PatternCluster>> {
        let memories = self.get_by_topic(topic)?;
        if memories.len() < min_cluster_size {
            return Ok(Vec::new());
        }

        let keyword_sets: Vec<HashSet<String>> = memories
            .iter()
            .map(|m| m.keywords.iter().map(|k| k.to_lowercase()).collect())
            .collect();

        let n = memories.len();
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut [usize], i: usize) -> usize {
            let mut i = i;
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let union_size = keyword_sets[i].union(&keyword_sets[j]).count();
                if union_size == 0 {
                    continue;
                }
                let shared = keyword_sets[i].intersection(&keyword_sets[j]).count();
                if shared as f32 / union_size as f32 > 0.5 {
                    let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                    if ri != rj {
                        parent[ri] = rj;
                    }
                }
            }
        }

        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            clusters.entry(root).or_default().push(i);
        }

        let mut result: Vec<PatternCluster> = Vec::new();
        for indices in clusters.values().filter(|c| c.len() >= min_cluster_size) {
            // The representative is the heaviest memory of the cluster.
            let Some(&best) = indices.iter().max_by(|&&a, &&b| {
                memories[a]
                    .weight
                    .partial_cmp(&memories[b].weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) else {
                continue;
            };

            let mut keywords: Vec<String> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            for keyword in indices.iter().flat_map(|&i| &memories[i].keywords) {
                if seen.insert(keyword.to_lowercase()) {
                    keywords.push(keyword.clone());
                }
            }

            result.push(PatternCluster {
                representative_summary: memories[best].summary.clone(),
                memory_ids: indices.iter().map(|&i| memories[i].id.clone()).collect(),
                keywords,
                count: indices.len(),
            });
        }

        // Largest first; the same input gives the same order.
        result.sort_by(|a, b| {
            b.count
                .cmp(&a.count)
                .then_with(|| a.memory_ids.cmp(&b.memory_ids))
        });
        Ok(result)
    }

    /// Store a pattern cluster as a concept of a memoir: named after its top
    /// keywords, defined by the representative summary, labelled with up to
    /// five keywords, with a confidence of `0.5 + 0.05 * count` up to 0.9.
    /// Mirrors `SqliteStore::extract_pattern_as_concept`.
    pub fn extract_pattern_as_concept(
        &self,
        cluster: &PatternCluster,
        memoir_id: &str,
    ) -> IcmResult<String> {
        let name = if cluster.keywords.is_empty() {
            // No keywords to name it by: use the start of the first id.
            let first: String = cluster
                .memory_ids
                .first()
                .map(|id| id.chars().take(8).collect())
                .unwrap_or_default();
            format!("pattern-{first}")
        } else {
            cluster
                .keywords
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("-")
        };
        let definition = format!(
            "{} (pattern detected across {} memories)",
            cluster.representative_summary, cluster.count
        );

        let mut concept = Concept::new(memoir_id.into(), name, definition);
        concept.source_memory_ids = cluster.memory_ids.clone();
        concept.confidence = (0.5 + cluster.count as f32 * 0.05).min(0.9);
        concept.labels = cluster
            .keywords
            .iter()
            .take(5)
            .map(|keyword| Label::new("pattern", keyword.as_str()))
            .collect();
        self.add_concept(concept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::tests::live_store;

    const DIMS: usize = 64;

    fn with_keywords(topic: &str, summary: &str, keywords: &[&str], weight: f32) -> Memory {
        let mut m = Memory::new(topic.into(), summary.into(), Importance::Medium);
        m.keywords = keywords.iter().map(|k| (*k).to_string()).collect();
        m.weight = weight;
        m
    }

    #[test]
    fn memories_with_overlapping_keywords_form_clusters_largest_first() {
        let Some(store) = live_store("pat_detect", DIMS) else {
            return;
        };
        // The third overlaps the first two by 3 of 4 keywords, over the 0.5
        // the grouping needs.
        let deploy: [&[&str]; 3] = [
            &["deploy", "k8s", "helm"],
            &["Deploy", "k8s", "helm"],
            &["deploy", "k8s", "helm", "argo"],
        ];
        for (i, kw) in deploy.iter().enumerate() {
            store
                .store(with_keywords("ops", &format!("deploy {i}"), kw, 1.0))
                .unwrap();
        }
        for (i, kw) in [["sql", "index"], ["sql", "index"]].iter().enumerate() {
            store
                .store(with_keywords("ops", &format!("db {i}"), kw, 1.0))
                .unwrap();
        }
        store
            .store(with_keywords("ops", "alone", &["rust"], 1.0))
            .unwrap();
        // A memory with no keywords never joins a cluster.
        store
            .store(with_keywords("ops", "no keywords", &[], 1.0))
            .unwrap();

        let clusters = store.detect_patterns("ops", 2).unwrap();
        assert_eq!(
            clusters.iter().map(|c| c.count).collect::<Vec<_>>(),
            vec![3, 2]
        );
        assert!(clusters[0].keywords.iter().any(|k| k == "helm"));
        assert_eq!(
            clusters[0]
                .keywords
                .iter()
                .filter(|k| k.eq_ignore_ascii_case("deploy"))
                .count(),
            1,
            "keywords are merged case-insensitively"
        );

        // A bigger minimum drops the smaller group; too few memories is empty.
        assert_eq!(store.detect_patterns("ops", 3).unwrap().len(), 1);
        assert!(store.detect_patterns("ops", 4).unwrap().is_empty());
        assert!(store.detect_patterns("missing", 2).unwrap().is_empty());
    }

    #[test]
    fn the_representative_is_the_heaviest_memory() {
        let Some(store) = live_store("pat_rep", DIMS) else {
            return;
        };
        store
            .store(with_keywords("t", "light", &["a", "b"], 0.2))
            .unwrap();
        store
            .store(with_keywords("t", "heavy", &["a", "b"], 0.9))
            .unwrap();
        let clusters = store.detect_patterns("t", 2).unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].representative_summary, "heavy");
    }

    #[test]
    fn a_cluster_becomes_a_labelled_concept() {
        let Some(store) = live_store("pat_concept", DIMS) else {
            return;
        };
        let memoir = store
            .create_memoir(Memoir::new("m".into(), String::new()))
            .unwrap();
        let cluster = PatternCluster {
            representative_summary: "use the staging key".into(),
            memory_ids: vec!["id-1".into(), "id-2".into()],
            keywords: ["a", "b", "c", "d", "e", "f"].map(String::from).to_vec(),
            count: 2,
        };

        let id = store.extract_pattern_as_concept(&cluster, &memoir).unwrap();
        let concept = store.get_concept(&id).unwrap().unwrap();
        assert_eq!(concept.name, "a-b-c");
        assert_eq!(
            concept.definition,
            "use the staging key (pattern detected across 2 memories)"
        );
        assert_eq!(concept.source_memory_ids, vec!["id-1", "id-2"]);
        assert!((concept.confidence - 0.6).abs() < 1e-6);
        assert_eq!(concept.labels.len(), 5);
        assert_eq!(concept.labels[0], Label::new("pattern", "a"));

        let large = PatternCluster {
            count: 20,
            keywords: vec!["x".into()],
            ..cluster.clone()
        };
        let id = store.extract_pattern_as_concept(&large, &memoir).unwrap();
        assert_eq!(store.get_concept(&id).unwrap().unwrap().confidence, 0.9);
    }

    /// A cluster without keywords is named from its first memory id, and one
    /// with no ids at all does not panic.
    #[test]
    fn a_cluster_without_keywords_or_ids_is_still_named() {
        let Some(store) = live_store("pat_noname", DIMS) else {
            return;
        };
        let memoir = store
            .create_memoir(Memoir::new("m".into(), String::new()))
            .unwrap();
        let mut cluster = PatternCluster {
            representative_summary: "s".into(),
            memory_ids: vec!["0123456789".into()],
            keywords: Vec::new(),
            count: 1,
        };
        let id = store.extract_pattern_as_concept(&cluster, &memoir).unwrap();
        assert_eq!(
            store.get_concept(&id).unwrap().unwrap().name,
            "pattern-01234567"
        );

        cluster.memory_ids = Vec::new();
        let id = store.extract_pattern_as_concept(&cluster, &memoir).unwrap();
        assert_eq!(store.get_concept(&id).unwrap().unwrap().name, "pattern-");
    }
}
