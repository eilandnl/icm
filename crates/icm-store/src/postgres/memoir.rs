//! PostgreSQL backend: memoirs, the concept graph.
//!
//! Same model as the SQLite backend: named memoirs holding concepts, and
//! typed links between concepts that must stay acyclic. Where SQLite walks
//! the graph in Rust one query at a time, this backend asks PostgreSQL for
//! a whole walk in one recursive query.

use super::*;

pub(super) const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS memoirs (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        consolidation_threshold INTEGER NOT NULL DEFAULT 50
    );

    CREATE TABLE IF NOT EXISTS concepts (
        id TEXT PRIMARY KEY,
        memoir_id TEXT NOT NULL REFERENCES memoirs(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        definition TEXT NOT NULL,
        labels TEXT NOT NULL DEFAULT '[]',
        confidence REAL NOT NULL DEFAULT 0.5,
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        source_memory_ids TEXT NOT NULL DEFAULT '[]',
        fts tsvector GENERATED ALWAYS AS (
            to_tsvector('simple', name || ' ' || definition || ' ' || labels)
        ) STORED,
        UNIQUE (memoir_id, name)
    );
    CREATE INDEX IF NOT EXISTS idx_concepts_memoir ON concepts(memoir_id);
    CREATE INDEX IF NOT EXISTS idx_concepts_confidence ON concepts(confidence);
    CREATE INDEX IF NOT EXISTS idx_concepts_fts ON concepts USING GIN (fts);

    CREATE TABLE IF NOT EXISTS concept_links (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
        target_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
        relation TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        created_at TIMESTAMPTZ NOT NULL,
        UNIQUE (source_id, target_id, relation),
        CHECK (source_id <> target_id)
    );
    CREATE INDEX IF NOT EXISTS idx_concept_links_source ON concept_links(source_id);
    CREATE INDEX IF NOT EXISTS idx_concept_links_target ON concept_links(target_id);";

const MEMOIR_COLS: &str = "id, name, description, created_at, updated_at, consolidation_threshold";
const CONCEPT_COLS: &str = "id, memoir_id, name, definition, labels, confidence, revision, \
     created_at, updated_at, source_memory_ids";
const LINK_COLS: &str = "id, source_id, target_id, relation, weight, created_at";

fn row_to_memoir(row: &postgres::Row) -> Memoir {
    let threshold: i32 = row.get(5);
    Memoir {
        id: row.get(0),
        name: row.get(1),
        description: row.get(2),
        created_at: row.get(3),
        updated_at: row.get(4),
        consolidation_threshold: threshold.max(0) as u32,
    }
}

fn row_to_concept(row: &postgres::Row) -> Concept {
    let labels: String = row.get(4);
    let sources: String = row.get(9);
    let revision: i32 = row.get(6);
    Concept {
        id: row.get(0),
        memoir_id: row.get(1),
        name: row.get(2),
        definition: row.get(3),
        labels: serde_json::from_str(&labels).unwrap_or_default(),
        confidence: row.get(5),
        revision: revision.max(0) as u32,
        created_at: row.get(7),
        updated_at: row.get(8),
        source_memory_ids: serde_json::from_str(&sources).unwrap_or_default(),
    }
}

fn row_to_link(row: &postgres::Row) -> ConceptLink {
    let relation: String = row.get(3);
    ConceptLink {
        id: row.get(0),
        source_id: row.get(1),
        target_id: row.get(2),
        relation: relation.parse().unwrap_or(Relation::RelatedTo),
        weight: row.get(4),
        created_at: row.get(5),
    }
}

fn not_found_if_none(changed: u64, id: &str) -> IcmResult<()> {
    if changed == 0 {
        return Err(IcmError::NotFound(id.to_string()));
    }
    Ok(())
}

impl MemoirStore for PostgresStore {
    // Memoir CRUD

    fn create_memoir(&self, memoir: Memoir) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("create_memoir".into()));
        }
        let threshold = memoir.consolidation_threshold as i32;
        let mut c = self.conn()?;
        c.execute(
            "INSERT INTO memoirs (id, name, description, created_at, updated_at, \
                                  consolidation_threshold) \
             VALUES ($1, $2, $3, $4, $5, $6)",
            &[
                &memoir.id,
                &memoir.name,
                &memoir.description,
                &memoir.created_at,
                &memoir.updated_at,
                &threshold,
            ],
        )
        .map_err(pg_err)?;
        Ok(memoir.id)
    }

    fn get_memoir(&self, id: &str) -> IcmResult<Option<Memoir>> {
        let mut c = self.conn()?;
        let row = c
            .query_opt(
                &format!("SELECT {MEMOIR_COLS} FROM memoirs WHERE id = $1"),
                &[&id],
            )
            .map_err(pg_err)?;
        Ok(row.as_ref().map(row_to_memoir))
    }

    fn get_memoir_by_name(&self, name: &str) -> IcmResult<Option<Memoir>> {
        let mut c = self.conn()?;
        let row = c
            .query_opt(
                &format!("SELECT {MEMOIR_COLS} FROM memoirs WHERE name = $1"),
                &[&name],
            )
            .map_err(pg_err)?;
        Ok(row.as_ref().map(row_to_memoir))
    }

    fn update_memoir(&self, memoir: &Memoir) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("update_memoir".into()));
        }
        let threshold = memoir.consolidation_threshold as i32;
        let mut c = self.conn()?;
        let changed = c
            .execute(
                "UPDATE memoirs SET name = $2, description = $3, updated_at = $4, \
                 consolidation_threshold = $5 WHERE id = $1",
                &[
                    &memoir.id,
                    &memoir.name,
                    &memoir.description,
                    &memoir.updated_at,
                    &threshold,
                ],
            )
            .map_err(pg_err)?;
        not_found_if_none(changed, &memoir.id)
    }

    fn delete_memoir(&self, id: &str) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("delete_memoir".into()));
        }
        let mut c = self.conn()?;
        let changed = c
            .execute("DELETE FROM memoirs WHERE id = $1", &[&id])
            .map_err(pg_err)?;
        not_found_if_none(changed, id)
    }

    fn list_memoirs(&self) -> IcmResult<Vec<Memoir>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!("SELECT {MEMOIR_COLS} FROM memoirs ORDER BY name COLLATE \"C\" LIMIT 500"),
                &[],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_memoir).collect())
    }

    // Concept CRUD

    fn add_concept(&self, concept: Concept) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("add_concept".into()));
        }
        let labels = serde_json::to_string(&concept.labels)?;
        let sources = serde_json::to_string(&concept.source_memory_ids)?;
        let revision = concept.revision as i32;
        let mut c = self.conn()?;
        c.execute(
            "INSERT INTO concepts (id, memoir_id, name, definition, labels, confidence, \
                                   revision, created_at, updated_at, source_memory_ids) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
            &[
                &concept.id,
                &concept.memoir_id,
                &concept.name,
                &concept.definition,
                &labels,
                &concept.confidence,
                &revision,
                &concept.created_at,
                &concept.updated_at,
                &sources,
            ],
        )
        .map_err(pg_err)?;
        Ok(concept.id)
    }

    fn get_concept(&self, id: &str) -> IcmResult<Option<Concept>> {
        let mut c = self.conn()?;
        let row = c
            .query_opt(
                &format!("SELECT {CONCEPT_COLS} FROM concepts WHERE id = $1"),
                &[&id],
            )
            .map_err(pg_err)?;
        Ok(row.as_ref().map(row_to_concept))
    }

    fn get_concept_by_name(&self, memoir_id: &str, name: &str) -> IcmResult<Option<Concept>> {
        let mut c = self.conn()?;
        let row = c
            .query_opt(
                &format!("SELECT {CONCEPT_COLS} FROM concepts WHERE memoir_id = $1 AND name = $2"),
                &[&memoir_id, &name],
            )
            .map_err(pg_err)?;
        Ok(row.as_ref().map(row_to_concept))
    }

    fn update_concept(&self, concept: &Concept) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("update_concept".into()));
        }
        let labels = serde_json::to_string(&concept.labels)?;
        let sources = serde_json::to_string(&concept.source_memory_ids)?;
        let revision = concept.revision as i32;
        let mut c = self.conn()?;
        let changed = c
            .execute(
                "UPDATE concepts SET memoir_id = $2, name = $3, definition = $4, labels = $5, \
                 confidence = $6, revision = $7, updated_at = $8, source_memory_ids = $9 \
                 WHERE id = $1",
                &[
                    &concept.id,
                    &concept.memoir_id,
                    &concept.name,
                    &concept.definition,
                    &labels,
                    &concept.confidence,
                    &revision,
                    &concept.updated_at,
                    &sources,
                ],
            )
            .map_err(pg_err)?;
        not_found_if_none(changed, &concept.id)
    }

    fn delete_concept(&self, id: &str) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("delete_concept".into()));
        }
        let mut c = self.conn()?;
        let changed = c
            .execute("DELETE FROM concepts WHERE id = $1", &[&id])
            .map_err(pg_err)?;
        not_found_if_none(changed, id)
    }

    // Concept search

    fn list_concepts(&self, memoir_id: &str) -> IcmResult<Vec<Concept>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {CONCEPT_COLS} FROM concepts WHERE memoir_id = $1 \
                     ORDER BY name COLLATE \"C\" LIMIT 1000"
                ),
                &[&memoir_id],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_concept).collect())
    }

    fn search_concepts_fts(
        &self,
        memoir_id: &str,
        query: &str,
        limit: usize,
    ) -> IcmResult<Vec<Concept>> {
        let text = feedback::and_query(query);
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {CONCEPT_COLS} \
                     FROM concepts, websearch_to_tsquery('simple', $1) AS q \
                     WHERE fts @@ q AND memoir_id = $2 \
                     ORDER BY confidence DESC, name COLLATE \"C\" ASC LIMIT $3"
                ),
                &[&text, &memoir_id, &sql_bound(limit)],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_concept).collect())
    }

    fn search_all_concepts_fts(&self, query: &str, limit: usize) -> IcmResult<Vec<Concept>> {
        let text = feedback::and_query(query);
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {CONCEPT_COLS} \
                     FROM concepts, websearch_to_tsquery('simple', $1) AS q \
                     WHERE fts @@ q \
                     ORDER BY confidence DESC, name COLLATE \"C\" ASC, id ASC LIMIT $2"
                ),
                &[&text, &sql_bound(limit)],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_concept).collect())
    }

    fn search_concepts_by_label(
        &self,
        memoir_id: &str,
        label: &Label,
        limit: usize,
    ) -> IcmResult<Vec<Concept>> {
        // JSON containment matches a label whole: namespace and value of the
        // same entry, with `%` and `_` meaning themselves.
        let wanted = serde_json::to_string(&[label])?;
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {CONCEPT_COLS} FROM concepts \
                     WHERE memoir_id = $1 AND labels::jsonb @> $2::text::jsonb \
                     ORDER BY confidence DESC, name COLLATE \"C\" ASC LIMIT $3"
                ),
                &[&memoir_id, &wanted, &sql_bound(limit)],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_concept).collect())
    }

    // Refinement

    fn refine_concept(
        &self,
        id: &str,
        new_definition: &str,
        new_source_ids: &[String],
    ) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("refine_concept".into()));
        }
        let mut c = self.conn()?;
        let mut tx = c.transaction().map_err(pg_err)?;
        // Locked, so two refinements merge their source ids one after the
        // other instead of the second overwriting the first.
        let row = tx
            .query_opt(
                "SELECT confidence, source_memory_ids FROM concepts WHERE id = $1 FOR UPDATE",
                &[&id],
            )
            .map_err(pg_err)?
            .ok_or_else(|| IcmError::NotFound(id.to_string()))?;
        let confidence: f32 = row.get(0);
        let current: String = row.get(1);
        let mut sources: Vec<String> = serde_json::from_str(&current).unwrap_or_default();
        for source in new_source_ids {
            if !sources.contains(source) {
                sources.push(source.clone());
            }
        }
        tx.execute(
            "UPDATE concepts SET definition = $2, revision = revision + 1, confidence = $3, \
             updated_at = $4, source_memory_ids = $5 WHERE id = $1",
            &[
                &id,
                &new_definition,
                &(confidence + 0.1).min(1.0),
                &Utc::now(),
                &serde_json::to_string(&sources)?,
            ],
        )
        .map_err(pg_err)?;
        tx.commit().map_err(pg_err)?;
        Ok(())
    }

    // Graph

    fn add_link(&self, link: ConceptLink) -> IcmResult<String> {
        if self.readonly {
            return Err(IcmError::ReadOnly("add_link".into()));
        }
        if link.source_id == link.target_id {
            return Err(IcmError::InvalidInput(format!(
                "self-link rejected: source and target are the same concept ({})",
                link.source_id
            )));
        }
        let mut c = self.conn()?;
        let mut tx = c.transaction().map_err(pg_err)?;
        // Two replicas each adding half of a cycle would both pass the check
        // below, so link inserts take turns.
        tx.query(
            "SELECT pg_advisory_xact_lock(hashtext('icm-concept-links'))",
            &[],
        )
        .map_err(pg_err)?;
        // The new edge closes a cycle if its source can be reached from its
        // target. `UNION` drops repeats, so a malformed graph cannot loop.
        let closes_cycle = tx
            .query_opt(
                "WITH RECURSIVE reach(id) AS ( \
                     SELECT $1::text \
                     UNION \
                     SELECT l.target_id FROM concept_links l JOIN reach r ON l.source_id = r.id \
                 ) SELECT 1 FROM reach WHERE id = $2",
                &[&link.target_id, &link.source_id],
            )
            .map_err(pg_err)?
            .is_some();
        if closes_cycle {
            return Err(IcmError::InvalidInput(format!(
                "concept link rejected: {} → {} would create a cycle in the graph",
                link.source_id, link.target_id
            )));
        }
        tx.execute(
            "INSERT INTO concept_links (id, source_id, target_id, relation, weight, created_at) \
             VALUES ($1, $2, $3, $4, $5, $6)",
            &[
                &link.id,
                &link.source_id,
                &link.target_id,
                &link.relation.to_string(),
                &link.weight,
                &link.created_at,
            ],
        )
        .map_err(pg_err)?;
        tx.commit().map_err(pg_err)?;
        Ok(link.id)
    }

    fn get_links_from(&self, concept_id: &str) -> IcmResult<Vec<ConceptLink>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {LINK_COLS} FROM concept_links WHERE source_id = $1 \
                     ORDER BY created_at, id"
                ),
                &[&concept_id],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_link).collect())
    }

    fn get_links_to(&self, concept_id: &str) -> IcmResult<Vec<ConceptLink>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {LINK_COLS} FROM concept_links WHERE target_id = $1 \
                     ORDER BY created_at, id"
                ),
                &[&concept_id],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_link).collect())
    }

    fn delete_link(&self, id: &str) -> IcmResult<()> {
        if self.readonly {
            return Err(IcmError::ReadOnly("delete_link".into()));
        }
        let mut c = self.conn()?;
        let changed = c
            .execute("DELETE FROM concept_links WHERE id = $1", &[&id])
            .map_err(pg_err)?;
        not_found_if_none(changed, id)
    }

    fn get_neighbors(
        &self,
        concept_id: &str,
        relation: Option<Relation>,
    ) -> IcmResult<Vec<Concept>> {
        let relation = relation.map(|r| r.to_string());
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {CONCEPT_COLS} FROM concepts WHERE id IN ( \
                         SELECT target_id FROM concept_links \
                          WHERE source_id = $1 AND ($2::text IS NULL OR relation = $2) \
                         UNION \
                         SELECT source_id FROM concept_links \
                          WHERE target_id = $1 AND ($2::text IS NULL OR relation = $2) \
                     ) ORDER BY name COLLATE \"C\" ASC, id ASC"
                ),
                &[&concept_id, &relation],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_concept).collect())
    }

    fn get_neighborhood(
        &self,
        concept_id: &str,
        depth: usize,
    ) -> IcmResult<(Vec<Concept>, Vec<ConceptLink>)> {
        let mut c = self.conn()?;
        // One walk over links in both directions, `depth` steps from the
        // root, each concept at the fewest steps it can be reached in.
        let reached = c
            .query(
                "WITH RECURSIVE walk(id, depth) AS ( \
                     SELECT $1::text, 0 \
                     UNION \
                     SELECT CASE WHEN l.source_id = w.id THEN l.target_id ELSE l.source_id END, \
                            w.depth + 1 \
                     FROM walk w \
                     JOIN concept_links l ON l.source_id = w.id OR l.target_id = w.id \
                     WHERE w.depth < $2 \
                 ) SELECT id, MIN(depth) AS depth FROM walk GROUP BY id \
                   ORDER BY MIN(depth), id",
                &[&concept_id, &i32::try_from(depth).unwrap_or(i32::MAX)],
            )
            .map_err(pg_err)?;
        let ids: Vec<String> = reached.iter().map(|row| row.get(0)).collect();
        let expanded: Vec<String> = reached
            .iter()
            .filter(|row| (row.get::<_, i32>(1) as usize) < depth)
            .map(|row| row.get(0))
            .collect();

        let rows = c
            .query(
                &format!("SELECT {CONCEPT_COLS} FROM concepts WHERE id = ANY($1::text[])"),
                &[&ids],
            )
            .map_err(pg_err)?;
        let mut by_id: HashMap<String, Concept> = rows
            .iter()
            .map(row_to_concept)
            .map(|concept| (concept.id.clone(), concept))
            .collect();
        // Breadth-first order, the root first.
        let concepts: Vec<Concept> = ids.iter().filter_map(|id| by_id.remove(id)).collect();
        if concepts.first().map(|root| root.id.as_str()) != Some(concept_id) {
            return Err(IcmError::NotFound(concept_id.to_string()));
        }

        // Every link touching a concept the walk went out from, once.
        let links = c
            .query(
                &format!(
                    "SELECT {LINK_COLS} FROM concept_links \
                     WHERE source_id = ANY($1::text[]) OR target_id = ANY($1::text[]) \
                     ORDER BY created_at, id"
                ),
                &[&expanded],
            )
            .map_err(pg_err)?;
        Ok((concepts, links.iter().map(row_to_link).collect()))
    }

    // Stats

    fn memoir_stats(&self, memoir_id: &str) -> IcmResult<MemoirStats> {
        let mut c = self.conn()?;
        let totals = c
            .query_one(
                "SELECT COUNT(*)::bigint, COALESCE(AVG(confidence), 0)::float8, \
                        (SELECT COUNT(*) FROM concept_links \
                          WHERE source_id IN (SELECT id FROM concepts WHERE memoir_id = $1))::bigint \
                 FROM concepts WHERE memoir_id = $1",
                &[&memoir_id],
            )
            .map_err(pg_err)?;
        let labels = c
            .query(
                "SELECT labels FROM concepts WHERE memoir_id = $1 AND labels <> '[]'",
                &[&memoir_id],
            )
            .map_err(pg_err)?;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for row in &labels {
            let raw: String = row.get(0);
            if let Ok(parsed) = serde_json::from_str::<Vec<Label>>(&raw) {
                for label in parsed {
                    *counts.entry(label.to_string()).or_insert(0) += 1;
                }
            }
        }
        let mut label_counts: Vec<(String, usize)> = counts.into_iter().collect();
        label_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(MemoirStats {
            total_concepts: totals.get::<_, i64>(0).max(0) as usize,
            total_links: totals.get::<_, i64>(2).max(0) as usize,
            avg_confidence: totals.get::<_, f64>(1) as f32,
            label_counts,
        })
    }

    fn get_links_for_memoir(&self, memoir_id: &str) -> IcmResult<Vec<ConceptLink>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                &format!(
                    "SELECT {LINK_COLS} FROM concept_links \
                     WHERE source_id IN (SELECT id FROM concepts WHERE memoir_id = $1) \
                     ORDER BY created_at, id LIMIT 5000"
                ),
                &[&memoir_id],
            )
            .map_err(pg_err)?;
        Ok(rows.iter().map(row_to_link).collect())
    }

    fn batch_memoir_concept_counts(&self) -> IcmResult<HashMap<String, usize>> {
        let mut c = self.conn()?;
        let rows = c
            .query(
                "SELECT memoir_id, COUNT(*)::bigint FROM concepts GROUP BY memoir_id",
                &[],
            )
            .map_err(pg_err)?;
        Ok(rows
            .iter()
            .map(|row| (row.get(0), row.get::<_, i64>(1).max(0) as usize))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postgres::tests::{live_database, live_store};

    const DIMS: usize = 64;

    fn memoir(store: &PostgresStore, name: &str) -> String {
        store
            .create_memoir(Memoir::new(name.into(), format!("about {name}")))
            .unwrap()
    }

    fn concept(store: &PostgresStore, memoir_id: &str, name: &str, definition: &str) -> String {
        store
            .add_concept(Concept::new(
                memoir_id.into(),
                name.into(),
                definition.into(),
            ))
            .unwrap()
    }

    fn link(store: &PostgresStore, from: &str, to: &str, relation: Relation) -> IcmResult<String> {
        store.add_link(ConceptLink::new(from.into(), to.into(), relation))
    }

    #[test]
    fn memoirs_are_unique_by_name_and_listed_in_name_order() {
        let Some(store) = live_store("mo_crud", DIMS) else {
            return;
        };
        let b = memoir(&store, "b-memoir");
        memoir(&store, "a-memoir");
        assert!(
            store
                .create_memoir(Memoir::new("a-memoir".into(), String::new()))
                .is_err()
        );

        let names: Vec<String> = store
            .list_memoirs()
            .unwrap()
            .into_iter()
            .map(|m| m.name)
            .collect();
        assert_eq!(names, vec!["a-memoir", "b-memoir"]);
        assert_eq!(store.get_memoir_by_name("b-memoir").unwrap().unwrap().id, b);

        let mut changed = store.get_memoir(&b).unwrap().unwrap();
        changed.description = "new".into();
        changed.consolidation_threshold = 7;
        store.update_memoir(&changed).unwrap();
        let again = store.get_memoir(&b).unwrap().unwrap();
        assert_eq!(
            (again.description.as_str(), again.consolidation_threshold),
            ("new", 7)
        );

        store.delete_memoir(&b).unwrap();
        assert!(store.get_memoir(&b).unwrap().is_none());
        assert!(matches!(
            store.delete_memoir(&b),
            Err(IcmError::NotFound(_))
        ));
        assert!(matches!(
            store.update_memoir(&changed),
            Err(IcmError::NotFound(_))
        ));
    }

    #[test]
    fn concepts_are_unique_within_a_memoir_only() {
        let Some(store) = live_store("mo_concepts", DIMS) else {
            return;
        };
        let m1 = memoir(&store, "m1");
        let m2 = memoir(&store, "m2");
        let id = concept(&store, &m1, "rust", "a language");
        concept(&store, &m2, "rust", "same name, other memoir");
        assert!(
            store
                .add_concept(Concept::new(m1.clone(), "rust".into(), "dup".into()))
                .is_err()
        );

        let mut c = store.get_concept(&id).unwrap().unwrap();
        assert_eq!(
            store.get_concept_by_name(&m1, "rust").unwrap().unwrap().id,
            id
        );
        c.definition = "changed".into();
        c.labels = vec![Label::new("domain", "lang")];
        c.source_memory_ids = vec!["mem-1".into()];
        store.update_concept(&c).unwrap();
        let again = store.get_concept(&id).unwrap().unwrap();
        assert_eq!(again.definition, "changed");
        assert_eq!(again.labels, vec![Label::new("domain", "lang")]);
        assert_eq!(again.source_memory_ids, vec!["mem-1"]);

        assert_eq!(store.list_concepts(&m1).unwrap().len(), 1);
        store.delete_concept(&id).unwrap();
        assert!(matches!(
            store.delete_concept(&id),
            Err(IcmError::NotFound(_))
        ));
        assert!(
            store
                .add_concept(Concept::new(
                    "no-such-memoir".into(),
                    "x".into(),
                    "y".into()
                ))
                .is_err()
        );
    }

    #[test]
    fn refining_bumps_revision_and_confidence_and_merges_sources() {
        let Some(store) = live_store("mo_refine", DIMS) else {
            return;
        };
        let m = memoir(&store, "m");
        let id = concept(&store, &m, "c", "v1");
        store
            .refine_concept(&id, "v2", &["a".into(), "b".into()])
            .unwrap();
        store
            .refine_concept(&id, "v3", &["b".into(), "c".into()])
            .unwrap();

        let c = store.get_concept(&id).unwrap().unwrap();
        assert_eq!((c.definition.as_str(), c.revision), ("v3", 3));
        assert!((c.confidence - 0.7).abs() < 1e-6);
        assert_eq!(c.source_memory_ids, vec!["a", "b", "c"]);
        for _ in 0..10 {
            store.refine_concept(&id, "v", &[]).unwrap();
        }
        assert_eq!(store.get_concept(&id).unwrap().unwrap().confidence, 1.0);
        assert!(matches!(
            store.refine_concept("missing", "x", &[]),
            Err(IcmError::NotFound(_))
        ));
    }

    #[test]
    fn links_reject_self_loops_and_cycles_but_allow_a_dag() {
        let Some(store) = live_store("mo_links", DIMS) else {
            return;
        };
        let m = memoir(&store, "m");
        let a = concept(&store, &m, "a", "a");
        let b = concept(&store, &m, "b", "b");
        let c = concept(&store, &m, "c", "c");

        assert!(matches!(
            link(&store, &a, &a, Relation::RelatedTo),
            Err(IcmError::InvalidInput(_))
        ));
        let ab = link(&store, &a, &b, Relation::DependsOn).unwrap();
        link(&store, &b, &c, Relation::DependsOn).unwrap();
        // c -> a closes a -> b -> c -> a.
        assert!(matches!(
            link(&store, &c, &a, Relation::DependsOn),
            Err(IcmError::InvalidInput(_))
        ));
        assert!(matches!(
            link(&store, &b, &a, Relation::RelatedTo),
            Err(IcmError::InvalidInput(_))
        ));
        // A second path, not a cycle.
        link(&store, &a, &c, Relation::PartOf).unwrap();
        // The same edge twice breaks the unique key.
        assert!(link(&store, &a, &b, Relation::DependsOn).is_err());

        assert_eq!(store.get_links_from(&a).unwrap().len(), 2);
        assert_eq!(store.get_links_to(&c).unwrap().len(), 2);
        assert_eq!(store.get_links_for_memoir(&m).unwrap().len(), 3);

        store.delete_link(&ab).unwrap();
        assert!(matches!(store.delete_link(&ab), Err(IcmError::NotFound(_))));
        // With a -> b gone, b -> a is no longer a cycle.
        link(&store, &b, &a, Relation::RelatedTo).unwrap();
    }

    /// Two writers each adding half of a cycle must not both succeed.
    #[test]
    fn concurrent_links_cannot_close_a_cycle() {
        let Some(url) = live_database("mo_race") else {
            return;
        };
        let store = PostgresStore::connect_url(&url, DIMS, false).unwrap();
        let m = memoir(&store, "m");
        let a = concept(&store, &m, "a", "a");
        let b = concept(&store, &m, "b", "b");

        for round in 0..10 {
            // Fresh edges each round.
            let relation = if round % 2 == 0 {
                Relation::DependsOn
            } else {
                Relation::CausedBy
            };
            let (from, to) = if round % 2 == 0 {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            };
            store
                .conn()
                .unwrap()
                .batch_execute("DELETE FROM concept_links")
                .unwrap();
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
            let handles: Vec<_> = [(from.clone(), to.clone()), (to, from)]
                .into_iter()
                .map(|(source, target)| {
                    let url = url.clone();
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        let store = PostgresStore::connect_url(&url, DIMS, false).unwrap();
                        barrier.wait();
                        store
                            .add_link(ConceptLink::new(source, target, relation))
                            .is_ok()
                    })
                })
                .collect();
            let wins = handles
                .into_iter()
                .filter(|_| true)
                .map(|h| h.join().unwrap())
                .filter(|ok| *ok)
                .count();
            assert_eq!(wins, 1, "round {round}: exactly one of the opposite edges");
        }
    }

    #[test]
    fn neighbors_follow_links_in_both_directions_and_filter_by_relation() {
        let Some(store) = live_store("mo_neighbors", DIMS) else {
            return;
        };
        let m = memoir(&store, "m");
        let hub = concept(&store, &m, "hub", "h");
        let up = concept(&store, &m, "up", "u");
        let down = concept(&store, &m, "down", "d");
        concept(&store, &m, "lonely", "l");
        link(&store, &up, &hub, Relation::PartOf).unwrap();
        link(&store, &hub, &down, Relation::DependsOn).unwrap();

        let names = |relation| -> Vec<String> {
            store
                .get_neighbors(&hub, relation)
                .unwrap()
                .into_iter()
                .map(|c| c.name)
                .collect()
        };
        assert_eq!(names(None), vec!["down", "up"]);
        assert_eq!(names(Some(Relation::PartOf)), vec!["up"]);
        assert_eq!(names(Some(Relation::DependsOn)), vec!["down"]);
        assert!(names(Some(Relation::Contradicts)).is_empty());
    }

    #[test]
    fn the_neighborhood_is_walked_to_the_requested_depth() {
        let Some(store) = live_store("mo_hood", DIMS) else {
            return;
        };
        let m = memoir(&store, "m");
        // a -> b -> c -> d, and b <- x
        let ids: Vec<String> = ["a", "b", "c", "d", "x"]
            .iter()
            .map(|n| concept(&store, &m, n, n))
            .collect();
        let [a, b, c, d, x] = [&ids[0], &ids[1], &ids[2], &ids[3], &ids[4]];
        link(&store, a, b, Relation::DependsOn).unwrap();
        link(&store, b, c, Relation::DependsOn).unwrap();
        link(&store, c, d, Relation::DependsOn).unwrap();
        link(&store, x, b, Relation::PartOf).unwrap();

        let shape = |depth| -> (Vec<String>, usize) {
            let (concepts, links) = store.get_neighborhood(b, depth).unwrap();
            (concepts.into_iter().map(|c| c.name).collect(), links.len())
        };
        assert_eq!(shape(0), (vec!["b".to_string()], 0));
        // One step: the root first, then a and x; their links to b.
        assert_eq!(
            shape(1),
            (vec!["b".to_string(), "a".into(), "c".into(), "x".into()], 3)
        );
        // Two steps reach d, and every link is listed once.
        assert_eq!(
            shape(2),
            (
                vec![
                    "b".to_string(),
                    "a".into(),
                    "c".into(),
                    "x".into(),
                    "d".into()
                ],
                4
            )
        );
        assert_eq!(shape(10).0.len(), 5);

        assert!(matches!(
            store.get_neighborhood("missing", 2),
            Err(IcmError::NotFound(_))
        ));
    }

    #[test]
    fn concepts_are_found_by_words_across_and_within_memoirs_and_by_exact_label() {
        let Some(store) = live_store("mo_search", DIMS) else {
            return;
        };
        let m1 = memoir(&store, "m1");
        let m2 = memoir(&store, "m2");
        let low = concept(&store, &m1, "borrow-checker", "rust ownership checker");
        let mut high = Concept::new(m1.clone(), "lifetimes".into(), "rust lifetime rules".into());
        high.confidence = 0.9;
        high.labels = vec![
            Label::new("domain", "rust"),
            Label::new("kind", "100%_done"),
        ];
        let high = store.add_concept(high).unwrap();
        let mut other = Concept::new(m2.clone(), "cargo".into(), "rust build tool".into());
        other.labels = vec![Label::new("domain", "tools"), Label::new("kind", "rust")];
        store.add_concept(other).unwrap();

        let in_m1: Vec<String> = store
            .search_concepts_fts(&m1, "rust", 10)
            .unwrap()
            .into_iter()
            .map(|c| c.id)
            .collect();
        assert_eq!(in_m1, vec![high.clone(), low], "highest confidence first");
        assert_eq!(store.search_all_concepts_fts("rust", 10).unwrap().len(), 3);
        // All words must match.
        assert_eq!(
            store
                .search_all_concepts_fts("rust ownership", 10)
                .unwrap()
                .len(),
            1
        );
        assert!(
            store
                .search_all_concepts_fts("-- **", 10)
                .unwrap()
                .is_empty()
        );
        assert_eq!(store.search_all_concepts_fts("rust", 1).unwrap().len(), 1);

        let by_label = |memoir: &str, label: Label| -> Vec<String> {
            store
                .search_concepts_by_label(memoir, &label, 10)
                .unwrap()
                .into_iter()
                .map(|c| c.id)
                .collect()
        };
        assert_eq!(
            by_label(&m1, Label::new("domain", "rust")),
            vec![high.clone()]
        );
        // The namespace of one label and the value of another do not combine.
        assert!(by_label(&m1, Label::new("domain", "100%_done")).is_empty());
        assert_eq!(by_label(&m1, Label::new("kind", "100%_done")), vec![high]);
        assert!(by_label(&m1, Label::new("kind", "100%_don_")).is_empty());
        assert!(by_label(&m2, Label::new("domain", "rust")).is_empty());
    }

    #[test]
    fn stats_count_concepts_links_confidence_and_labels() {
        let Some(store) = live_store("mo_stats", DIMS) else {
            return;
        };
        let m = memoir(&store, "m");
        let other = memoir(&store, "other");
        let mut a = Concept::new(m.clone(), "a".into(), "a".into());
        a.labels = vec![Label::new("t", "x"), Label::new("t", "y")];
        a.confidence = 0.25;
        let a = store.add_concept(a).unwrap();
        let mut b = Concept::new(m.clone(), "b".into(), "b".into());
        b.labels = vec![Label::new("t", "x")];
        b.confidence = 0.75;
        let b = store.add_concept(b).unwrap();
        concept(&store, &other, "z", "z");
        link(&store, &a, &b, Relation::RelatedTo).unwrap();

        let stats = store.memoir_stats(&m).unwrap();
        assert_eq!((stats.total_concepts, stats.total_links), (2, 1));
        assert!((stats.avg_confidence - 0.5).abs() < 1e-6);
        assert_eq!(
            stats.label_counts,
            vec![("t:x".to_string(), 2), ("t:y".to_string(), 1)]
        );
        let empty = store.memoir_stats("missing").unwrap();
        assert_eq!((empty.total_concepts, empty.avg_confidence), (0, 0.0));

        let counts = store.batch_memoir_concept_counts().unwrap();
        assert_eq!((counts[&m], counts[&other]), (2, 1));
    }

    #[test]
    fn deleting_a_memoir_or_concept_cascades_to_what_hangs_off_it() {
        let Some(store) = live_store("mo_cascade", DIMS) else {
            return;
        };
        let m = memoir(&store, "m");
        let a = concept(&store, &m, "a", "a");
        let b = concept(&store, &m, "b", "b");
        let c = concept(&store, &m, "c", "c");
        link(&store, &a, &b, Relation::RelatedTo).unwrap();
        link(&store, &b, &c, Relation::RelatedTo).unwrap();

        store.delete_concept(&b).unwrap();
        assert!(store.get_links_from(&a).unwrap().is_empty());
        assert!(store.get_links_to(&c).unwrap().is_empty());

        store.delete_memoir(&m).unwrap();
        assert!(store.get_concept(&a).unwrap().is_none());
        assert!(store.list_concepts(&m).unwrap().is_empty());
    }
}
