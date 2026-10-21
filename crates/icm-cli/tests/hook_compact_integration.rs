//! End-to-end test for `icm hook compact` (PreCompact) with the default
//! `[extraction.conversation]`: conversation extraction is opt-in, so the
//! hook queues nothing for the LLM and runs the keyword pass, as before it.
//!
//! Spawns the compiled `icm` binary with a private `HOME=<tempdir>`, an
//! explicit `--db` and `ICM_CONFIG`, so the user's real configuration and
//! store are never touched.

use std::io::Write;
use std::process::{Command, Stdio};

/// Path to the `icm` binary built for the current test run.
const ICM: &str = env!("CARGO_BIN_EXE_icm");

#[test]
fn hook_compact_queues_no_conversation_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("memories.db");
    let config = tmp.path().join("config.toml");
    // An LLM provider is set: only the conversation switch keeps it idle.
    std::fs::write(&config, "[extraction.summarizer]\nprovider = \"claude\"\n").unwrap();
    let transcript = tmp.path().join("t.jsonl");
    let line = |role: &str, text: &str| {
        serde_json::json!({
            "type": role,
            "timestamp": "2026-10-16T09:00:00.000Z",
            "message": {"role": role, "content": text},
        })
        .to_string()
    };
    std::fs::write(
        &transcript,
        format!(
            "{}\n{}\n",
            line("user", "deploy the billing service to the staging cluster"),
            line(
                "assistant",
                "We decided to use PostgreSQL 16 for billing because it supports logical replication."
            )
        ),
    )
    .unwrap();
    let input = serde_json::json!({
        "session_id": "s1",
        "transcript_path": transcript.to_str().unwrap(),
        "cwd": "/work/myproj",
    })
    .to_string();

    let mut child = Command::new(ICM)
        .env("HOME", tmp.path())
        .env("ICM_CONFIG", &config)
        .env("ICM_NO_EMBEDDINGS", "1")
        .env_remove("ICM_DB")
        .env_remove("ICM_WORKER")
        .env_remove("XDG_DATA_HOME")
        .env_remove("XDG_CONFIG_HOME")
        .args(["--db", db.to_str().unwrap(), "hook", "compact"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn icm");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(input.as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait icm");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "hook compact failed: {stderr}");
    assert!(!stderr.contains("conversation excerpts"), "{stderr}");
    assert!(!stderr.contains("worker"), "{stderr}");

    // Nothing queued for the LLM (a dry run drains nothing).
    let out = Command::new(ICM)
        .env("HOME", tmp.path())
        .env("ICM_CONFIG", &config)
        .env("ICM_NO_EMBEDDINGS", "1")
        .env_remove("ICM_DB")
        .args(["--db", db.to_str().unwrap(), "extract-pending", "--dry-run"])
        .output()
        .expect("run icm");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("No pending extractions."), "{stdout}");
}
