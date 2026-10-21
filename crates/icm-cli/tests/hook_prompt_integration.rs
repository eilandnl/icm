//! End-to-end tests for `icm hook prompt` (UserPromptSubmit): the
//! `[recall]` settings decide whether memories are injected and how many.
//!
//! Spawns the compiled `icm` binary with a private `HOME=<tempdir>`, an
//! explicit `--db` and `ICM_CONFIG`, so the user's real configuration and
//! store are never touched. Embeddings are off: the hook recalls without an
//! embedder.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

/// Path to the `icm` binary built for the current test run.
const ICM: &str = env!("CARGO_BIN_EXE_icm");

fn icm(
    home: &Path,
    config: Option<&Path>,
    args: &[&str],
    stdin: Option<&str>,
) -> std::process::Output {
    let mut cmd = Command::new(ICM);
    cmd.env("HOME", home)
        .env("ICM_NO_EMBEDDINGS", "1")
        .env_remove("ICM_DB")
        .env_remove("ICM_CONFIG")
        .env_remove("XDG_DATA_HOME")
        .env_remove("XDG_CONFIG_HOME");
    if let Some(config) = config {
        cmd.env("ICM_CONFIG", config);
    }
    let mut child = cmd
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn icm");
    if let Some(input) = stdin {
        child
            .stdin
            .take()
            .expect("stdin")
            .write_all(input.as_bytes())
            .expect("write stdin");
    }
    child.wait_with_output().expect("wait icm")
}

/// A store with eight memories of project `myproj` that all match the
/// prompt, worded apart so the render-side paraphrase filter keeps them.
fn seeded(home: &Path) -> String {
    let db = home.join("memories.db").to_str().unwrap().to_string();
    let facts = [
        "kubernetes ingress terminates TLS with cert-manager issuer letsencrypt",
        "kubernetes namespace billing runs postgres sixteen replica pair",
        "kubernetes deployment dashboard listens port 8080 behind traefik",
        "kubernetes cronjob nightly backup writes bucket archives europe",
        "kubernetes secret oauth client rotated quarterly by platform team",
        "kubernetes node pool spot instances drained before upgrades",
        "kubernetes helm chart pinned version 4.2 for redis cache",
        "kubernetes horizontal autoscaler targets seventy percent cpu",
    ];
    for fact in facts {
        let out = icm(
            home,
            None,
            &["--db", &db, "store", "-t", "decisions-myproj", "-c", fact],
            None,
        );
        assert!(
            out.status.success(),
            "store failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    db
}

/// Memories injected by the hook for a prompt matching all of them.
fn injected(home: &Path, db: &str, config: Option<&Path>) -> usize {
    let input = r#"{"user_message":"what do we know about kubernetes here?","cwd":"/work/myproj"}"#;
    let out = icm(home, config, &["--db", db, "hook", "prompt"], Some(input));
    assert!(
        out.status.success(),
        "hook prompt failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| l.starts_with("- kubernetes"))
        .count()
}

#[test]
fn hook_prompt_injects_recall_limit_memories() {
    let tmp = tempfile::tempdir().unwrap();
    let db = seeded(tmp.path());
    let config = tmp.path().join("config.toml");
    std::fs::write(&config, "[recall]\nlimit = 3\n").unwrap();
    assert_eq!(injected(tmp.path(), &db, Some(&config)), 3);
}

#[test]
fn hook_prompt_injects_nothing_when_recall_is_disabled() {
    let tmp = tempfile::tempdir().unwrap();
    let db = seeded(tmp.path());
    let config = tmp.path().join("config.toml");
    std::fs::write(&config, "[recall]\nenabled = false\n").unwrap();
    assert_eq!(injected(tmp.path(), &db, Some(&config)), 0);
}

#[test]
fn hook_prompt_injects_five_memories_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let db = seeded(tmp.path());
    assert_eq!(injected(tmp.path(), &db, None), 5);
}
