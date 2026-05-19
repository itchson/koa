use std::fs;
use std::path::Path;
use std::process::{Command, Output};

const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn model_verify_exits_two_when_assets_are_missing() {
    let root = tempfile::tempdir().expect("tempdir");
    assert!(koa(root.path(), ["init"]).status.success());

    let output = koa(root.path(), ["model", "verify"]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    assert!(stdout(&output).contains("model directory"));
}

#[test]
fn model_manifest_then_verify_succeeds_for_complete_assets() {
    let root = tempfile::tempdir().expect("tempdir");
    assert!(koa(root.path(), ["init"]).status.success());
    seed_model_dir(&root.path().join(".koa/models/gemma-4-E2B-it"));

    let manifest = koa(root.path(), ["model", "manifest", "--revision", REVISION]);
    assert!(manifest.status.success(), "{}", stderr(&manifest));
    assert!(stdout(&manifest).contains(REVISION));

    let verify = koa(root.path(), ["model", "verify"]);
    assert!(verify.status.success(), "{}", stderr(&verify));
    assert!(stdout(&verify).contains("\"manifest_ok\": true"));
}

#[test]
fn model_manifest_rejects_branch_like_revision() {
    let root = tempfile::tempdir().expect("tempdir");
    assert!(koa(root.path(), ["init"]).status.success());
    seed_model_dir(&root.path().join(".koa/models/gemma-4-E2B-it"));

    let output = koa(
        root.path(),
        ["model", "manifest", "--revision", "release-2026a"],
    );
    assert!(!output.status.success());
    assert!(stderr(&output).contains("40-character Hugging Face commit SHA"));
}

fn koa<const N: usize>(root: &Path, args: [&str; N]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_koa"))
        .arg("--root")
        .arg(root)
        .args(args)
        .output()
        .expect("run koa")
}

fn seed_model_dir(model_dir: &Path) {
    fs::create_dir_all(model_dir).expect("model dir");
    fs::write(
        model_dir.join("config.json"),
        br#"{"model_type":"gemma4","text_config":{"model_type":"gemma4_text"}}"#,
    )
    .expect("config");
    fs::write(
        model_dir.join("tokenizer_config.json"),
        br#"{"chat_template":"{{ messages }}"}"#,
    )
    .expect("tokenizer config");
    fs::write(model_dir.join("tokenizer.json"), br#"{"version":"1.0"}"#).expect("tokenizer");
    fs::write(
        model_dir.join("chat_template.jinja"),
        br#"{% for message in messages %}{{ message.content }}{% endfor %}"#,
    )
    .expect("chat template");
    fs::write(
        model_dir.join("generation_config.json"),
        br#"{"max_new_tokens":128}"#,
    )
    .expect("generation config");
    fs::write(
        model_dir.join("processor_config.json"),
        br#"{"processor_class":"Gemma4Processor"}"#,
    )
    .expect("processor config");
    write_test_safetensors(&model_dir.join("model.safetensors"));
}

fn write_test_safetensors(path: &Path) {
    let data = 1.0f32.to_le_bytes();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &data)
        .expect("tensor view");
    let bytes = safetensors::serialize([("weight".to_string(), view)], None).expect("safetensors");
    fs::write(path, bytes).expect("weights");
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
