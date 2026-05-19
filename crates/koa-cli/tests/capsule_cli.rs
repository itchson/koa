use std::path::Path;
use std::process::{Command, Output};

#[test]
fn capsule_doctor_reports_missing_rootfs_without_synthesizing_one() {
    let root = tempfile::tempdir().expect("tempdir");
    assert!(koa(root.path(), ["init"]).status.success());
    assert!(!root.path().join(".koa/capsules/base-rootfs").exists());

    let output = koa(root.path(), ["capsule", "doctor"]);
    assert_eq!(output.status.code(), Some(2), "{}", stderr(&output));
    let stdout = stdout(&output);
    assert!(stdout.contains("\"mode\""));
    assert!(stdout.contains("\"ok\": false"));
    assert!(stdout.contains("base_rootfs: fail"));
    assert!(stdout.contains("will not synthesize a fake rootfs"));
}

fn koa<const N: usize>(root: &Path, args: [&str; N]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_koa"))
        .arg("--root")
        .arg(root)
        .args(args)
        .output()
        .expect("run koa")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
