use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::doctor::{Doctor, DoctorReport};
use crate::error::{CapsuleError, Result};

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CgroupLimits {
    pub memory_max_bytes: Option<u64>,
    pub pids_max: Option<u64>,
    pub cpu: Option<CpuLimit>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CpuLimit {
    pub quota_usec: u64,
    pub period_usec: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecRequest {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub env: BTreeMap<String, String>,
    pub clear_env: bool,
    pub limits: CgroupLimits,
}

impl ExecRequest {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: PathBuf::from("/"),
            env: BTreeMap::new(),
            clear_env: false,
            limits: CgroupLimits::default(),
        }
    }

    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = cwd.into();
        self
    }

    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn clear_env(mut self) -> Self {
        self.clear_env = true;
        self
    }

    pub fn limits(mut self, limits: CgroupLimits) -> Self {
        self.limits = limits;
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.program.trim().is_empty() {
            return Err(CapsuleError::InvalidCommand("program is empty".to_owned()));
        }
        reject_nul("program", &self.program)?;
        for arg in &self.args {
            reject_nul("argument", arg)?;
        }
        for (key, value) in &self.env {
            reject_nul("environment key", key)?;
            reject_nul("environment value", value)?;
            if key.is_empty() || key.contains('=') {
                return Err(CapsuleError::InvalidCommand(format!(
                    "invalid environment key `{key}`"
                )));
            }
        }
        if !is_capsule_absolute_path(&self.cwd) {
            return Err(CapsuleError::InvalidCommand(
                "cwd must be an absolute path inside the capsule root".to_owned(),
            ));
        }
        if let Some(cpu) = &self.limits.cpu {
            if cpu.quota_usec == 0 || cpu.period_usec == 0 {
                return Err(CapsuleError::InvalidCommand(
                    "cpu quota and period must be positive".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn doctor_report(&self) -> DoctorReport {
        Doctor::run()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapsuleExit {
    pub code: Option<i32>,
    pub signal: Option<i32>,
}

impl CapsuleExit {
    pub fn exited(code: i32) -> Self {
        Self {
            code: Some(code),
            signal: None,
        }
    }

    pub fn signaled(signal: i32) -> Self {
        Self {
            code: None,
            signal: Some(signal),
        }
    }

    pub fn success(&self) -> bool {
        self.code == Some(0) && self.signal.is_none()
    }
}

fn reject_nul(label: &str, value: &str) -> Result<()> {
    if value.as_bytes().contains(&0) {
        Err(CapsuleError::InvalidCommand(format!(
            "{label} contains a NUL byte"
        )))
    } else {
        Ok(())
    }
}

fn is_capsule_absolute_path(path: &std::path::Path) -> bool {
    let text = path.as_os_str().to_string_lossy();
    !text.is_empty()
        && text.starts_with('/')
        && !text.contains('\\')
        && !text.split('/').any(|part| part == "..")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_rejects_nul_bytes() {
        let request = ExecRequest::new("sh").arg("bad\0arg");
        assert!(request.validate().is_err());
    }

    #[test]
    fn request_rejects_relative_cwd() {
        let request = ExecRequest::new("sh").cwd("tmp");
        assert!(request.validate().is_err());
    }
}
