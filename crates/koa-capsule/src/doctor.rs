use serde::{Deserialize, Serialize};

use crate::error::{CapsuleError, Result};
use crate::metadata::unix_secs;
use crate::sys;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DoctorCheck {
    pub id: String,
    pub status: CheckStatus,
    pub required: bool,
    pub details: String,
}

impl DoctorCheck {
    pub fn pass(id: &'static str, required: bool, details: impl Into<String>) -> Self {
        Self {
            id: id.to_owned(),
            status: CheckStatus::Pass,
            required,
            details: details.into(),
        }
    }

    pub fn warn(id: &'static str, required: bool, details: impl Into<String>) -> Self {
        Self {
            id: id.to_owned(),
            status: CheckStatus::Warn,
            required,
            details: details.into(),
        }
    }

    pub fn fail(id: &'static str, required: bool, details: impl Into<String>) -> Self {
        Self {
            id: id.to_owned(),
            status: CheckStatus::Fail,
            required,
            details: details.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DoctorReport {
    pub generated_at_unix: u64,
    pub checks: Vec<DoctorCheck>,
}

impl DoctorReport {
    pub fn ensure_prerequisites(&self) -> Result<()> {
        let failures = self
            .checks
            .iter()
            .filter(|check| check.required && check.status == CheckStatus::Fail)
            .map(|check| format!("{} ({})", check.id, check.details))
            .collect::<Vec<_>>();

        if failures.is_empty() {
            Ok(())
        } else {
            Err(CapsuleError::DoctorFailed(failures.join("; ")))
        }
    }

    pub fn required_failures(&self) -> impl Iterator<Item = &DoctorCheck> {
        self.checks
            .iter()
            .filter(|check| check.required && check.status == CheckStatus::Fail)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Doctor;

impl Doctor {
    pub fn run() -> DoctorReport {
        DoctorReport {
            generated_at_unix: unix_secs(),
            checks: sys::doctor_checks(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_prerequisites_fails_closed() {
        let report = DoctorReport {
            generated_at_unix: 1,
            checks: vec![
                DoctorCheck::pass("optional", false, "ok"),
                DoctorCheck::fail("required", true, "missing"),
            ],
        };

        let err = report.ensure_prerequisites().unwrap_err();
        assert!(err.to_string().contains("required"));
    }

    #[test]
    fn warnings_do_not_block_execution() {
        let report = DoctorReport {
            generated_at_unix: 1,
            checks: vec![DoctorCheck::warn("wsl2", false, "generic linux")],
        };

        report.ensure_prerequisites().unwrap();
    }
}
