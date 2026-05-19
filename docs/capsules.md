# Koa Capsules

Koa capsules are not Docker wrappers. The `koa-capsule` crate uses Linux primitives directly:

- namespaces
- overlayfs
- cgroup v2
- seccomp BPF
- `chroot`
- `execve`

On Windows, `koa doctor` probes the configured WSL2 distribution for the capsule substrate. Direct capsule execution still belongs to a Linux/WSL process because the runtime primitives are Linux syscalls, not Win32 APIs.

## Current v0.0.1 State

Implemented:

- capsule metadata
- overlay upperdir snapshots and restores
- fail-closed doctor checks
- direct Linux exec path
- unsupported-platform hard failures

Not yet implemented:

- Windows-host bridge for launching the Linux capsule runner
- rootfs import command
- network allowlist policy
- persisted per-command stdout/stderr audit records

## Rootfs Requirement

Capsule execution needs a real rootfs with at least `/bin/sh`, `/proc`, and the commands the capsule should execute. Koa will not synthesize a fake rootfs.
