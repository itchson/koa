#[cfg(target_os = "linux")]
mod imp {
    use std::collections::{BTreeMap, BTreeSet};
    use std::env;
    use std::ffi::{CString, OsString};
    use std::fs::{self, OpenOptions};
    use std::io::{self, Write};
    use std::os::fd::RawFd;
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Component, Path, PathBuf};
    use std::ptr;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::doctor::DoctorCheck;
    use crate::error::{CapsuleError, IoContext, Result};
    use crate::exec::{CapsuleExit, CgroupLimits, ExecRequest};
    use crate::metadata::CapsuleMetadata;

    const SYNC_START: u8 = b'G';
    const SETUP_NAMESPACE: u8 = 1;
    const SETUP_FORK_PAYLOAD: u8 = 2;
    const SETUP_PRIVATE_MOUNTS: u8 = 3;
    const SETUP_CHROOT: u8 = 4;
    const SETUP_CHDIR: u8 = 5;
    const SETUP_PROC: u8 = 6;
    const SETUP_HOSTNAME: u8 = 7;
    const SETUP_SECCOMP: u8 = 8;
    const SETUP_EXEC: u8 = 9;

    pub(crate) fn doctor_checks() -> Vec<DoctorCheck> {
        let mut checks = Vec::new();

        let release = fs::read_to_string("/proc/sys/kernel/osrelease")
            .unwrap_or_else(|_| "unknown linux kernel".to_owned());
        checks.push(DoctorCheck::pass(
            "linux_kernel",
            true,
            format!("running on {}", release.trim()),
        ));
        if is_wsl2_release(&release) {
            checks.push(DoctorCheck::pass(
                "wsl2_or_linux",
                false,
                "WSL2 kernel detected",
            ));
        } else {
            checks.push(DoctorCheck::warn(
                "wsl2_or_linux",
                false,
                "generic Linux kernel detected",
            ));
        }

        checks.push(check_probe(
            "namespaces",
            true,
            probe_namespaces(),
            "mount, pid, uts, and ipc namespaces can be created",
        ));
        checks.push(check_probe(
            "overlayfs",
            true,
            probe_overlayfs(),
            "overlayfs mounted successfully in a private namespace",
        ));
        checks.push(check_probe(
            "cgroup_v2",
            true,
            probe_cgroup_v2(),
            "cgroup v2 is available and writable from the current cgroup",
        ));
        checks.push(check_probe(
            "seccomp",
            true,
            probe_seccomp(),
            "seccomp filter installed successfully in a probe process",
        ));

        checks
    }

    pub(crate) fn mount_overlay(
        lower: &Path,
        upper: &Path,
        work: &Path,
        merged: &Path,
    ) -> Result<()> {
        fs::create_dir_all(upper).with_path("creating", upper)?;
        fs::create_dir_all(work).with_path("creating", work)?;
        fs::create_dir_all(merged).with_path("creating", merged)?;

        if let Some(info) = mount_info_for(merged)? {
            if info.fs_type == "overlay" {
                return Ok(());
            }
            return Err(CapsuleError::ExecSetup {
                stage: "overlay",
                message: format!(
                    "{} is already mounted as {}",
                    merged.display(),
                    info.fs_type
                ),
            });
        }

        mount_overlay_raw(lower, upper, work, merged).map_err(|source| CapsuleError::Io {
            op: "mounting overlayfs on",
            path: merged.to_path_buf(),
            source,
        })
    }

    pub(crate) fn unmount(target: &Path) -> Result<()> {
        if mount_info_for(target)?.is_none() {
            return Ok(());
        }
        let target_c = cstring_path(target)?;
        let rc = unsafe { libc::umount2(target_c.as_ptr(), 0) };
        if rc == 0 {
            Ok(())
        } else {
            Err(CapsuleError::Io {
                op: "unmounting",
                path: target.to_path_buf(),
                source: io::Error::last_os_error(),
            })
        }
    }

    pub(crate) fn is_mount_point(target: &Path) -> Result<bool> {
        Ok(mount_info_for(target)?.is_some())
    }

    pub(crate) fn exec_capsule(
        metadata: &CapsuleMetadata,
        request: &ExecRequest,
    ) -> Result<CapsuleExit> {
        fs::create_dir_all(metadata.merged_dir.join("proc"))
            .with_path("creating", metadata.merged_dir.join("proc"))?;
        let exec_path = resolve_program(metadata, request)?;
        let cwd = request.cwd.as_path();
        let host_cwd = host_path_in_root(&metadata.merged_dir, cwd);
        if !host_cwd.is_dir() {
            return Err(CapsuleError::InvalidCommand(format!(
                "cwd `{}` does not exist inside capsule root",
                cwd.display()
            )));
        }

        let argv_storage = build_argv(&request.program, &request.args)?;
        let argv_ptrs = ptrs_with_null(&argv_storage);
        let env_storage = build_env(request)?;
        let env_ptrs = ptrs_with_null(&env_storage);
        let exec_path_c = CString::new(exec_path.as_bytes()).map_err(|_| {
            CapsuleError::InvalidCommand("resolved program path contains a NUL byte".to_owned())
        })?;
        let root_c = cstring_path(&metadata.merged_dir)?;
        let cwd_c = CString::new(cwd.as_os_str().as_bytes())
            .map_err(|_| CapsuleError::InvalidCommand("cwd contains a NUL byte".to_owned()))?;

        let sync_pipe = Pipe::new()?;
        let error_pipe = Pipe::new()?;
        set_cloexec(error_pipe.write)?;

        let supervisor_pid = unsafe { libc::fork() };
        if supervisor_pid < 0 {
            return Err(CapsuleError::Io {
                op: "forking",
                path: metadata.merged_dir.clone(),
                source: io::Error::last_os_error(),
            });
        }

        if supervisor_pid == 0 {
            unsafe {
                close_fd(sync_pipe.write);
                close_fd(error_pipe.read);
                supervisor_main(
                    sync_pipe.read,
                    error_pipe.write,
                    root_c.as_ptr(),
                    cwd_c.as_ptr(),
                    exec_path_c.as_ptr(),
                    argv_ptrs.as_ptr(),
                    env_ptrs.as_ptr(),
                );
            }
        }

        close_fd(sync_pipe.read);
        close_fd(error_pipe.write);

        let cgroup = match Cgroup::create(&metadata.id.to_string(), &request.limits) {
            Ok(cgroup) => cgroup,
            Err(err) => {
                close_fd(sync_pipe.write);
                let _ = wait_for_pid(supervisor_pid);
                close_fd(error_pipe.read);
                return Err(err);
            }
        };

        if let Err(err) = cgroup.add_pid(supervisor_pid) {
            close_fd(sync_pipe.write);
            let _ = wait_for_pid(supervisor_pid);
            close_fd(error_pipe.read);
            let _ = cgroup.cleanup();
            return Err(err);
        }

        write_start(sync_pipe.write)?;
        close_fd(sync_pipe.write);

        let wait_status = wait_for_pid(supervisor_pid)?;
        let setup_error = read_setup_error(error_pipe.read);
        close_fd(error_pipe.read);
        let cleanup_result = cgroup.cleanup();

        if let Some((stage, errno)) = setup_error {
            return Err(CapsuleError::ExecSetup {
                stage: stage_name(stage),
                message: io::Error::from_raw_os_error(errno).to_string(),
            });
        }

        cleanup_result?;
        Ok(wait_status_to_exit(wait_status))
    }

    fn check_probe(
        id: &'static str,
        required: bool,
        result: std::result::Result<(), String>,
        success: &'static str,
    ) -> DoctorCheck {
        match result {
            Ok(()) => DoctorCheck::pass(id, required, success),
            Err(details) => DoctorCheck::fail(id, required, details),
        }
    }

    fn is_wsl2_release(release: &str) -> bool {
        let lower = release.to_ascii_lowercase();
        lower.contains("microsoft") || lower.contains("wsl2")
    }

    fn probe_namespaces() -> std::result::Result<(), String> {
        probe_in_child(|| {
            let flags =
                libc::CLONE_NEWNS | libc::CLONE_NEWPID | libc::CLONE_NEWUTS | libc::CLONE_NEWIPC;
            if unsafe { libc::unshare(flags) } == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        })
        .map_err(|err| err.to_string())
    }

    fn probe_overlayfs() -> std::result::Result<(), String> {
        let base = temp_probe_dir("overlay");
        let lower = base.join("lower");
        let upper = base.join("upper");
        let work = base.join("work");
        let merged = base.join("merged");
        let setup = (|| -> io::Result<()> {
            fs::create_dir_all(&lower)?;
            fs::create_dir_all(&upper)?;
            fs::create_dir_all(&work)?;
            fs::create_dir_all(&merged)?;
            fs::write(lower.join(".probe"), b"overlay")?;
            Ok(())
        })();
        if let Err(err) = setup {
            let _ = fs::remove_dir_all(&base);
            return Err(err.to_string());
        }

        let result = probe_in_child(|| {
            if unsafe { libc::unshare(libc::CLONE_NEWNS) } != 0 {
                return Err(io::Error::last_os_error());
            }
            make_mounts_private()?;
            mount_overlay_raw(&lower, &upper, &work, &merged)?;
            let merged_c = cstring_path_io(&merged)?;
            if unsafe { libc::umount2(merged_c.as_ptr(), 0) } != 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });

        let _ = fs::remove_dir_all(&base);
        result.map_err(|err| err.to_string())
    }

    fn probe_cgroup_v2() -> std::result::Result<(), String> {
        let base = current_cgroup_dir().map_err(|err| err.to_string())?;
        let controllers = base.join("cgroup.controllers");
        let controllers_text = fs::read_to_string(&controllers)
            .map_err(|err| format!("could not read {}: {}", controllers.display(), err))?;
        if controllers_text.trim().is_empty() {
            return Err(format!("{} is empty", controllers.display()));
        }

        let probe = base.join(format!(
            "koa-capsule.probe.{}.{}",
            std::process::id(),
            now_nanos()
        ));
        fs::create_dir(&probe).map_err(|err| {
            format!(
                "could not create delegated cgroup {}: {}",
                probe.display(),
                err
            )
        })?;
        fs::remove_dir(&probe).map_err(|err| {
            format!(
                "could not remove delegated cgroup {}: {}",
                probe.display(),
                err
            )
        })?;
        Ok(())
    }

    fn probe_seccomp() -> std::result::Result<(), String> {
        probe_in_child(apply_seccomp_allow_all).map_err(|err| err.to_string())
    }

    fn probe_in_child<F>(probe: F) -> io::Result<()>
    where
        F: FnOnce() -> io::Result<()>,
    {
        let pipe = Pipe::new_io()?;
        let pid = unsafe { libc::fork() };
        if pid < 0 {
            return Err(io::Error::last_os_error());
        }
        if pid == 0 {
            unsafe {
                close_fd(pipe.read);
                let errno = match probe() {
                    Ok(()) => 0,
                    Err(err) => err.raw_os_error().unwrap_or(libc::EIO),
                };
                let bytes = errno.to_ne_bytes();
                let _ = libc::write(pipe.write, bytes.as_ptr().cast(), bytes.len());
                libc::_exit(0);
            }
        }

        close_fd(pipe.write);
        let mut buf = [0u8; 4];
        let mut read_total = 0usize;
        while read_total < buf.len() {
            let rc = unsafe {
                libc::read(
                    pipe.read,
                    buf[read_total..].as_mut_ptr().cast(),
                    buf.len() - read_total,
                )
            };
            if rc == 0 {
                break;
            }
            if rc < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                close_fd(pipe.read);
                let _ = wait_for_pid(pid);
                return Err(err);
            }
            read_total += rc as usize;
        }
        close_fd(pipe.read);
        let _ = wait_for_pid(pid);
        let errno = i32::from_ne_bytes(buf);
        if errno == 0 {
            Ok(())
        } else {
            Err(io::Error::from_raw_os_error(errno))
        }
    }

    fn mount_overlay_raw(lower: &Path, upper: &Path, work: &Path, merged: &Path) -> io::Result<()> {
        let lower = overlay_path(lower)?;
        let upper = overlay_path(upper)?;
        let work = overlay_path(work)?;
        let data = CString::new(format!("lowerdir={lower},upperdir={upper},workdir={work}"))
            .map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "overlay data contains NUL")
            })?;
        let source = CString::new("overlay").unwrap();
        let fstype = CString::new("overlay").unwrap();
        let target = cstring_path_io(merged)?;
        let rc = unsafe {
            libc::mount(
                source.as_ptr(),
                target.as_ptr(),
                fstype.as_ptr(),
                0,
                data.as_ptr().cast(),
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    fn overlay_path(path: &Path) -> io::Result<String> {
        let value = path.to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "overlay path is not UTF-8")
        })?;
        if value.contains(',') || value.contains(':') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "overlay paths may not contain comma or colon",
            ));
        }
        Ok(value.to_owned())
    }

    fn make_mounts_private() -> io::Result<()> {
        let slash = CString::new("/").unwrap();
        let rc = unsafe {
            libc::mount(
                ptr::null(),
                slash.as_ptr(),
                ptr::null(),
                (libc::MS_PRIVATE | libc::MS_REC) as libc::c_ulong,
                ptr::null(),
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    fn mount_proc() -> io::Result<()> {
        let source = CString::new("proc").unwrap();
        let target = CString::new("/proc").unwrap();
        let fstype = CString::new("proc").unwrap();
        let rc = unsafe {
            libc::mount(
                source.as_ptr(),
                target.as_ptr(),
                fstype.as_ptr(),
                (libc::MS_NOSUID | libc::MS_NOEXEC | libc::MS_NODEV) as libc::c_ulong,
                ptr::null(),
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    fn mount_info_for(target: &Path) -> Result<Option<MountInfo>> {
        let target = normalize_for_mountinfo(target)?;
        let content = fs::read_to_string("/proc/self/mountinfo")
            .with_path("reading", "/proc/self/mountinfo")?;
        Ok(content
            .lines()
            .filter_map(parse_mountinfo_line)
            .find(|info| info.mount_point == target))
    }

    fn normalize_for_mountinfo(path: &Path) -> Result<PathBuf> {
        if path.exists() {
            path.canonicalize().with_path("canonicalizing", path)
        } else {
            let parent = path.parent().unwrap_or_else(|| Path::new("."));
            let parent = parent.canonicalize().with_path("canonicalizing", parent)?;
            Ok(parent.join(path.file_name().unwrap_or_default()))
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct MountInfo {
        mount_point: PathBuf,
        fs_type: String,
    }

    fn parse_mountinfo_line(line: &str) -> Option<MountInfo> {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        let separator = parts.iter().position(|part| *part == "-")?;
        if parts.len() <= separator + 1 || parts.len() <= 4 {
            return None;
        }
        Some(MountInfo {
            mount_point: PathBuf::from(unescape_mountinfo(parts[4])),
            fs_type: parts[separator + 1].to_owned(),
        })
    }

    fn unescape_mountinfo(value: &str) -> String {
        let bytes = value.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\\' && i + 3 < bytes.len() {
                if let Ok(octal) = u8::from_str_radix(&value[i + 1..i + 4], 8) {
                    out.push(octal);
                    i += 4;
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn resolve_program(metadata: &CapsuleMetadata, request: &ExecRequest) -> Result<String> {
        if request.program.contains('/') {
            let host_path = if request.program.starts_with('/') {
                host_path_in_root(&metadata.merged_dir, Path::new(&request.program))
            } else {
                host_path_in_root(&metadata.merged_dir, &request.cwd).join(&request.program)
            };
            ensure_executable(&host_path)?;
            return Ok(request.program.clone());
        }

        let path = effective_path(request);
        for entry in path.split(':') {
            if entry.is_empty() {
                continue;
            }
            let inside = Path::new(entry).join(&request.program);
            let host = host_path_in_root(&metadata.merged_dir, &inside);
            if is_executable_file(&host) {
                return Ok(inside.to_string_lossy().into_owned());
            }
        }

        Err(CapsuleError::InvalidCommand(format!(
            "program `{}` was not found in capsule PATH",
            request.program
        )))
    }

    fn ensure_executable(path: &Path) -> Result<()> {
        if is_executable_file(path) {
            Ok(())
        } else {
            Err(CapsuleError::InvalidCommand(format!(
                "`{}` is not executable inside capsule root",
                path.display()
            )))
        }
    }

    fn is_executable_file(path: &Path) -> bool {
        fs::metadata(path)
            .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }

    fn host_path_in_root(root: &Path, inside: &Path) -> PathBuf {
        let relative = inside.strip_prefix(Path::new("/")).unwrap_or(inside);
        root.join(relative)
    }

    fn effective_path(request: &ExecRequest) -> String {
        if let Some(path) = request.env.get("PATH") {
            return path.clone();
        }
        if !request.clear_env {
            if let Some(path) = env::var_os("PATH").and_then(|value| value.into_string().ok()) {
                return path;
            }
        }
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_owned()
    }

    fn build_argv(program: &str, args: &[String]) -> Result<Vec<CString>> {
        let mut argv = Vec::with_capacity(args.len() + 1);
        argv.push(cstring_str(program, "program")?);
        for arg in args {
            argv.push(cstring_str(arg, "argument")?);
        }
        Ok(argv)
    }

    fn build_env(request: &ExecRequest) -> Result<Vec<CString>> {
        let mut env_map = BTreeMap::<Vec<u8>, Vec<u8>>::new();
        if !request.clear_env {
            for (key, value) in env::vars_os() {
                if let (Some(key), Some(value)) = (os_string_no_nul(key), os_string_no_nul(value)) {
                    env_map.insert(key, value);
                }
            }
        }
        for (key, value) in &request.env {
            env_map.insert(key.as_bytes().to_vec(), value.as_bytes().to_vec());
        }
        if !env_map.contains_key(b"PATH".as_slice()) {
            env_map.insert(
                b"PATH".to_vec(),
                b"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_vec(),
            );
        }

        env_map
            .into_iter()
            .map(|(key, value)| {
                let mut item = key;
                item.push(b'=');
                item.extend(value);
                CString::new(item).map_err(|_| {
                    CapsuleError::InvalidCommand("environment contains a NUL byte".to_owned())
                })
            })
            .collect()
    }

    fn os_string_no_nul(value: OsString) -> Option<Vec<u8>> {
        let bytes = value.into_vec();
        if bytes.contains(&0) {
            None
        } else {
            Some(bytes)
        }
    }

    fn ptrs_with_null(items: &[CString]) -> Vec<*const libc::c_char> {
        let mut ptrs = items.iter().map(|item| item.as_ptr()).collect::<Vec<_>>();
        ptrs.push(ptr::null());
        ptrs
    }

    fn cstring_str(value: &str, label: &str) -> Result<CString> {
        CString::new(value.as_bytes())
            .map_err(|_| CapsuleError::InvalidCommand(format!("{label} contains a NUL byte")))
    }

    fn cstring_path(path: &Path) -> Result<CString> {
        cstring_path_io(path).map_err(|source| CapsuleError::Io {
            op: "converting path",
            path: path.to_path_buf(),
            source,
        })
    }

    fn cstring_path_io(path: &Path) -> io::Result<CString> {
        CString::new(path.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))
    }

    unsafe fn supervisor_main(
        sync_read: RawFd,
        error_write: RawFd,
        root: *const libc::c_char,
        cwd: *const libc::c_char,
        exec_path: *const libc::c_char,
        argv: *const *const libc::c_char,
        envp: *const *const libc::c_char,
    ) -> ! {
        unsafe {
            if !read_start(sync_read) {
                libc::_exit(126);
            }
            close_fd(sync_read);

            let flags =
                libc::CLONE_NEWNS | libc::CLONE_NEWPID | libc::CLONE_NEWUTS | libc::CLONE_NEWIPC;
            if libc::unshare(flags) != 0 {
                write_setup_error_and_exit(error_write, SETUP_NAMESPACE);
            }
            if make_mounts_private().is_err() {
                write_setup_error_and_exit(error_write, SETUP_PRIVATE_MOUNTS);
            }

            let payload_pid = libc::fork();
            if payload_pid < 0 {
                write_setup_error_and_exit(error_write, SETUP_FORK_PAYLOAD);
            }
            if payload_pid == 0 {
                payload_main(error_write, root, cwd, exec_path, argv, envp);
            }

            close_fd(error_write);
            let status = wait_for_pid_raw(payload_pid).unwrap_or(255 << 8);
            exit_with_wait_status(status);
        }
    }

    unsafe fn payload_main(
        error_write: RawFd,
        root: *const libc::c_char,
        cwd: *const libc::c_char,
        exec_path: *const libc::c_char,
        argv: *const *const libc::c_char,
        envp: *const *const libc::c_char,
    ) -> ! {
        unsafe {
            if libc::chroot(root) != 0 {
                write_setup_error_and_exit(error_write, SETUP_CHROOT);
            }
            if libc::chdir(cwd) != 0 {
                write_setup_error_and_exit(error_write, SETUP_CHDIR);
            }
            if mount_proc().is_err() {
                write_setup_error_and_exit(error_write, SETUP_PROC);
            }
            let hostname = CString::new("koa-capsule").unwrap();
            if libc::sethostname(hostname.as_ptr(), "koa-capsule".len()) != 0 {
                write_setup_error_and_exit(error_write, SETUP_HOSTNAME);
            }
            if apply_seccomp_denylist().is_err() {
                write_setup_error_and_exit(error_write, SETUP_SECCOMP);
            }
            libc::execve(exec_path, argv, envp);
            write_setup_error_and_exit(error_write, SETUP_EXEC);
        }
    }

    unsafe fn read_start(fd: RawFd) -> bool {
        unsafe {
            let mut byte = 0u8;
            loop {
                let rc = libc::read(fd, (&mut byte as *mut u8).cast(), 1);
                if rc == 1 {
                    return byte == SYNC_START;
                }
                if rc == 0 {
                    return false;
                }
                if io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                    return false;
                }
            }
        }
    }

    fn write_start(fd: RawFd) -> Result<()> {
        let byte = [SYNC_START];
        let rc = unsafe { libc::write(fd, byte.as_ptr().cast(), byte.len()) };
        if rc == 1 {
            Ok(())
        } else {
            Err(CapsuleError::Io {
                op: "starting isolated process",
                path: PathBuf::from("sync-pipe"),
                source: io::Error::last_os_error(),
            })
        }
    }

    unsafe fn write_setup_error_and_exit(fd: RawFd, stage: u8) -> ! {
        unsafe {
            let errno = io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO);
            let mut record = [0u8; 8];
            record[0] = stage;
            record[4..8].copy_from_slice(&errno.to_ne_bytes());
            let _ = libc::write(fd, record.as_ptr().cast(), record.len());
            libc::_exit(127);
        }
    }

    fn read_setup_error(fd: RawFd) -> Option<(u8, i32)> {
        let mut record = [0u8; 8];
        let mut read_total = 0usize;
        while read_total < record.len() {
            let rc = unsafe {
                libc::read(
                    fd,
                    record[read_total..].as_mut_ptr().cast(),
                    record.len() - read_total,
                )
            };
            if rc == 0 {
                break;
            }
            if rc < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                break;
            }
            read_total += rc as usize;
        }
        if read_total == record.len() {
            Some((
                record[0],
                i32::from_ne_bytes([record[4], record[5], record[6], record[7]]),
            ))
        } else {
            None
        }
    }

    fn stage_name(stage: u8) -> &'static str {
        match stage {
            SETUP_NAMESPACE => "namespace",
            SETUP_FORK_PAYLOAD => "fork_payload",
            SETUP_PRIVATE_MOUNTS => "private_mounts",
            SETUP_CHROOT => "chroot",
            SETUP_CHDIR => "chdir",
            SETUP_PROC => "proc_mount",
            SETUP_HOSTNAME => "hostname",
            SETUP_SECCOMP => "seccomp",
            SETUP_EXEC => "execve",
            _ => "unknown",
        }
    }

    fn wait_for_pid(pid: libc::pid_t) -> Result<i32> {
        wait_for_pid_raw(pid).map_err(|source| CapsuleError::Io {
            op: "waiting for",
            path: PathBuf::from(format!("pid:{pid}")),
            source,
        })
    }

    fn wait_for_pid_raw(pid: libc::pid_t) -> io::Result<i32> {
        let mut status = 0;
        loop {
            let rc = unsafe { libc::waitpid(pid, &mut status, 0) };
            if rc == pid {
                return Ok(status);
            }
            if rc < 0 {
                let err = io::Error::last_os_error();
                if err.raw_os_error() == Some(libc::EINTR) {
                    continue;
                }
                return Err(err);
            }
        }
    }

    fn wait_status_to_exit(status: i32) -> CapsuleExit {
        if libc::WIFEXITED(status) {
            CapsuleExit::exited(libc::WEXITSTATUS(status))
        } else if libc::WIFSIGNALED(status) {
            CapsuleExit::signaled(libc::WTERMSIG(status))
        } else {
            CapsuleExit::exited(255)
        }
    }

    unsafe fn exit_with_wait_status(status: i32) -> ! {
        unsafe {
            if libc::WIFEXITED(status) {
                libc::_exit(libc::WEXITSTATUS(status));
            }
            if libc::WIFSIGNALED(status) {
                let signal = libc::WTERMSIG(status);
                libc::signal(signal, libc::SIG_DFL);
                libc::raise(signal);
                libc::_exit(128 + signal);
            }
            libc::_exit(255);
        }
    }

    fn apply_seccomp_allow_all() -> io::Result<()> {
        let mut filter = vec![bpf_stmt(
            (libc::BPF_RET | libc::BPF_K) as u16,
            libc::SECCOMP_RET_ALLOW,
        )];
        install_seccomp_filter(&mut filter)
    }

    fn apply_seccomp_denylist() -> io::Result<()> {
        let mut filter = vec![bpf_stmt(
            (libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16,
            0,
        )];
        for syscall in dangerous_syscalls() {
            filter.push(bpf_jump(
                (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16,
                syscall as u32,
                0,
                1,
            ));
            filter.push(bpf_stmt(
                (libc::BPF_RET | libc::BPF_K) as u16,
                libc::SECCOMP_RET_ERRNO | libc::EPERM as u32,
            ));
        }
        filter.push(bpf_stmt(
            (libc::BPF_RET | libc::BPF_K) as u16,
            libc::SECCOMP_RET_ALLOW,
        ));
        install_seccomp_filter(&mut filter)
    }

    fn dangerous_syscalls() -> Vec<libc::c_long> {
        vec![
            libc::SYS_add_key,
            libc::SYS_keyctl,
            libc::SYS_request_key,
            libc::SYS_bpf,
            libc::SYS_perf_event_open,
            libc::SYS_ptrace,
            libc::SYS_userfaultfd,
            libc::SYS_mount,
            libc::SYS_umount2,
            libc::SYS_pivot_root,
            libc::SYS_open_by_handle_at,
            libc::SYS_init_module,
            libc::SYS_finit_module,
            libc::SYS_delete_module,
            libc::SYS_kexec_load,
            libc::SYS_reboot,
            libc::SYS_swapon,
            libc::SYS_swapoff,
            libc::SYS_sethostname,
            libc::SYS_setdomainname,
            libc::SYS_acct,
        ]
    }

    fn install_seccomp_filter(filter: &mut [libc::sock_filter]) -> io::Result<()> {
        let no_new_privs = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
        if no_new_privs != 0 {
            return Err(io::Error::last_os_error());
        }
        let mut program = libc::sock_fprog {
            len: filter.len() as libc::c_ushort,
            filter: filter.as_mut_ptr(),
        };
        let rc = unsafe {
            libc::prctl(
                libc::PR_SET_SECCOMP,
                libc::SECCOMP_MODE_FILTER,
                &mut program as *mut libc::sock_fprog,
            )
        };
        if rc == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    fn bpf_stmt(code: u16, k: u32) -> libc::sock_filter {
        libc::sock_filter {
            code,
            jt: 0,
            jf: 0,
            k,
        }
    }

    fn bpf_jump(code: u16, k: u32, jt: u8, jf: u8) -> libc::sock_filter {
        libc::sock_filter { code, jt, jf, k }
    }

    struct Cgroup {
        path: PathBuf,
    }

    impl Cgroup {
        fn create(id: &str, limits: &CgroupLimits) -> Result<Self> {
            let base = current_cgroup_dir().map_err(|source| CapsuleError::Io {
                op: "locating current cgroup",
                path: PathBuf::from("/proc/self/cgroup"),
                source,
            })?;
            enable_controllers_if_needed(&base, limits)?;
            let path = base.join(format!(
                "koa-capsule.{id}.{}.{}",
                std::process::id(),
                now_nanos()
            ));
            fs::create_dir(&path).with_path("creating", &path)?;
            let cgroup = Self { path };
            if let Err(err) = cgroup.apply_limits(limits) {
                let _ = cgroup.cleanup();
                return Err(err);
            }
            Ok(cgroup)
        }

        fn apply_limits(&self, limits: &CgroupLimits) -> Result<()> {
            if let Some(memory) = limits.memory_max_bytes {
                write_string(self.path.join("memory.max"), &memory.to_string())?;
            }
            if let Some(pids) = limits.pids_max {
                write_string(self.path.join("pids.max"), &pids.to_string())?;
            }
            if let Some(cpu) = &limits.cpu {
                write_string(
                    self.path.join("cpu.max"),
                    &format!("{} {}", cpu.quota_usec, cpu.period_usec),
                )?;
            }
            Ok(())
        }

        fn add_pid(&self, pid: libc::pid_t) -> Result<()> {
            write_string(self.path.join("cgroup.procs"), &pid.to_string())
        }

        fn cleanup(&self) -> Result<()> {
            fs::remove_dir(&self.path).with_path("removing", &self.path)
        }
    }

    fn enable_controllers_if_needed(base: &Path, limits: &CgroupLimits) -> Result<()> {
        let mut needed = BTreeSet::new();
        if limits.memory_max_bytes.is_some() {
            needed.insert("memory");
        }
        if limits.pids_max.is_some() {
            needed.insert("pids");
        }
        if limits.cpu.is_some() {
            needed.insert("cpu");
        }
        if needed.is_empty() {
            return Ok(());
        }

        let available_path = base.join("cgroup.controllers");
        let available =
            fs::read_to_string(&available_path).with_path("reading", &available_path)?;
        for controller in &needed {
            if !available.split_whitespace().any(|item| item == *controller) {
                return Err(CapsuleError::ExecSetup {
                    stage: "cgroup",
                    message: format!(
                        "controller `{controller}` is not available in {}",
                        base.display()
                    ),
                });
            }
        }
        let subtree = base.join("cgroup.subtree_control");
        let content = needed
            .iter()
            .map(|controller| format!("+{controller}"))
            .collect::<Vec<_>>()
            .join(" ");
        write_string(subtree, &content)
    }

    fn current_cgroup_dir() -> io::Result<PathBuf> {
        let content = fs::read_to_string("/proc/self/cgroup")?;
        for line in content.lines() {
            if let Some(path) = line.strip_prefix("0::") {
                let relative = path.trim_start_matches('/');
                let mut base = PathBuf::from("/sys/fs/cgroup");
                if !relative.is_empty() {
                    let relative_path = Path::new(relative);
                    if relative_path.components().any(|component| {
                        matches!(
                            component,
                            Component::ParentDir | Component::RootDir | Component::Prefix(_)
                        )
                    }) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "cgroup path escapes /sys/fs/cgroup",
                        ));
                    }
                    base.push(relative_path);
                }
                return Ok(base);
            }
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "cgroup v2 entry was not found in /proc/self/cgroup",
        ))
    }

    fn write_string(path: PathBuf, content: &str) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .open(&path)
            .with_path("opening", &path)?;
        file.write_all(content.as_bytes())
            .with_path("writing", &path)
    }

    struct Pipe {
        read: RawFd,
        write: RawFd,
    }

    impl Pipe {
        fn new() -> Result<Self> {
            Self::new_io().map_err(|source| CapsuleError::Io {
                op: "creating pipe",
                path: PathBuf::from("pipe"),
                source,
            })
        }

        fn new_io() -> io::Result<Self> {
            let mut fds = [0; 2];
            if unsafe { libc::pipe(fds.as_mut_ptr()) } == 0 {
                Ok(Self {
                    read: fds[0],
                    write: fds[1],
                })
            } else {
                Err(io::Error::last_os_error())
            }
        }
    }

    fn set_cloexec(fd: RawFd) -> Result<()> {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags < 0 {
            return Err(CapsuleError::Io {
                op: "reading fd flags",
                path: PathBuf::from(format!("fd:{fd}")),
                source: io::Error::last_os_error(),
            });
        }
        if unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } == 0 {
            Ok(())
        } else {
            Err(CapsuleError::Io {
                op: "setting fd flags",
                path: PathBuf::from(format!("fd:{fd}")),
                source: io::Error::last_os_error(),
            })
        }
    }

    fn close_fd(fd: RawFd) {
        unsafe {
            libc::close(fd);
        }
    }

    fn temp_probe_dir(name: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "koa-capsule-{name}-{}-{}",
            std::process::id(),
            now_nanos()
        ))
    }

    fn now_nanos() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_mountinfo_with_escaped_spaces() {
            let line =
                "35 24 0:31 / /tmp/koa\\040capsule rw,relatime - overlay overlay rw,lowerdir=/a";
            let parsed = parse_mountinfo_line(line).unwrap();
            assert_eq!(parsed.mount_point, PathBuf::from("/tmp/koa capsule"));
            assert_eq!(parsed.fs_type, "overlay");
        }

        #[test]
        fn host_paths_are_kept_under_root_for_absolute_inside_paths() {
            let root = Path::new("/capsule/root");
            assert_eq!(
                host_path_in_root(root, Path::new("/usr/bin/sh")),
                PathBuf::from("/capsule/root/usr/bin/sh")
            );
        }

        #[test]
        fn default_path_is_available_when_env_is_cleared() {
            let request = ExecRequest::new("sh").clear_env();
            assert!(effective_path(&request).contains("/usr/bin"));
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use std::path::Path;

    use crate::doctor::DoctorCheck;
    use crate::error::{CapsuleError, Result};
    use crate::exec::{CapsuleExit, ExecRequest};
    use crate::metadata::CapsuleMetadata;

    pub(crate) fn doctor_checks() -> Vec<DoctorCheck> {
        vec![
            DoctorCheck::fail("linux_kernel", true, "capsules require Linux or WSL2"),
            DoctorCheck::fail("namespaces", true, "Linux namespaces are unavailable"),
            DoctorCheck::fail("overlayfs", true, "overlayfs is unavailable"),
            DoctorCheck::fail("cgroup_v2", true, "cgroup v2 is unavailable"),
            DoctorCheck::fail("seccomp", true, "seccomp is unavailable"),
        ]
    }

    pub(crate) fn mount_overlay(
        _lower: &Path,
        _upper: &Path,
        _work: &Path,
        _merged: &Path,
    ) -> Result<()> {
        Err(CapsuleError::UnsupportedPlatform {
            feature: "overlayfs",
        })
    }

    pub(crate) fn unmount(_target: &Path) -> Result<()> {
        Err(CapsuleError::UnsupportedPlatform {
            feature: "overlayfs",
        })
    }

    pub(crate) fn is_mount_point(_target: &Path) -> Result<bool> {
        Ok(false)
    }

    pub(crate) fn exec_capsule(
        _metadata: &CapsuleMetadata,
        _request: &ExecRequest,
    ) -> Result<CapsuleExit> {
        Err(CapsuleError::UnsupportedPlatform {
            feature: "capsule exec",
        })
    }
}

pub(crate) use imp::*;
