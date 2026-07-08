// CamillaDSP - A flexible tool for processing audio
// Copyright (C) 2026 Henrik Enquist
//
// This file is part of CamillaDSP.
//
// CamillaDSP is free software; you can redistribute it and/or modify it
// under the terms of either:
//
// a) the GNU General Public License version 3,
//    or
// b) the Mozilla Public License Version 2.0.
//
// You should have received copies of the GNU General Public License and the
// Mozilla Public License along with this program. If not, see
// <https://www.gnu.org/licenses/> and <https://www.mozilla.org/MPL/2.0/>.

//! Thin wrapper around real-time thread priority promotion.
//!
//! On macOS and Windows this re-exports `audio_thread_priority`, which uses the platform-idiomatic
//! audio scheduling APIs (mach time-constraint policy and MMCSS "Pro Audio").
//!
//! On Linux `audio_thread_priority` only promotes threads through the `rtkit` D-Bus service, and is
//! a no-op when built without the `dbus` feature. rtkit is the right choice for unprivileged desktop
//! processes, and it ships as a dependency of both PulseAudio and PipeWire, so when either the
//! `pulse-backend` or `pipewire-backend` feature is enabled (via the `rtkit` feature) we use
//! `audio_thread_priority` with D-Bus.
//!
//! The plain ALSA-only build targets minimal/headless systems that often have no D-Bus at all. There
//! we promote the thread directly with `pthread_setschedparam(SCHED_FIFO)`, which needs no D-Bus and
//! works wherever the process may request real-time scheduling (running as root, holding
//! `CAP_SYS_NICE`, or with an `RLIMIT_RTPRIO` limit configured, e.g. via `/etc/security/limits.conf`).
//!
//! The native path is hopefully temporary: the plan is to contribute it upstream so the no-D-Bus
//! Linux build promotes the thread instead of being a no-op, and then drop it here.

#[cfg(any(not(target_os = "linux"), feature = "rtkit"))]
pub use audio_thread_priority::{
    demote_current_thread_from_real_time, promote_current_thread_to_real_time,
};

#[cfg(all(target_os = "linux", not(feature = "rtkit")))]
pub use native::{demote_current_thread_from_real_time, promote_current_thread_to_real_time};

#[cfg(all(target_os = "linux", not(feature = "rtkit")))]
mod native {
    use std::fmt;

    /// Default real-time priority to request for the SCHED_FIFO policy. Matches the value used by
    /// rtkit (`RT_PRIO_DEFAULT`), and stays well below the kernel IRQ threads (typically around 50)
    /// that feed the audio device. Override with the `CAMILLADSP_RT_PRIORITY` environment variable.
    const DEFAULT_RT_PRIORITY: libc::c_int = 10;

    /// Environment variable used to override the real-time priority. Accepts an integer 1-99. Higher
    /// values preempt more work but must stay below the audio interface's IRQ thread, or they starve
    /// the very threads that deliver audio.
    const RT_PRIORITY_ENV: &str = "CAMILLADSP_RT_PRIORITY";

    /// The real-time priority to request, from `CAMILLADSP_RT_PRIORITY` or the default.
    fn requested_priority() -> libc::c_int {
        match std::env::var(RT_PRIORITY_ENV) {
            Ok(value) => match value.trim().parse::<libc::c_int>() {
                Ok(priority) if (1..=99).contains(&priority) => priority,
                _ => {
                    warn!(
                        "Ignoring invalid {RT_PRIORITY_ENV}=\"{value}\", \
                         expected an integer 1-99. Using default {DEFAULT_RT_PRIORITY}."
                    );
                    DEFAULT_RT_PRIORITY
                }
            },
            Err(_) => DEFAULT_RT_PRIORITY,
        }
    }

    /// Ensures threads/processes forked from a real-time thread do not inherit real-time
    /// scheduling. Not reliably exposed by `libc` across targets, so defined here.
    const SCHED_RESET_ON_FORK: libc::c_int = 0x4000_0000;

    /// Opaque handle holding the scheduling policy and parameters the thread had before promotion,
    /// so they can be restored on demotion.
    pub struct RtPriorityHandle {
        policy: libc::c_int,
        param: libc::sched_param,
    }

    #[derive(Debug)]
    pub struct RtPriorityError {
        message: String,
    }

    impl RtPriorityError {
        /// The `pthread_*` functions return the error number directly and do not set `errno`, so the
        /// return code is converted here rather than reading `errno` via `last_os_error()`.
        fn from_os_error(context: &str, code: libc::c_int) -> Self {
            RtPriorityError {
                message: format!("{context}: {}", std::io::Error::from_raw_os_error(code)),
            }
        }
    }

    impl fmt::Display for RtPriorityError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.message)
        }
    }

    impl std::error::Error for RtPriorityError {}

    /// Promote the calling thread to real-time priority using SCHED_FIFO.
    ///
    /// The buffer size and sample rate are unused here (they matter only for the rtkit path, which
    /// derives an `RLIMIT_RTTIME` budget from them); the signature is kept to match
    /// `audio_thread_priority`.
    pub fn promote_current_thread_to_real_time(
        _audio_buffer_frames: u32,
        _audio_samplerate_hz: u32,
    ) -> Result<RtPriorityHandle, RtPriorityError> {
        let pthread_id = unsafe { libc::pthread_self() };

        // Remember the current policy and parameters so demotion can restore them.
        let mut policy = 0;
        let mut param = unsafe { std::mem::zeroed::<libc::sched_param>() };
        let ret = unsafe { libc::pthread_getschedparam(pthread_id, &mut policy, &mut param) };
        if ret != 0 {
            return Err(RtPriorityError::from_os_error("pthread_getschedparam", ret));
        }
        let handle = RtPriorityHandle { policy, param };

        let mut rt_param = unsafe { std::mem::zeroed::<libc::sched_param>() };
        rt_param.sched_priority = requested_priority();
        let ret = unsafe {
            libc::pthread_setschedparam(
                pthread_id,
                libc::SCHED_FIFO | SCHED_RESET_ON_FORK,
                &rt_param,
            )
        };
        if ret != 0 {
            return Err(RtPriorityError::from_os_error("pthread_setschedparam", ret));
        }
        Ok(handle)
    }

    /// Restore the calling thread to the scheduling policy and parameters it had before promotion.
    pub fn demote_current_thread_from_real_time(
        handle: RtPriorityHandle,
    ) -> Result<(), RtPriorityError> {
        let pthread_id = unsafe { libc::pthread_self() };
        // Promotion set SCHED_RESET_ON_FORK, and the kernel forbids an unprivileged thread from
        // clearing it, so keep the flag when restoring the saved policy or the call fails with EPERM.
        let policy = handle.policy | SCHED_RESET_ON_FORK;
        let ret = unsafe { libc::pthread_setschedparam(pthread_id, policy, &handle.param) };
        if ret != 0 {
            return Err(RtPriorityError::from_os_error("pthread_setschedparam", ret));
        }
        Ok(())
    }
}
