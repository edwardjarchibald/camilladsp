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

//! Feed-forward, log-domain, soft-knee peak compressor.
//!
//! Implements the Giannoulis / Massberg / Reiss (2012) "Digital Dynamic Range
//! Compressor Design — A Tutorial and Analysis" feed-forward topology, as used
//! by the Elephant Ears Clarity multiband compressor (`clarity-mbc.js`
//! `ClarityCompressor`, a port of the Clarity Challenge `compressor_qmul.py`).
//!
//! It differs from the built-in [`Compressor`](super::compressor) in three ways
//! that make it a distinct component rather than a configuration of the
//! existing one:
//!
//! - **Smoothing domain.** The built-in compressor smooths the *detected level*
//!   and then applies a hard-knee curve. This one applies the (soft-knee) static
//!   curve to the *instantaneous* level first, then branch-smooths the resulting
//!   **gain reduction** `yl = xg - yg`.
//! - **Soft knee.** A quadratic knee of width `knee_width` dB around the
//!   threshold (hard knee when the width is 0).
//! - **Per-channel detection.** Each processed channel has its own feed-forward
//!   detector and envelope state — there is no summed sidechain. This matches
//!   Clarity's per-ear independent processing (one instance per band compresses
//!   that band's left and right channels independently).
//!
//! Per sample, for a channel with envelope state `env` (dB of smoothed gain
//! reduction, >= 0):
//!
//! ```text
//! xi    = if x == 0 { 1e-12 } else { x }
//! xg    = max(20*log10(|xi|), -120)
//! yg    = static curve (hard/soft knee) applied to xg
//! yl    = xg - yg                                  // instantaneous reduction
//! alpha = if yl > env { attack } else { release }  // branch on reduction
//! env   = alpha*env + (1 - alpha)*yl
//! out   = xi * 10^((makeup - env)/20)              // makeup inside the comp
//! ```
//!
//! `attack` / `release` are the one-pole coefficients `exp(-1/(fs*t))` with `t`
//! in seconds — identical to the built-in compressor's time-constant math.
//!
//! The static gain computer, envelope, and makeup all run in `f64` regardless of
//! the `PrcFmt` build width, so the `32bit` build reproduces the Float64
//! reference (and the exported golden vectors) exactly.

// Everything in this module deliberately computes in f64. The many
// `<PrcFmt value> as f64` casts are no-ops under the default f64 `PrcFmt` (which
// clippy flags) but load-bearing under the `32bit` feature (PrcFmt = f32).
#![allow(clippy::unnecessary_cast)]

use crate::PrcFmt;
use crate::Res;
use crate::audiochunk::AudioChunk;
use crate::config;
use crate::filters::limiter::Limiter;
use crate::processors::Processor;
use crate::utils::decibels::db_to_linear;

const EPS: f64 = 1e-12;
const MIN_DB: f64 = -120.0;

#[derive(Clone, Debug)]
pub struct FeedForwardCompressor {
    pub name: String,
    pub channels: usize,
    pub process_channels: Vec<usize>,
    /// One-pole coefficient `exp(-1/(fs*attack_seconds))`.
    pub attack: f64,
    /// One-pole coefficient `exp(-1/(fs*release_seconds))`.
    pub release: f64,
    pub threshold: f64,
    pub factor: f64,
    pub knee_width: f64,
    pub makeup_gain: f64,
    pub limiter: Option<Limiter>,
    pub samplerate: usize,
    /// Smoothed gain reduction (dB, >= 0) per processed channel, persisted
    /// across chunks. Also the live per-band gain-reduction readout.
    pub env: Vec<f64>,
}

impl FeedForwardCompressor {
    /// Creates a FeedForwardCompressor from a config struct.
    pub fn from_config(
        name: &str,
        config: config::FeedForwardCompressorParameters,
        samplerate: usize,
        _chunksize: usize,
    ) -> Self {
        let name = name.to_string();
        let channels = config.channels;
        let srate = samplerate as f64;
        let mut process_channels = config.process_channels();
        if process_channels.is_empty() {
            for n in 0..channels {
                process_channels.push(n);
            }
        }
        let attack = (-1.0 / (srate * config.attack as f64)).exp();
        let release = (-1.0 / (srate * config.release as f64)).exp();
        let clip_limit = config.clip_limit.map(db_to_linear);
        let env = vec![0.0; process_channels.len()];

        debug!(
            "Creating feed-forward compressor '{}', channels: {}, process_channels: {:?}, attack: {}, release: {}, threshold: {}, factor: {}, knee_width: {}, makeup_gain: {}, soft_clip: {}, clip_limit: {:?}",
            name,
            channels,
            process_channels,
            attack,
            release,
            config.threshold,
            config.factor,
            config.knee_width(),
            config.makeup_gain(),
            config.soft_clip(),
            clip_limit
        );

        let limiter = config.clip_limit.map(|limit| {
            let limitconf = config::LimiterParameters {
                clip_limit: limit,
                soft_clip: config.soft_clip,
            };
            Limiter::from_config("Limiter", limitconf)
        });

        FeedForwardCompressor {
            name,
            channels,
            process_channels,
            attack,
            release,
            threshold: config.threshold as f64,
            factor: config.factor as f64,
            knee_width: config.knee_width() as f64,
            makeup_gain: config.makeup_gain() as f64,
            limiter,
            samplerate,
            env,
        }
    }

    /// The Giannoulis static gain computer: map input level `xg` (dB) to output
    /// level `yg` (dB) with a hard or quadratic soft knee.
    #[inline]
    fn gain_computer(&self, xg: f64, inv_ratio: f64, knee_coeff: f64) -> f64 {
        let t = self.threshold;
        let w = self.knee_width;
        if w == 0.0 {
            if xg < t { xg } else { t + (xg - t) * inv_ratio }
        } else if 2.0 * (xg - t) < -w {
            xg
        } else if 2.0 * (xg - t).abs() <= w {
            let over = xg - t + w / 2.0;
            xg + knee_coeff * over * over
        } else {
            t + (xg - t) * inv_ratio
        }
    }

    /// Run the per-sample state machine over one channel, advancing its
    /// envelope state. Computed in f64 to match the reference under any build
    /// width (see the module-level note on the casts).
    fn compress_channel(
        &self,
        waveform: &mut [PrcFmt],
        env: &mut f64,
        inv_ratio: f64,
        knee_coeff: f64,
    ) {
        let makeup = self.makeup_gain;
        let mut e = *env;
        for sample in waveform.iter_mut() {
            let s = *sample as f64;
            let xi = if s == 0.0 { EPS } else { s };
            let mut xg = 20.0 * xi.abs().log10();
            if xg < MIN_DB {
                xg = MIN_DB;
            }
            let yg = self.gain_computer(xg, inv_ratio, knee_coeff);
            let yl = xg - yg;
            let alpha = if yl > e { self.attack } else { self.release };
            e = alpha * e + (1.0 - alpha) * yl;
            let out = xi * 10.0_f64.powf((makeup - e) / 20.0);
            *sample = out as PrcFmt;
        }
        *env = e;
    }

    fn apply_limiter(&self, input: &mut [PrcFmt]) {
        if let Some(limiter) = &self.limiter {
            limiter.apply_clip(input);
        }
    }

    /// Read the current per-channel smoothed gain reduction (dB, >= 0) into a
    /// caller-supplied slice. Read-only — never touches DSP state. This is the
    /// hook for live gain-reduction metering (the fast-follow that surfaces it
    /// over the websocket); the offline gain-trace tests read it too.
    pub fn gain_reductions_db(&self, out: &mut [f64]) {
        for (dst, e) in out.iter_mut().zip(self.env.iter()) {
            *dst = *e;
        }
    }
}

impl Processor for FeedForwardCompressor {
    fn name(&self) -> &str {
        &self.name
    }

    /// Apply a FeedForwardCompressor to an AudioChunk, modifying it in-place.
    fn process_chunk(&mut self, input: &mut AudioChunk) -> Res<()> {
        let inv_ratio = 1.0 / self.factor;
        let knee_coeff = if self.knee_width == 0.0 {
            0.0
        } else {
            (inv_ratio - 1.0) / (2.0 * self.knee_width)
        };
        // Take the envelope state out so `&self` methods and `&mut env[idx]` do
        // not alias; put it back afterwards (Vec::default is a no-op alloc).
        let mut env = std::mem::take(&mut self.env);
        for (idx, ch) in self.process_channels.iter().enumerate() {
            self.compress_channel(
                &mut input.waveforms[*ch],
                &mut env[idx],
                inv_ratio,
                knee_coeff,
            );
            self.apply_limiter(&mut input.waveforms[*ch]);
        }
        self.env = env;
        Ok(())
    }

    fn update_parameters(&mut self, config: config::Processor) {
        if let config::Processor::FeedForwardCompressor {
            parameters: config, ..
        } = config
        {
            let channels = config.channels;
            let srate = self.samplerate as f64;
            let mut process_channels = config.process_channels();
            if process_channels.is_empty() {
                for n in 0..channels {
                    process_channels.push(n);
                }
            }
            let attack = (-1.0 / (srate * config.attack as f64)).exp();
            let release = (-1.0 / (srate * config.release as f64)).exp();
            let clip_limit = config.clip_limit.map(db_to_linear);

            let limiter = config.clip_limit.map(|limit| {
                let limitconf = config::LimiterParameters {
                    clip_limit: limit,
                    soft_clip: config.soft_clip,
                };
                Limiter::from_config("Limiter", limitconf)
            });

            // Preserve envelope state where the channel set is unchanged; resize
            // (zero-fill) only if the number of processed channels changed.
            if process_channels.len() != self.env.len() {
                self.env = vec![0.0; process_channels.len()];
            }
            self.channels = channels;
            self.process_channels = process_channels;
            self.attack = attack;
            self.release = release;
            self.threshold = config.threshold as f64;
            self.factor = config.factor as f64;
            self.knee_width = config.knee_width() as f64;
            self.makeup_gain = config.makeup_gain() as f64;
            self.limiter = limiter;

            debug!(
                "Updated feed-forward compressor '{}', process_channels: {:?}, attack: {}, release: {}, threshold: {}, factor: {}, knee_width: {}, makeup_gain: {}, soft_clip: {}, clip_limit: {:?}",
                self.name,
                self.process_channels,
                attack,
                release,
                config.threshold,
                config.factor,
                config.knee_width(),
                config.makeup_gain(),
                config.soft_clip(),
                clip_limit
            );
        } else {
            // This should never happen unless there is a bug somewhere else
            panic!("Invalid config change!");
        }
    }
}

/// Validate the feed-forward compressor config, to give a helpful message
/// instead of a panic.
pub fn validate_feed_forward_compressor(
    config: &config::FeedForwardCompressorParameters,
) -> Res<()> {
    let channels = config.channels;
    if config.attack <= 0.0 {
        return Err(config::ConfigError::new("Attack value must be larger than zero.").into());
    }
    if config.release <= 0.0 {
        return Err(config::ConfigError::new("Release value must be larger than zero.").into());
    }
    if config.factor <= 0.0 {
        return Err(config::ConfigError::new("Ratio (factor) must be larger than zero.").into());
    }
    if config.knee_width() < 0.0 {
        return Err(config::ConfigError::new("Knee width cannot be negative.").into());
    }
    for ch in config.process_channels().iter() {
        if *ch >= channels {
            let msg = format!(
                "Invalid channel to process: {}, max is: {}.",
                *ch,
                channels - 1
            );
            return Err(config::ConfigError::new(&msg).into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: usize = 48000;

    fn params(
        threshold: PrcFmt,
        factor: PrcFmt,
        makeup: PrcFmt,
        knee: PrcFmt,
    ) -> config::FeedForwardCompressorParameters {
        config::FeedForwardCompressorParameters {
            channels: 1,
            process_channels: None,
            attack: 0.015,
            release: 0.1,
            threshold,
            factor,
            makeup_gain: Some(makeup),
            knee_width: Some(knee),
            soft_clip: None,
            clip_limit: None,
        }
    }

    fn build(p: config::FeedForwardCompressorParameters) -> FeedForwardCompressor {
        FeedForwardCompressor::from_config("test", p, FS, 1024)
    }

    fn run(comp: &mut FeedForwardCompressor, signal: &[PrcFmt]) -> Vec<PrcFmt> {
        let mut chunk =
            AudioChunk::new(vec![signal.to_vec()], 0.0, 0.0, signal.len(), signal.len());
        comp.process_chunk(&mut chunk).unwrap();
        chunk.waveforms[0].clone()
    }

    #[test]
    fn hard_knee_static_curve() {
        // Threshold -20 dB, ratio 4, no knee, no makeup.
        let comp = build(params(-20.0, 4.0, 0.0, 0.0));
        let inv_ratio = 1.0 / comp.factor;
        // Below threshold: yg == xg (no reduction).
        assert!((comp.gain_computer(-40.0, inv_ratio, 0.0) - (-40.0)).abs() < 1e-12);
        // At threshold: yg == threshold.
        assert!((comp.gain_computer(-20.0, inv_ratio, 0.0) - (-20.0)).abs() < 1e-12);
        // 20 dB over: yg = T + (xg-T)/ratio = -20 + 20/4 = -15.
        assert!((comp.gain_computer(0.0, inv_ratio, 0.0) - (-15.0)).abs() < 1e-12);
    }

    #[test]
    fn soft_knee_is_continuous_at_knee_edges() {
        // Soft knee of 10 dB around threshold -20 dB, ratio 4.
        let comp = build(params(-20.0, 4.0, 0.0, 10.0));
        let inv_ratio = 1.0 / comp.factor;
        let knee_coeff = (inv_ratio - 1.0) / (2.0 * comp.knee_width);
        let w = comp.knee_width;
        let t = comp.threshold;
        // Just below the lower knee edge (xg = T - W/2): no reduction, yg == xg.
        let lower = t - w / 2.0 - 1e-6;
        assert!((comp.gain_computer(lower, inv_ratio, knee_coeff) - lower).abs() < 1e-4);
        // At the upper knee edge (xg = T + W/2) the knee meets the ratio line.
        let upper = t + w / 2.0;
        let hard = t + (upper - t) * inv_ratio;
        assert!((comp.gain_computer(upper, inv_ratio, knee_coeff) - hard).abs() < 1e-4);
    }

    #[test]
    fn soft_passage_receives_full_makeup() {
        // Acceptance criterion 4: content below threshold, once the envelope
        // settles, gets ~ the full makeup gain (env -> 0 => gain -> 10^(mk/20)).
        let makeup = 12.0;
        let mut comp = build(params(-20.0, 4.0, makeup, 0.0));
        // Constant low-level tone well below threshold.
        let level = db_to_linear(-50.0);
        let signal: Vec<PrcFmt> = (0..48000).map(|_| level).collect();
        let out = run(&mut comp, &signal);
        let expected = level * db_to_linear(makeup);
        let tail = &out[out.len() - 100..];
        for v in tail {
            assert!(
                (*v - expected).abs() / expected < 1e-3,
                "soft passage should get full makeup: {v} vs {expected}"
            );
        }
        // Envelope reduction should have settled to ~0 dB.
        assert!(
            comp.env[0].abs() < 1e-3,
            "env should settle to 0, got {}",
            comp.env[0]
        );
    }

    #[test]
    fn loud_passage_is_compressed() {
        // Content above threshold ends with net gain below the raw makeup.
        let makeup = 12.0;
        let mut comp = build(params(-20.0, 4.0, makeup, 0.0));
        let level = db_to_linear(0.0); // 0 dBFS, 20 dB over threshold
        let signal: Vec<PrcFmt> = (0..48000).map(|_| level).collect();
        run(&mut comp, &signal);
        // Steady-state reduction should approach xg - yg = 0 - (-15) = 15 dB.
        assert!(
            (comp.env[0] - 15.0).abs() < 0.5,
            "loud steady-state reduction should be ~15 dB, got {}",
            comp.env[0]
        );
    }

    #[test]
    fn block_boundary_matches_whole_signal() {
        // Envelope state must persist across chunks.
        let signal: Vec<PrcFmt> = (0..2048)
            .map(|i| (i as PrcFmt * 0.05).sin() * 0.4)
            .collect();

        let mut whole = build(params(-30.0, 3.0, 6.0, 6.0));
        let whole_out = run(&mut whole, &signal);

        let mut blocked = build(params(-30.0, 3.0, 6.0, 6.0));
        let mut blocked_out = Vec::new();
        for chunk in signal.chunks(128) {
            blocked_out.extend(run(&mut blocked, chunk));
        }

        for (a, b) in whole_out.iter().zip(blocked_out.iter()) {
            assert!((a - b).abs() < 1e-12, "block mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn zero_input_stays_finite() {
        // eps substitution: a zero sample yields eps * gain, never NaN/Inf.
        let mut comp = build(params(-20.0, 4.0, 6.0, 0.0));
        let signal = vec![0.0 as PrcFmt; 256];
        let out = run(&mut comp, &signal);
        for v in out {
            assert!(v.is_finite(), "output must be finite, got {v}");
        }
    }

    #[test]
    fn validate_rejects_bad_configs() {
        assert!(validate_feed_forward_compressor(&params(-20.0, 4.0, 0.0, 0.0)).is_ok());

        let mut bad = params(-20.0, 4.0, 0.0, 0.0);
        bad.attack = 0.0;
        assert!(validate_feed_forward_compressor(&bad).is_err());

        let mut bad = params(-20.0, 0.0, 0.0, 0.0);
        bad.factor = 0.0;
        assert!(validate_feed_forward_compressor(&bad).is_err());

        let mut bad = params(-20.0, 4.0, 0.0, -1.0);
        bad.knee_width = Some(-1.0);
        assert!(validate_feed_forward_compressor(&bad).is_err());

        let mut bad = params(-20.0, 4.0, 0.0, 0.0);
        bad.process_channels = Some(vec![5]);
        assert!(validate_feed_forward_compressor(&bad).is_err());
    }
}
