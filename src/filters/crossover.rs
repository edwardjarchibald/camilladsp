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

//! Linkwitz–Riley order-4 crossover band, with D'Appolito phase correction.
//!
//! One `Crossover` instance renders exactly **one band** of an N-band
//! Linkwitz–Riley crossover for a single channel, in place. It is the native
//! counterpart of `ClarityCrossover.xover_component` in the Elephant Ears
//! Clarity multiband compressor (`clarity-mbc.js`), itself a port of the
//! Clarity Challenge `crossover.py` (FILTER_ORDER 4).
//!
//! # Topology
//!
//! Each crossover edge is realized as an order-2 Butterworth section
//! **squared** (a single degree-4 IIR, not a cascade of Q-varied biquads), so
//! it matches the reference bit-for-bit rather than the biquad-cascade LR of
//! `BiquadCombo::LinkwitzRileyLowpass/Highpass`:
//!
//! - Butterworth-2 closed form: `K = tan(pi*wn/2)`, `D = 1 + sqrt(2)*K + K^2`.
//! - LR4 = Butterworth-2 squared: convolve the length-3 numerator/denominator
//!   with themselves → length-5 (degree-4) coefficients.
//!
//! For >= 3 bands the bands do not sum flat without phase correction. The
//! reference applies a D'Appolito all-pass "phi" per crossover. The literal
//! `make_all_pass(b_low, a, b_high, a)` is `((b_low + b_high)*a, a^2)`, a
//! degree-8 filter with four exactly-canceling pole/zero pairs; in floating
//! point the cancellation leaks quadruple poles of radius ~0.99 that blow up
//! for crossovers below ~300 Hz (the 62.5/125 Hz edges in the 9-band layout).
//! We apply the exact cancellation and realize the identical transfer function
//! as the stable degree-4 `((b_low + b_high), a_low)` — matching the note in
//! `ClarityCrossover`.
//!
//! # Band cascade (`xover_component` routing)
//!
//! With `n = freq.len()` crossovers (`numBands = n + 1`), band `band` is:
//! the lowpass at `freq[band]` (unless it is the top band), then the highpasses
//! of every crossover below it, then the phi corrections of the crossovers
//! above it (only when there are >= 3 bands).
//!
//! # Numerics
//!
//! The recursion coefficients and per-section state are held in `f64`
//! regardless of the `PrcFmt` build width. This is a deliberate deviation from
//! the surrounding filters (which run in `PrcFmt`): the reference is Float64,
//! and the degree-4 sections at the low crossovers have poles close enough to
//! the unit circle that `f32` state would visibly diverge from the golden
//! vectors under the `32bit` feature.

use std::f64::consts::{PI, SQRT_2};

use crate::PrcFmt;
use crate::Res;
use crate::config;
use crate::filters::Filter;

/// One direct-form-II-transposed IIR section of degree 4 (length-5 coefficient
/// arrays), with `scipy.signal.lfilter` semantics: coefficients normalized by
/// `a[0]`, zero initial state, and persistent state across calls. Held in `f64`
/// so streaming any block size reproduces the whole-signal reference exactly.
#[derive(Clone, Debug)]
struct Df2tSection {
    bn: [f64; 5],
    an: [f64; 5],
    z: [f64; 4],
}

impl Df2tSection {
    fn new(b: [f64; 5], a: [f64; 5]) -> Self {
        let a0 = a[0];
        let mut bn = [0.0; 5];
        let mut an = [0.0; 5];
        for i in 0..5 {
            bn[i] = b[i] / a0;
            an[i] = a[i] / a0;
        }
        Df2tSection {
            bn,
            an,
            z: [0.0; 4],
        }
    }

    /// Filter `buf` in place, advancing the persistent state.
    #[inline]
    fn process(&mut self, buf: &mut [PrcFmt]) {
        for sample in buf.iter_mut() {
            // `as f64` is a no-op under the default f64 PrcFmt but load-bearing
            // under the `32bit` feature (PrcFmt = f32); the recursion is f64.
            #[allow(clippy::unnecessary_cast)]
            let xi = *sample as f64;
            let yi = self.bn[0] * xi + self.z[0];
            self.z[0] = self.bn[1] * xi + self.z[1] - self.an[1] * yi;
            self.z[1] = self.bn[2] * xi + self.z[2] - self.an[2] * yi;
            self.z[2] = self.bn[3] * xi + self.z[3] - self.an[3] * yi;
            self.z[3] = self.bn[4] * xi - self.an[4] * yi;
            *sample = yi as PrcFmt;
        }
    }
}

#[derive(Clone, Copy)]
enum BandType {
    Low,
    High,
}

/// Order-2 Butterworth biquad (length-3 coefficients), matching
/// `scipy.signal.butter(2, wn, btype)`.
fn butterworth2(wn: f64, btype: BandType) -> ([f64; 3], [f64; 3]) {
    let k = (PI * wn / 2.0).tan();
    let k2 = k * k;
    let d = 1.0 + SQRT_2 * k + k2;
    let a = [1.0, 2.0 * (k2 - 1.0) / d, (1.0 - SQRT_2 * k + k2) / d];
    let b = match btype {
        BandType::Low => [k2 / d, 2.0 * k2 / d, k2 / d],
        BandType::High => [1.0 / d, -2.0 / d, 1.0 / d],
    };
    (b, a)
}

/// Convolve two degree-2 (length-3) polynomials into a degree-4 (length-5) one.
fn conv3(p: [f64; 3], q: [f64; 3]) -> [f64; 5] {
    let mut out = [0.0; 5];
    for i in 0..3 {
        for j in 0..3 {
            out[i + j] += p[i] * q[j];
        }
    }
    out
}

/// LR4 = Butterworth order-2 squared. Length-5 (degree-4) coefficients.
fn linkwitz_riley4(wn: f64, btype: BandType) -> ([f64; 5], [f64; 5]) {
    let (b, a) = butterworth2(wn, btype);
    (conv3(b, b), conv3(a, a))
}

#[derive(Clone, Debug)]
pub struct Crossover {
    pub name: String,
    samplerate: usize,
    sections: Vec<Df2tSection>,
}

impl Crossover {
    /// Creates a Crossover from a config struct.
    pub fn from_config(
        name: &str,
        samplerate: usize,
        parameters: config::CrossoverParameters,
    ) -> Self {
        let name = name.to_string();
        let sections = Crossover::build_sections(samplerate, &parameters);
        debug!(
            "Creating crossover '{}', freq: {:?}, band: {}, cascade length: {}",
            name,
            parameters.freq,
            parameters.band,
            sections.len()
        );
        Crossover {
            name,
            samplerate,
            sections,
        }
    }

    /// Build the ordered degree-4 section cascade for the configured band —
    /// the exact `xover_component` routing (lowpass, then lower highpasses,
    /// then upper phi corrections).
    fn build_sections(samplerate: usize, params: &config::CrossoverParameters) -> Vec<Df2tSection> {
        let nyquist = samplerate as f64 / 2.0;
        let n = params.freq.len();

        // Per crossover edge: LR4 lowpass, LR4 highpass, phi all-pass.
        let mut low = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut phi = Vec::with_capacity(n);
        for freq in &params.freq {
            // See the note in `Df2tSection::process`: the cast is load-bearing
            // under the `32bit` feature (PrcFmt = f32).
            #[allow(clippy::unnecessary_cast)]
            let wn = (*freq as f64) / nyquist;
            let (low_b, low_a) = linkwitz_riley4(wn, BandType::Low);
            let (high_b, high_a) = linkwitz_riley4(wn, BandType::High);
            // Stable phi: ((b_low + b_high), a_low), degree 4.
            let mut phi_b = [0.0; 5];
            for i in 0..5 {
                phi_b[i] = low_b[i] + high_b[i];
            }
            low.push((low_b, low_a));
            high.push((high_b, high_a));
            phi.push((phi_b, low_a));
        }

        let band = params.band;
        let mut cascade = Vec::new();

        // Lowpass component (every band except the top one).
        if band < n {
            let (b, a) = low[band];
            cascade.push(Df2tSection::new(b, a));
        }
        // Highpass components (all crossovers below this band).
        for section in high.iter().take(band) {
            let (b, a) = *section;
            cascade.push(Df2tSection::new(b, a));
        }
        // Phi (phase-correction) components — only for >= 3 bands.
        if n + 1 > 2 {
            if band + 2 == n {
                let (b, a) = phi[band + 1];
                cascade.push(Df2tSection::new(b, a));
            } else if band + 2 < n {
                for section in phi.iter().take(n).skip(band + 1) {
                    let (b, a) = *section;
                    cascade.push(Df2tSection::new(b, a));
                }
            }
        }

        cascade
    }
}

impl Filter for Crossover {
    fn name(&self) -> &str {
        &self.name
    }

    fn process_waveform(&mut self, waveform: &mut [PrcFmt]) -> Res<()> {
        for section in self.sections.iter_mut() {
            section.process(waveform);
        }
        Ok(())
    }

    fn update_parameters(&mut self, config: config::Filter) {
        if let config::Filter::Crossover {
            parameters: config, ..
        } = config
        {
            let name = self.name.clone();
            *self = Crossover::from_config(&name, self.samplerate, config);
        } else {
            // This should never happen unless there is a bug somewhere else
            panic!("Invalid config change!");
        }
    }
}

/// Validate a Crossover config, to give a helpful message instead of a panic.
pub fn validate_config(samplerate: usize, conf: &config::CrossoverParameters) -> Res<()> {
    let maxfreq = samplerate as PrcFmt / 2.0;
    if conf.freq.is_empty() {
        return Err(
            config::ConfigError::new("Crossover needs at least one crossover frequency").into(),
        );
    }
    for freq in &conf.freq {
        if *freq <= 0.0 {
            return Err(config::ConfigError::new("Crossover frequencies must be > 0").into());
        } else if *freq >= maxfreq {
            return Err(
                config::ConfigError::new("Crossover frequencies must be < samplerate/2").into(),
            );
        }
    }
    for pair in conf.freq.windows(2) {
        if pair[1] <= pair[0] {
            return Err(config::ConfigError::new(
                "Crossover frequencies must be strictly ascending",
            )
            .into());
        }
    }
    let num_bands = conf.freq.len() + 1;
    if conf.band >= num_bands {
        let msg = format!(
            "Crossover band {} is out of range, must be 0..={}",
            conf.band,
            num_bands - 1
        );
        return Err(config::ConfigError::new(&msg).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::Filter;

    /// The 9-band Clarity crossover layout (8 edges), including the low
    /// 62.5/125 Hz crossovers that stress the phi stability.
    const CLARITY_9BAND: [PrcFmt; 8] = [
        62.5, 125.0, 353.55, 707.11, 1414.21, 2828.43, 5656.85, 11314.0,
    ];
    const FS: usize = 48000;

    /// Complex value as (re, im).
    type Cplx = (f64, f64);

    fn c_mul(a: Cplx, b: Cplx) -> Cplx {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }
    fn c_add(a: Cplx, b: Cplx) -> Cplx {
        (a.0 + b.0, a.1 + b.1)
    }
    fn c_div(a: Cplx, b: Cplx) -> Cplx {
        let d = b.0 * b.0 + b.1 * b.1;
        ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
    }

    /// Evaluate the polynomial sum(c[k] * z^-k) at z = e^{j*w}.
    fn poly_at(coeffs: &[f64], w: f64) -> Cplx {
        let mut acc = (0.0, 0.0);
        for (k, c) in coeffs.iter().enumerate() {
            let angle = -(k as f64) * w;
            acc = c_add(acc, (c * angle.cos(), c * angle.sin()));
        }
        acc
    }

    /// Complex frequency response of one Df2tSection at angular frequency w.
    fn section_response(section: &Df2tSection, w: f64) -> Cplx {
        c_div(poly_at(&section.bn, w), poly_at(&section.an, w))
    }

    /// Complex frequency response of a band's whole cascade at w.
    fn band_response(sections: &[Df2tSection], w: f64) -> Cplx {
        let mut acc = (1.0, 0.0);
        for s in sections {
            acc = c_mul(acc, section_response(s, w));
        }
        acc
    }

    fn build_band(freqs: &[PrcFmt], band: usize) -> Crossover {
        let params = config::CrossoverParameters {
            freq: freqs.to_vec(),
            band,
        };
        Crossover::from_config("test", FS, params)
    }

    #[test]
    fn flat_reconstruction_9band() {
        let num_bands = CLARITY_9BAND.len() + 1;
        let crossovers: Vec<Crossover> = (0..num_bands)
            .map(|b| build_band(&CLARITY_9BAND, b))
            .collect();

        let nyquist = FS as f64 / 2.0;
        // Log-spaced grid across 50 Hz .. 20 kHz.
        let n_points = 400;
        let f_lo = 50.0_f64.ln();
        let f_hi = 20000.0_f64.ln();
        let mut max_dev_db = 0.0_f64;
        for i in 0..=n_points {
            let f = (f_lo + (f_hi - f_lo) * (i as f64) / (n_points as f64)).exp();
            let w = PI * f / nyquist;
            let mut sum = (0.0, 0.0);
            for xover in &crossovers {
                sum = c_add(sum, band_response(&xover.sections, w));
            }
            let mag = (sum.0 * sum.0 + sum.1 * sum.1).sqrt();
            let dev_db = 20.0 * mag.log10();
            if dev_db.abs() > max_dev_db {
                max_dev_db = dev_db.abs();
            }
        }
        assert!(
            max_dev_db < 0.1,
            "9-band reconstruction deviates {max_dev_db:.4} dB (limit 0.1 dB)"
        );
    }

    #[test]
    fn single_crossover_sums_flat() {
        // Two-band split: no phi correction, but low + high must still sum flat.
        let low = build_band(&[1000.0], 0);
        let high = build_band(&[1000.0], 1);
        let nyquist = FS as f64 / 2.0;
        for &f in &[50.0, 200.0, 1000.0, 5000.0, 18000.0] {
            let w = PI * f / nyquist;
            let sum = c_add(
                band_response(&low.sections, w),
                band_response(&high.sections, w),
            );
            let mag = (sum.0 * sum.0 + sum.1 * sum.1).sqrt();
            let dev_db = 20.0 * mag.log10();
            assert!(dev_db.abs() < 0.1, "2-band at {f} Hz: {dev_db:.4} dB");
        }
    }

    #[test]
    fn top_band_is_all_highpass() {
        // Band N (the top) has no lowpass and no phi; it is n highpasses.
        let n = CLARITY_9BAND.len();
        let top = build_band(&CLARITY_9BAND, n);
        assert_eq!(top.sections.len(), n);
        // DC (w -> 0) is fully rejected by highpasses.
        let dc = band_response(&top.sections, 1e-6);
        let mag = (dc.0 * dc.0 + dc.1 * dc.1).sqrt();
        assert!(mag < 1e-6, "top band should reject DC, got {mag}");
    }

    #[test]
    fn band_zero_is_lowpass_only() {
        // Band 0 is the single lowpass plus the phi corrections of the
        // crossovers above it — but no highpasses; it must pass DC at unity.
        let xover = build_band(&CLARITY_9BAND, 0);
        let dc = band_response(&xover.sections, 1e-6);
        let mag = (dc.0 * dc.0 + dc.1 * dc.1).sqrt();
        assert!((mag - 1.0).abs() < 1e-6, "band 0 should pass DC, got {mag}");
    }

    #[test]
    fn stateful_stream_matches_whole_signal() {
        // Filtering in one call must equal filtering split across blocks,
        // proving the persistent DF2T state is correct.
        let signal: Vec<PrcFmt> = (0..512)
            .map(|i| (i as PrcFmt * 0.13).sin() * 0.5 + (i as PrcFmt * 0.0007).sin())
            .collect();

        let mut whole = build_band(&CLARITY_9BAND, 3);
        let mut whole_sig = signal.clone();
        whole.process_waveform(&mut whole_sig).unwrap();

        let mut blocked = build_band(&CLARITY_9BAND, 3);
        let mut blocked_sig = signal.clone();
        for chunk in blocked_sig.chunks_mut(64) {
            blocked.process_waveform(chunk).unwrap();
        }

        for (a, b) in whole_sig.iter().zip(blocked_sig.iter()) {
            assert!((a - b).abs() < 1e-12, "block boundary mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn validate_rejects_bad_configs() {
        let ok = config::CrossoverParameters {
            freq: vec![1000.0, 2000.0],
            band: 1,
        };
        assert!(validate_config(FS, &ok).is_ok());

        let empty = config::CrossoverParameters {
            freq: vec![],
            band: 0,
        };
        assert!(validate_config(FS, &empty).is_err());

        let not_ascending = config::CrossoverParameters {
            freq: vec![2000.0, 1000.0],
            band: 0,
        };
        assert!(validate_config(FS, &not_ascending).is_err());

        let above_nyquist = config::CrossoverParameters {
            freq: vec![30000.0],
            band: 0,
        };
        assert!(validate_config(FS, &above_nyquist).is_err());

        let band_oob = config::CrossoverParameters {
            freq: vec![1000.0, 2000.0],
            band: 3,
        };
        assert!(validate_config(FS, &band_oob).is_err());
    }
}
