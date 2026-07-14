# Elephant Ears fork of CamillaDSP

This is a **rebasing fork** of [`HEnquist/camilladsp`](https://github.com/HEnquist/camilladsp),
maintained for the Elephant Ears project. It adds two purely **additive** native
DSP components used by the Clarity compensation model:

- **`Crossover`** filter — a Linkwitz–Riley order-4 (Butterworth-2 squared)
  crossover band with stabilized D'Appolito "phi" all-pass correction, faithful
  to `elephant_ears_shared/js/lib/computation/clarity-mbc.js`. Enables ≥3-band
  reconstruction that sums flat.
- **`FeedForwardCompressor`** processor — a feed-forward, log-domain, soft-knee
  peak compressor (Giannoulis/Massberg/Reiss 2012 algorithm) with per-band
  makeup applied *inside* the compressor (`c = 10^((makeup − reduction)/20)`).

Neither changes the behavior of any existing component; registration into the
shared `Filter` / `Processor` dispatch enums is the only edit to existing files,
and it is compiler-enforced additive (new variant + new match arm).

## Branch model

| Branch | Role |
|--------|------|
| `master` | Fast-forward mirror of `upstream/master`. **Never commit here.** |
| `elephant-ears` | Carries the additive Elephant Ears commits. This is the release branch. |

Remotes (in a working clone):

- `upstream` → `https://github.com/HEnquist/camilladsp.git`
- `origin` → `https://github.com/edwardjarchibald/camilladsp.git`

## Rebase workflow

Keep the fork current with upstream by rebasing (not merging), so the additive
commits stay a clean, legible set on top of upstream:

```sh
git fetch upstream
git switch master && git merge --ff-only upstream/master   # mirror master
git switch elephant-ears
git rebase upstream/master
# resolve the (few, localized) registration conflicts if any
git push --force-with-lease origin elephant-ears
```

Because every change is additive, conflicts are confined to the handful of
registration insertion points (`config/mod.rs` enums, `filters/mod.rs` /
`processors/mod.rs` module lists + validate arms, `pipeline.rs` construction
arms, `config/utils.rs` diff/validate arms).

## Releases

Tag fork releases as `v<upstream>-ee.<n>` (e.g. `v4.1.3-ee.1`) so both the base
upstream version and the fork revision are legible. The upstream
`.github/workflows/publish.yml` builds the full platform matrix unchanged (no new
dependencies, pure-Rust additive files), so tagging a release produces drop-in
replacement binaries. `ci_test.yml` (fmt/clippy/test) gates the additive code.

## Validation

Same gates as upstream: `cargo fmt --all -- --check`,
`cargo clippy --all-targets --all-features -- -D warnings`, `cargo test`
(including the `32bit` feature). The new components additionally carry
flat-reconstruction, golden-vector (against `clarity-mbc.js`), and
soft-vs-loud gain-trace tests.
