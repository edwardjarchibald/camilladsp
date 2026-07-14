// Generate JS -> Rust golden vectors for the Clarity native components.
//
// Runs the Elephant Ears source-of-truth `clarity-mbc.js` (ClarityCrossover and
// ClarityCompressor) over deterministic inputs and writes the inputs and
// reference outputs to `golden.json`, next to this script. The Rust tests
// (`filters::crossover`, `processors::feed_forward_compressor`) read that JSON
// and assert the native port reproduces it.
//
// Usage:
//   node generate_golden.js [path/to/clarity-mbc.js]
//
// The path defaults to a sibling elephant_ears checkout; pass it explicitly if
// your layout differs. Only the committed golden.json is needed to run the Rust
// tests — this script documents how it was produced and how to regenerate it.

'use strict';

const fs = require('fs');
const path = require('path');

const DEFAULT_MBC = path.resolve(
    __dirname,
    '../../../elephant_ears_V1/elephant_ears_shared/js/lib/computation/clarity-mbc.js'
);
const mbcPath = process.argv[2] || DEFAULT_MBC;
if (!fs.existsSync(mbcPath)) {
    console.error(`clarity-mbc.js not found at ${mbcPath}`);
    console.error('Pass the path as the first argument.');
    process.exit(1);
}
const { ClarityCrossover, ClarityCompressor } = require(mbcPath);

const SAMPLE_RATE = 48000;
const N = 4096;

/** Deterministic broadband input: incommensurate tones across the band. */
function crossoverInput() {
    const x = new Array(N);
    for (let i = 0; i < N; i++) {
        const t = i / SAMPLE_RATE;
        x[i] =
            0.3 *
            (Math.sin(2 * Math.PI * 100 * t) +
                0.5 * Math.sin(2 * Math.PI * 1000 * t) +
                0.3 * Math.sin(2 * Math.PI * 8000 * t) +
                0.2 * Math.sin(2 * Math.PI * 3000 * t));
    }
    return x;
}

/** 1 kHz tone with a triangular amplitude envelope from -50 dB to 0 dB and
 * back, so attack, release, and the knee are all exercised. */
function compressorInput() {
    const x = new Array(N);
    for (let i = 0; i < N; i++) {
        const t = i / SAMPLE_RATE;
        const frac = i / (N - 1);
        const tri = 1 - Math.abs(2 * frac - 1); // 0 -> 1 -> 0
        const db = -50 + 50 * tri; // -50 dB .. 0 dB .. -50 dB
        const amp = Math.pow(10, db / 20);
        x[i] = amp * Math.sin(2 * Math.PI * 1000 * t);
    }
    return x;
}

// --- Crossover golden vectors (9-band Clarity layout) ---
const XOVER_FREQ = [62.5, 125, 353.55, 707.11, 1414.21, 2828.43, 5656.85, 11314];
const xoverIn = crossoverInput();
const crossover = new ClarityCrossover(XOVER_FREQ, SAMPLE_RATE);
const bands = [];
for (let b = 0; b < crossover.numBands; b++) {
    bands.push(Array.from(crossover.xoverComponent(xoverIn, b)));
}

// --- Compressor golden vectors (several parameter cases) ---
const compIn = compressorInput();
const cases = [
    { threshold: -20, ratio: 4, attack: 15, release: 100, makeupGain: 6, kneeWidth: 0 },
    { threshold: -30, ratio: 3, attack: 5, release: 50, makeupGain: 12, kneeWidth: 10 },
    { threshold: -40, ratio: 8, attack: 1, release: 200, makeupGain: 3, kneeWidth: 6 },
    { threshold: -10, ratio: 2, attack: 20, release: 80, makeupGain: 0, kneeWidth: 0 }
].map((p) => {
    const comp = new ClarityCompressor({ ...p, sampleRate: SAMPLE_RATE });
    return { ...p, output: Array.from(comp.process(compIn)) };
});

const golden = {
    sampleRate: SAMPLE_RATE,
    crossover: { freq: XOVER_FREQ, input: xoverIn, bands },
    compressor: { input: compIn, cases }
};

const outPath = path.resolve(__dirname, 'golden.json');
fs.writeFileSync(outPath, JSON.stringify(golden));
console.log(
    `Wrote ${outPath}: ${bands.length} crossover bands, ${cases.length} compressor cases, ${N} samples each.`
);
