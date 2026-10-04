// Forward pass of the residual policy/value network
// (src/connectfive/network.py: PolicyValueNetwork) in plain TypeScript.
// Activations are channels-last (15 x 15 x C) Float32Arrays. Weights come from
// scripts/export_web_model.py, which writes model.json plus weights.bin.

import { GEMM_WASM_BASE64 } from './gemm-wasm';
import { PatternBoard, SIZE } from './patterns';

const CELLS = SIZE * SIZE;
const EPSILON = 1e-6;

export type ModelManifest = {
  format: 'connectfive-web-model';
  version: 1;
  name: string;
  config: { residual_blocks: number; channels: number; value_hidden: number; input_planes: number };
  tensors: { name: string; shape: number[]; offset: number; length: number }[];
};

type Tensors = Record<string, Float32Array>;

export class Network {
  private readonly blocks: number;
  private readonly channels: number;
  private readonly planes: number;
  private readonly hidden: number;
  private readonly a: Float32Array;
  private readonly b: Float32Array;
  private readonly c: Float32Array;
  private readonly input: Float32Array;
  private readonly valueFeatures = new Float32Array(CELLS * 4);
  private readonly valueHidden: Float32Array;
  private readonly cache = new Map<number, { logits: Float32Array; value: number }>();
  calls = 0;
  /** SIMD WebAssembly convolutions when available, else plain TypeScript. */
  private readonly simd: SimdConv | null;
  readonly backend: 'wasm-simd' | 'js';

  constructor(readonly manifest: ModelManifest, private readonly t: Tensors, allowSimd = true) {
    this.blocks = manifest.config.residual_blocks;
    this.channels = manifest.config.channels;
    this.planes = manifest.config.input_planes;
    this.hidden = manifest.config.value_hidden;
    this.a = new Float32Array(CELLS * this.channels);
    this.b = new Float32Array(CELLS * this.channels);
    this.c = new Float32Array(CELLS * this.channels);
    this.input = new Float32Array(CELLS * this.planes);
    this.valueHidden = new Float32Array(this.hidden);
    this.simd = allowSimd ? SimdConv.create() : null;
    if (this.simd) {
      this.simd.prepare(t['Conv_0/kernel'], this.planes, this.channels);
      for (let block = 0; block < this.blocks; block += 1) {
        this.simd.prepare(t[`ResidualBlock_${block}/Conv_0/kernel`], this.channels, this.channels);
        this.simd.prepare(t[`ResidualBlock_${block}/Conv_1/kernel`], this.channels, this.channels);
      }
      this.simd.reserve(Math.max(this.planes, this.channels), this.channels);
    }
    this.backend = this.simd ? 'wasm-simd' : 'js';
  }

  private conv(input: Float32Array, cin: number, kernel: Float32Array, cout: number, output: Float32Array) {
    if (this.simd) this.simd.conv3x3(input, cin, kernel, cout, output);
    else conv3x3(input, cin, kernel, cout, output);
  }

  static fromBuffers(manifest: ModelManifest, weights: ArrayBuffer, allowSimd = true) {
    const data = new Float32Array(weights);
    const tensors: Tensors = {};
    for (const tensor of manifest.tensors) {
      tensors[tensor.name] = data.subarray(tensor.offset, tensor.offset + tensor.length);
    }
    return new Network(manifest, tensors, allowSimd);
  }

  /** Raw policy logits (225, action order) and value for the side to move. */
  evaluate(board: PatternBoard) {
    const key = board.hash;
    const cached = this.cache.get(key);
    if (cached) return cached;
    this.calls += 1;
    this.encode(board);
    const result = this.forward();
    if (this.cache.size > 200_000) this.cache.clear();
    this.cache.set(key, result);
    return result;
  }

  /** A copy of the network input for `board`, keyed by its hash (for worker pools). */
  encodeRequest(board: PatternBoard) {
    this.encode(board);
    return { hash: board.hash, features: this.input.slice() };
  }

  /** Forward pass on explicit features (15 x 15 x planes), used by tests and worker pools. */
  evaluateFeatures(features: Float32Array) {
    this.input.set(features);
    return this.forward();
  }

  private encode(board: PatternBoard) {
    const array = board.toArray();
    const player = board.player;
    const planes = this.planes;
    const blackToMove = player === 0 ? 1 : 0;
    for (let cell = 0; cell < CELLS; cell += 1) {
      const base = cell * planes;
      this.input[base] = array[cell] === player ? 1 : 0;
      this.input[base + 1] = array[cell] === 1 - player ? 1 : 0;
      this.input[base + 2] = blackToMove;
      if (planes === 4) this.input[base + 3] = 1;
    }
  }

  private forward() {
    const t = this.t;
    const C = this.channels;
    this.conv(this.input, this.planes, t['Conv_0/kernel'], C, this.a);
    layerNorm(this.a, C, t['LayerNorm_0/scale'], t['LayerNorm_0/bias']);
    relu(this.a);
    for (let block = 0; block < this.blocks; block += 1) {
      const p = `ResidualBlock_${block}/`;
      this.conv(this.a, C, t[`${p}Conv_0/kernel`], C, this.b);
      layerNorm(this.b, C, t[`${p}LayerNorm_0/scale`], t[`${p}LayerNorm_0/bias`]);
      relu(this.b);
      this.conv(this.b, C, t[`${p}Conv_1/kernel`], C, this.c);
      layerNorm(this.c, C, t[`${p}LayerNorm_1/scale`], t[`${p}LayerNorm_1/bias`]);
      for (let i = 0; i < this.a.length; i += 1) {
        const sum = this.a[i] + this.c[i];
        this.a[i] = sum > 0 ? sum : 0;
      }
    }

    // Policy head: 1x1 conv to 2 channels, ReLU, dense 2 -> 1 per cell.
    const logits = new Float32Array(CELLS);
    const pk = t['policy_conv/kernel'];
    const pw = t['policy_logits/kernel'];
    const pb = t['policy_logits/bias'][0];
    for (let cell = 0; cell < CELLS; cell += 1) {
      const base = cell * C;
      let h0 = 0;
      let h1 = 0;
      for (let i = 0; i < C; i += 1) {
        const x = this.a[base + i];
        h0 += x * pk[i * 2];
        h1 += x * pk[i * 2 + 1];
      }
      logits[cell] = (h0 > 0 ? h0 : 0) * pw[0] + (h1 > 0 ? h1 : 0) * pw[1] + pb;
    }

    // Value head: 1x1 conv to 4, LayerNorm, ReLU, dense 900 -> hidden -> 1, tanh.
    const vk = t['value_conv/kernel'];
    const vf = this.valueFeatures;
    for (let cell = 0; cell < CELLS; cell += 1) {
      const base = cell * C;
      for (let o = 0; o < 4; o += 1) {
        let sum = 0;
        for (let i = 0; i < C; i += 1) sum += this.a[base + i] * vk[i * 4 + o];
        vf[cell * 4 + o] = sum;
      }
    }
    layerNorm(vf, 4, t['value_norm/scale'], t['value_norm/bias']);
    relu(vf);
    const hk = t['value_hidden/kernel'];
    const hb = t['value_hidden/bias'];
    const hidden = this.valueHidden;
    hidden.set(hb);
    for (let i = 0; i < vf.length; i += 1) {
      const x = vf[i];
      if (x === 0) continue;
      const row = i * this.hidden;
      for (let o = 0; o < this.hidden; o += 1) hidden[o] += x * hk[row + o];
    }
    const ok = t['value_output/kernel'];
    let out = t['value_output/bias'][0];
    for (let o = 0; o < this.hidden; o += 1) out += (hidden[o] > 0 ? hidden[o] : 0) * ok[o];
    return { logits, value: Math.tanh(out) };
  }
}

/**
 * 3x3 convolution as im2col + SIMD matrix multiply in WebAssembly.
 * Kernels are transposed once to (out, 3 * 3 * in) rows inside wasm memory.
 */
class SimdConv {
  private f32: Float32Array;
  private used = 0;
  private readonly kernels = new Map<Float32Array, number>();
  private patches = -1;
  private out = -1;

  private constructor(
    private readonly memory: WebAssembly.Memory,
    private readonly gemm: (p: number, k: number, out: number, rows: number, K: number, cout: number) => void,
  ) {
    this.f32 = new Float32Array(memory.buffer);
  }

  static create(): SimdConv | null {
    try {
      if (typeof WebAssembly === 'undefined') return null;
      const bytes = Uint8Array.from(atob(GEMM_WASM_BASE64), (c) => c.charCodeAt(0));
      if (!WebAssembly.validate(bytes)) return null;
      const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
      const { memory, gemm } = instance.exports as unknown as {
        memory: WebAssembly.Memory;
        gemm: (p: number, k: number, out: number, rows: number, K: number, cout: number) => void;
      };
      return new SimdConv(memory, gemm);
    } catch {
      return null;
    }
  }

  private allocate(floats: number) {
    const start = this.used;
    this.used += Math.ceil(floats / 4) * 4;
    const needed = this.used * 4;
    if (needed > this.memory.buffer.byteLength) {
      this.memory.grow(Math.ceil((needed - this.memory.buffer.byteLength) / 65_536));
      this.f32 = new Float32Array(this.memory.buffer);
    }
    return start;
  }

  prepare(kernel: Float32Array, cin: number, cout: number) {
    const K = 9 * cin;
    const offset = this.allocate(cout * K);
    // Flax layout (ky, kx, in, out) -> rows (out, ky * kx * in).
    for (let tap = 0; tap < 9; tap += 1) {
      for (let i = 0; i < cin; i += 1) {
        for (let o = 0; o < cout; o += 1) this.f32[offset + o * K + tap * cin + i] = kernel[(tap * cin + i) * cout + o];
      }
    }
    this.kernels.set(kernel, offset);
  }

  /** Allocate the im2col and output buffers once all kernels are prepared. */
  reserve(maxCin: number, maxCout: number) {
    this.patches = this.allocate(SIZE * SIZE * 9 * maxCin);
    this.out = this.allocate(SIZE * SIZE * maxCout);
  }

  conv3x3(input: Float32Array, cin: number, kernel: Float32Array, cout: number, output: Float32Array) {
    const K = 9 * cin;
    const f32 = this.f32;
    const patches = this.patches;
    f32.fill(0, patches, patches + SIZE * SIZE * K);
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        const rowBase = patches + (y * SIZE + x) * K;
        for (let ky = 0; ky < 3; ky += 1) {
          const yy = y + ky - 1;
          if (yy < 0 || yy >= SIZE) continue;
          for (let kx = 0; kx < 3; kx += 1) {
            const xx = x + kx - 1;
            if (xx < 0 || xx >= SIZE) continue;
            const source = (yy * SIZE + xx) * cin;
            f32.set(input.subarray(source, source + cin), rowBase + (ky * 3 + kx) * cin);
          }
        }
      }
    }
    this.gemm(patches * 4, (this.kernels.get(kernel) as number) * 4, this.out * 4, SIZE * SIZE, K, cout);
    output.set(f32.subarray(this.out, this.out + SIZE * SIZE * cout));
  }
}

/** SAME-padded 3x3 cross-correlation; kernel layout (3, 3, in, out). */
function conv3x3(input: Float32Array, cin: number, kernel: Float32Array, cout: number, output: Float32Array) {
  output.fill(0);
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const outBase = (y * SIZE + x) * cout;
      for (let ky = 0; ky < 3; ky += 1) {
        const yy = y + ky - 1;
        if (yy < 0 || yy >= SIZE) continue;
        for (let kx = 0; kx < 3; kx += 1) {
          const xx = x + kx - 1;
          if (xx < 0 || xx >= SIZE) continue;
          const inBase = (yy * SIZE + xx) * cin;
          const kBase = (ky * 3 + kx) * cin * cout;
          for (let i = 0; i < cin; i += 1) {
            const v = input[inBase + i];
            if (v === 0) continue;
            const kRow = kBase + i * cout;
            for (let o = 0; o < cout; o += 1) output[outBase + o] += v * kernel[kRow + o];
          }
        }
      }
    }
  }
}

/** Flax LayerNorm over the channel axis (fast variance, epsilon 1e-6). */
function layerNorm(values: Float32Array, channels: number, scale: Float32Array, bias: Float32Array) {
  for (let base = 0; base < values.length; base += channels) {
    let mean = 0;
    let square = 0;
    for (let i = 0; i < channels; i += 1) {
      const v = values[base + i];
      mean += v;
      square += v * v;
    }
    mean /= channels;
    const variance = Math.max(0, square / channels - mean * mean);
    const inv = 1 / Math.sqrt(variance + EPSILON);
    for (let i = 0; i < channels; i += 1) {
      values[base + i] = (values[base + i] - mean) * inv * scale[i] + bias[i];
    }
  }
}

function relu(values: Float32Array) {
  for (let i = 0; i < values.length; i += 1) if (values[i] < 0) values[i] = 0;
}
