// TypeScript port of src/connectfive/patterns.py: an incremental exact-five
// pattern board. Every empty cell caches, for each player and direction, the
// "line level" a stone placed there would create:
//   6 exact five, 5 open four, 4 four, 3 open three, 2 closed three,
//   1 open two, 0 nothing. Overlines never count as fives.
// Keep this file in lockstep with the Python version; parity fixtures in
// web/tests/engine.test.mjs check that both agree.

export const SIZE = 15;
export const PAD = 5;
export const WIDTH = SIZE + 2 * PAD;
export const EMPTY = 0;
export const BLACK = 1;
export const WHITE = 2;
export const WALL = 3;
const DIRECTIONS = [1, WIDTH, WIDTH + 1, WIDTH - 1];

export const ACTION_TO_INDEX = new Int16Array(SIZE * SIZE);
export const INDEX_TO_ACTION = new Int16Array(WIDTH * WIDTH).fill(-1);
for (let action = 0; action < SIZE * SIZE; action += 1) {
  const index = (Math.floor(action / SIZE) + PAD) * WIDTH + (action % SIZE) + PAD;
  ACTION_TO_INDEX[action] = index;
  INDEX_TO_ACTION[index] = action;
}
const onBoard = (index: number) => index >= 0 && index < WIDTH * WIDTH && INDEX_TO_ACTION[index] >= 0;

const NEIGHBOURS: number[][] = [];
export const LINE_CELLS: number[][][] = [];
for (let index = 0; index < WIDTH * WIDTH; index += 1) {
  NEIGHBOURS.push([]);
  LINE_CELLS.push([]);
  if (INDEX_TO_ACTION[index] < 0) continue;
  for (let dr = -2; dr <= 2; dr += 1) {
    for (let dc = -2; dc <= 2; dc += 1) {
      const other = index + dr * WIDTH + dc;
      if ((dr || dc) && onBoard(other)) NEIGHBOURS[index].push(other);
    }
  }
  for (const d of DIRECTIONS) {
    const line: number[] = [];
    for (let k = -PAD; k <= PAD; k += 1) if (onBoard(index + k * d)) line.push(index + k * d);
    LINE_CELLS[index].push(line);
  }
}

// Search-facing scores, identical to the Python constants.
export const SCORE_FIVE = 10_000_000;
export const SCORE_FORCE = 1_000_000;
const SCORE_FOUR_THREE = 100_000;
const SCORE_DOUBLE_THREE = 50_000;
const ORDER_WEIGHTS = [0, 40, 300, 2_500, 2_000, 0, 0];
const ATTACK_WEIGHTS = [0, 30, 200, 1_500, 800, 0, 0];
const DEFENCE_WEIGHTS = [0, 25, 150, 1_000, 600, 5_000, 0];

// ----- line levels -------------------------------------------------------

function runThroughCenter(window: number[]) {
  let run = 1;
  for (let i = PAD - 1; i >= 0 && window[i] === 1; i -= 1) run += 1;
  for (let i = PAD + 1; i <= 2 * PAD && window[i] === 1; i += 1) run += 1;
  return run;
}

// Memo over 11-cell ternary windows (0 empty, 1 own, 2 blocked); -1 = unknown.
const levelMemo = new Int8Array(3 ** 11).fill(-1);

function windowKey(window: number[]) {
  let key = 0;
  for (const value of window) key = key * 3 + value;
  return key;
}

function level(window: number[]): number {
  const key = windowKey(window);
  const cached = levelMemo[key];
  if (cached >= 0) return cached;
  let result = 0;
  const run = runThroughCenter(window);
  if (run === 5) result = 6;
  else if (run < 5) {
    const empties: number[] = [];
    for (let i = 1; i < 2 * PAD; i += 1) if (window[i] === 0) empties.push(i);
    let winning = 0;
    for (const i of empties) {
      window[i] = 1;
      if (runThroughCenter(window) === 5) winning += 1;
      window[i] = 0;
    }
    if (winning >= 2) result = 5;
    else if (winning === 1) result = 4;
    else {
      let best = 0;
      for (const i of empties) {
        window[i] = 1;
        best = Math.max(best, level(window));
        window[i] = 0;
        if (best === 5) break;
      }
      result = best === 5 ? 3 : best === 4 ? 2 : best === 3 ? 1 : 0;
    }
  }
  levelMemo[key] = result;
  return result;
}

const TRANSLATE = [
  [0, 0, 0, 0],
  [0, 1, 2, 2], // black
  [0, 2, 1, 2], // white
];
// Raw 10-cell neighbourhoods (values 0-3) packed base 4 -> black * 8 + white.
const lineMemo = new Int8Array(4 ** 10).fill(-1);
const scratch: number[] = Array.from({ length: 2 * PAD + 1 }, () => 0);

function lineLevels(rawKey: number, raw: number[]) {
  const cached = lineMemo[rawKey];
  if (cached >= 0) return cached;
  let packed = 0;
  for (const player of [BLACK, WHITE]) {
    const translate = TRANSLATE[player];
    for (let i = 0; i < PAD; i += 1) scratch[i] = translate[raw[i]];
    scratch[PAD] = 1;
    for (let i = PAD; i < 2 * PAD; i += 1) scratch[i + 1] = translate[raw[i]];
    packed = packed * 8 + level(scratch);
  }
  lineMemo[rawKey] = packed;
  return packed;
}

// ----- cell summaries ------------------------------------------------------

export const WIN_FLAG = 1;
export const FORCE_FLAG = 2;
export const FOUR_FLAG = 4;

// 7^4 combinations of four direction levels.
const SUMMARY_ORDER = new Int32Array(7 ** 4);
const SUMMARY_ATTACK = new Int32Array(7 ** 4);
const SUMMARY_DEFENCE = new Int32Array(7 ** 4);
const SUMMARY_FLAGS = new Uint8Array(7 ** 4);
for (let key = 0; key < 7 ** 4; key += 1) {
  const levels = [Math.floor(key / 343) % 7, Math.floor(key / 49) % 7, Math.floor(key / 7) % 7, key % 7];
  const counts: number[] = Array.from({ length: 7 }, () => 0);
  for (const value of levels) counts[value] += 1;
  const win = counts[6] > 0;
  const fours = counts[5] + counts[4];
  const force = !win && (counts[5] > 0 || fours >= 2);
  let order = 0;
  let attack = 0;
  let defence = 0;
  for (const value of levels) {
    order += ORDER_WEIGHTS[value];
    attack += ATTACK_WEIGHTS[value];
    defence += DEFENCE_WEIGHTS[value];
  }
  if (win) order += SCORE_FIVE;
  else if (force) {
    order += SCORE_FORCE;
    defence += 8_000;
  } else if (counts[4] && counts[3]) {
    order += SCORE_FOUR_THREE;
    attack += 20_000;
    defence += 10_000;
  } else if (counts[3] >= 2) {
    order += SCORE_DOUBLE_THREE;
    attack += 20_000;
    defence += 5_000;
  }
  SUMMARY_ORDER[key] = order;
  SUMMARY_ATTACK[key] = attack;
  SUMMARY_DEFENCE[key] = defence;
  SUMMARY_FLAGS[key] = (win ? WIN_FLAG : 0) | (force ? FORCE_FLAG : 0) | (win || fours > 0 ? FOUR_FLAG : 0);
}

// ----- Zobrist hashing (two 32-bit halves combined into a 53-bit key) -----

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}
const random = mulberry32(20260930);
const ZOBRIST_HIGH = [new Uint32Array(WIDTH * WIDTH), new Uint32Array(WIDTH * WIDTH), new Uint32Array(WIDTH * WIDTH)];
const ZOBRIST_LOW = [new Uint32Array(WIDTH * WIDTH), new Uint32Array(WIDTH * WIDTH), new Uint32Array(WIDTH * WIDTH)];
for (const player of [BLACK, WHITE]) {
  for (let index = 0; index < WIDTH * WIDTH; index += 1) {
    ZOBRIST_HIGH[player][index] = random();
    ZOBRIST_LOW[player][index] = random() & 0x1fffff;
  }
}

// ----- board ---------------------------------------------------------------

export class PatternBoard {
  readonly cells = new Int8Array(WIDTH * WIDTH).fill(WALL);
  readonly near = new Int16Array(WIDTH * WIDTH);
  readonly candidates = new Set<number>();
  readonly moves: number[] = [];
  // levels[player][index * 4 + direction]
  readonly levels = [new Uint8Array(0), new Uint8Array(WIDTH * WIDTH * 4), new Uint8Array(WIDTH * WIDTH * 4)];
  readonly order = [new Int32Array(0), new Int32Array(WIDTH * WIDTH), new Int32Array(WIDTH * WIDTH)];
  readonly attack = [new Int32Array(0), new Int32Array(WIDTH * WIDTH), new Int32Array(WIDTH * WIDTH)];
  readonly defence = [new Int32Array(0), new Int32Array(WIDTH * WIDTH), new Int32Array(WIDTH * WIDTH)];
  readonly flags = [new Uint8Array(0), new Uint8Array(WIDTH * WIDTH), new Uint8Array(WIDTH * WIDTH)];
  turn = BLACK;
  private hashHigh = 0;
  private hashLow = 0;
  private readonly raw: number[] = Array.from({ length: 2 * PAD }, () => 0);
  // Undo log: (player * 4 * WIDTH^2 + offset, old level) pairs, and where each
  // move's entries start (-1 for stones loaded by fromArray, which undo
  // recomputes). Restoring is exact and much cheaper than recomputing.
  private readonly changes: number[] = [];
  private readonly moveStarts: number[] = [];
  private recording = false;

  constructor() {
    for (let action = 0; action < SIZE * SIZE; action += 1) this.cells[ACTION_TO_INDEX[action]] = EMPTY;
    for (const player of [BLACK, WHITE]) {
      for (let action = 0; action < SIZE * SIZE; action += 1) this.summarize(ACTION_TO_INDEX[action], player);
    }
  }

  /** Build from a flat environment board: -1 empty, 0 black, 1 white. */
  static fromArray(board: ArrayLike<number>, player: number) {
    const result = new PatternBoard();
    for (let action = 0; action < SIZE * SIZE; action += 1) {
      if (board[action] === -1) continue;
      const index = ACTION_TO_INDEX[action];
      const stone = board[action] === 0 ? BLACK : WHITE;
      result.cells[index] = stone;
      result.hashHigh = (result.hashHigh ^ ZOBRIST_HIGH[stone][index]) >>> 0;
      result.hashLow ^= ZOBRIST_LOW[stone][index];
      result.moves.push(action);
      result.moveStarts.push(-1);
      for (const neighbour of NEIGHBOURS[index]) result.near[neighbour] += 1;
    }
    result.turn = player === 0 ? BLACK : WHITE;
    for (let action = 0; action < SIZE * SIZE; action += 1) {
      const index = ACTION_TO_INDEX[action];
      if (result.cells[index] !== EMPTY) continue;
      if (result.near[index]) result.candidates.add(index);
      for (let direction = 0; direction < 4; direction += 1) result.refresh(index, direction);
    }
    return result;
  }

  get hash() {
    return this.hashHigh * 2_097_152 + this.hashLow;
  }

  /** Environment player id (0 black, 1 white) of the side to move. */
  get player() {
    return this.turn - 1;
  }

  isFull() {
    return this.moves.length === SIZE * SIZE;
  }

  /** Flat environment board: -1 empty, 0 black, 1 white. */
  toArray() {
    const board = new Int8Array(SIZE * SIZE);
    for (let action = 0; action < SIZE * SIZE; action += 1) {
      const cell = this.cells[ACTION_TO_INDEX[action]];
      board[action] = cell === EMPTY ? -1 : cell - 1;
    }
    return board;
  }

  play(index: number) {
    const stone = this.turn;
    this.cells[index] = stone;
    this.hashHigh = (this.hashHigh ^ ZOBRIST_HIGH[stone][index]) >>> 0;
    this.hashLow ^= ZOBRIST_LOW[stone][index];
    this.moves.push(INDEX_TO_ACTION[index]);
    this.candidates.delete(index);
    for (const neighbour of NEIGHBOURS[index]) {
      this.near[neighbour] += 1;
      if (this.cells[neighbour] === EMPTY) this.candidates.add(neighbour);
    }
    this.moveStarts.push(this.changes.length);
    this.recording = true;
    this.refreshLines(index);
    this.recording = false;
    this.turn = stone === BLACK ? WHITE : BLACK;
  }

  undo() {
    const index = ACTION_TO_INDEX[this.moves.pop() as number];
    const stone = this.cells[index];
    this.cells[index] = EMPTY;
    this.hashHigh = (this.hashHigh ^ ZOBRIST_HIGH[stone][index]) >>> 0;
    this.hashLow ^= ZOBRIST_LOW[stone][index];
    for (const neighbour of NEIGHBOURS[index]) {
      this.near[neighbour] -= 1;
      if (!this.near[neighbour]) this.candidates.delete(neighbour);
    }
    if (this.near[index]) this.candidates.add(index);
    const start = this.moveStarts.pop() as number;
    if (start < 0) {
      this.refreshLines(index);
    } else {
      // The undone cell's own levels were never touched while it was occupied.
      const changes = this.changes;
      const span = 4 * WIDTH * WIDTH;
      while (changes.length > start) {
        const old = changes.pop() as number;
        const key = changes.pop() as number;
        const player = key >= span ? WHITE : BLACK;
        const offset = key - (player === WHITE ? span : 0);
        this.levels[player][offset] = old;
        this.summarize(offset >> 2, player);
      }
    }
    this.turn = stone;
  }

  winningCells(player: number) {
    const result: number[] = [];
    const flags = this.flags[player];
    for (const index of this.candidates) if (flags[index] & WIN_FLAG) result.push(index);
    return result;
  }

  private refreshLines(index: number) {
    const lines = LINE_CELLS[index];
    for (let direction = 0; direction < 4; direction += 1) {
      for (const cell of lines[direction]) if (this.cells[cell] === EMPTY) this.refresh(cell, direction);
    }
  }

  private refresh(index: number, direction: number) {
    const step = DIRECTIONS[direction];
    let key = 0;
    let slot = 0;
    for (let k = -PAD; k <= PAD; k += 1) {
      if (k === 0) continue;
      const value = this.cells[index + k * step];
      this.raw[slot] = value;
      key = key * 4 + value;
      slot += 1;
    }
    const packed = lineLevels(key, this.raw);
    const black = packed >> 3;
    const white = packed & 7;
    const offset = index * 4 + direction;
    if (this.levels[BLACK][offset] !== black) {
      if (this.recording) this.changes.push(offset, this.levels[BLACK][offset]);
      this.levels[BLACK][offset] = black;
      this.summarize(index, BLACK);
    }
    if (this.levels[WHITE][offset] !== white) {
      if (this.recording) this.changes.push(4 * WIDTH * WIDTH + offset, this.levels[WHITE][offset]);
      this.levels[WHITE][offset] = white;
      this.summarize(index, WHITE);
    }
  }

  private summarize(index: number, player: number) {
    const levels = this.levels[player];
    const base = index * 4;
    const key = ((levels[base] * 7 + levels[base + 1]) * 7 + levels[base + 2]) * 7 + levels[base + 3];
    this.order[player][index] = SUMMARY_ORDER[key];
    this.attack[player][index] = SUMMARY_ATTACK[key];
    this.defence[player][index] = SUMMARY_DEFENCE[key];
    this.flags[player][index] = SUMMARY_FLAGS[key];
  }
}

/** Exposed for parity tests. */
export function lineLevelForTest(window: number[]) {
  return level([...window]);
}
