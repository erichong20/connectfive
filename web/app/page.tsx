'use client';

import { Bot, CircleHelp, Code2, RotateCcw, UserRound } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import type { BotRequest, BotResponse } from '@/lib/engine/bot.worker';

type WebMcpTool = {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute(input: unknown): unknown | Promise<unknown>;
};

declare global {
  interface Document {
    readonly modelContext?: {
      registerTool(tool: WebMcpTool, options?: { signal?: AbortSignal }): void | Promise<void>;
    };
  }
}

const SIZE = 15;
const CELLS = SIZE * SIZE;
const COLUMNS = 'ABCDEFGHJKLMNOP';
const STAR_POINTS = new Set([48, 56, 112, 168, 176]);

type Stone = 0 | 1 | null;
type Player = 0 | 1;
type Result = Player | 'draw' | null;
type BotKind = 'neural' | 'lookahead' | 'tactical' | 'random';

const BOT_LABELS: Record<BotKind, string> = {
  neural: 'Neural',
  lookahead: 'Look-ahead',
  tactical: 'Tactical',
  random: 'Random',
};
const BOT_HINTS: Record<BotKind, string> = {
  neural: 'Strongest: a self-play-trained network guiding a tree search, with exact win and block checks.',
  lookahead: 'Look-ahead considers the opponent’s strongest reply before moving.',
  tactical: 'Tactical wins, blocks, and extends its best line one move at a time.',
  random: 'Random plays any open intersection.',
};
// The neural bot searches for up to this long (or this many simulations) per move.
// With at least 4 cores it runs the stronger 64-channel az-r9 network on a pool
// of network workers (batched search); otherwise the smaller az-r2 serially.
// See experiments/2026-10-04-browser-az-r7/ and 2026-10-05-alphazero-lite-r9/.
const NEURAL_MODEL_URL = '/models/az-r9/';
const NEURAL_FALLBACK_MODEL_URL = '/models/az-r2/';
const NEURAL_MIN_CORES = 4;
const NEURAL_MAX_WORKERS = 4;
// Built from lib/engine/*.worker.ts by scripts/build-engine.mjs.
const NEURAL_WORKER_URL = '/engine/bot-worker.js';
const NEURAL_NET_WORKER_URL = '/engine/net-worker.js';

function neuralSetup() {
  const cores = typeof navigator === 'undefined' ? 1 : navigator.hardwareConcurrency || 1;
  return cores >= NEURAL_MIN_CORES
    ? { modelUrl: NEURAL_MODEL_URL, workers: Math.min(NEURAL_MAX_WORKERS, cores - 1) }
    : { modelUrl: NEURAL_FALLBACK_MODEL_URL, workers: 1 };
}
const NEURAL_TIME_LIMIT_MS = 1_500;
const NEURAL_MAX_SIMULATIONS = 600;

const ROOT_WIDTH = 16;
const REPLY_WIDTH = 12;

function coordinate(index: number) {
  const row = Math.floor(index / SIZE) + 1;
  return `${COLUMNS[index % SIZE]}${row}`;
}

function coordinateIndex(value: string) {
  const match = /^([A-HJ-P])(1[0-5]|[1-9])$/i.exec(value.trim());
  if (!match) throw new Error('Use a Go coordinate from A1 through P15, skipping I.');
  return (Number(match[2]) - 1) * SIZE + COLUMNS.indexOf(match[1].toUpperCase());
}

function winningRun(board: Stone[], lastMove: number, player: Player) {
  const row = Math.floor(lastMove / SIZE);
  const col = lastMove % SIZE;
  const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];

  for (const [dr, dc] of directions) {
    const run = [lastMove];
    for (const sign of [-1, 1]) {
      for (let distance = 1; distance < SIZE; distance += 1) {
        const r = row + dr * distance * sign;
        const c = col + dc * distance * sign;
        if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) break;
        const index = r * SIZE + c;
        if (board[index] !== player) break;
        run.push(index);
      }
    }
    if (run.length === 5) return run;
  }
  return [];
}

function runAndOpenEnds(
  board: Stone[],
  index: number,
  player: Player,
  dr: number,
  dc: number,
) {
  const row = Math.floor(index / SIZE);
  const col = index % SIZE;
  let run = 1;
  let openEnds = 0;

  for (const sign of [-1, 1]) {
    for (let distance = 1; distance < SIZE; distance += 1) {
      const r = row + sign * dr * distance;
      const c = col + sign * dc * distance;
      if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) break;
      const stone = board[r * SIZE + c];
      if (stone === player) {
        run += 1;
        continue;
      }
      if (stone === null) openEnds += 1;
      break;
    }
  }
  return { run, openEnds };
}

function isExactFiveAfter(board: Stone[], index: number, player: Player) {
  if (board[index] !== null) return false;
  return [[0, 1], [1, 0], [1, 1], [1, -1]].some(([dr, dc]) => (
    runAndOpenEnds(board, index, player, dr, dc).run === 5
  ));
}

function patternValue(run: number, openEnds: number) {
  if (run === 5) return 100_000;
  if (run === 4) return openEnds === 2 ? 12_000 : openEnds === 1 ? 3_000 : 0;
  if (run === 3) return openEnds === 2 ? 1_200 : openEnds === 1 ? 250 : 0;
  if (run === 2) return openEnds === 2 ? 80 : openEnds === 1 ? 15 : 0;
  return 0;
}

function nearbyMoves(board: Stone[], open: number[]) {
  const occupied = board.flatMap((stone, index) => (stone === null ? [] : [index]));
  if (occupied.length === 0) return [Math.floor(CELLS / 2)];
  const nearby = open.filter((index) => {
    const row = Math.floor(index / SIZE);
    const col = index % SIZE;
    return occupied.some((stoneIndex) => {
      const stoneRow = Math.floor(stoneIndex / SIZE);
      const stoneCol = stoneIndex % SIZE;
      return Math.max(Math.abs(stoneRow - row), Math.abs(stoneCol - col)) <= 2;
    });
  });
  return nearby.length > 0 ? nearby : open;
}

function tacticalScore(board: Stone[], index: number, player: Player) {
  const opponent = (1 - player) as Player;
  const row = Math.floor(index / SIZE);
  const col = index % SIZE;
  let score = 0;

  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    const own = runAndOpenEnds(board, index, player, dr, dc);
    const theirs = runAndOpenEnds(board, index, opponent, dr, dc);
    score += patternValue(own.run, own.openEnds);
    score += Math.floor(1.1 * patternValue(theirs.run, theirs.openEnds));
  }

  for (let r = Math.max(0, row - 2); r <= Math.min(SIZE - 1, row + 2); r += 1) {
    for (let c = Math.max(0, col - 2); c <= Math.min(SIZE - 1, col + 2); c += 1) {
      if (board[r * SIZE + c] !== null) {
        score += Math.max(Math.abs(r - row), Math.abs(c - col)) === 1 ? 8 : 2;
      }
    }
  }

  const center = Math.floor(SIZE / 2);
  return score - Math.abs(row - center) - Math.abs(col - center);
}

function orderedCandidates(
  board: Stone[],
  open: number[],
  player: Player,
  width: number,
) {
  const opponent = (1 - player) as Player;
  const nearby = nearbyMoves(board, open);
  const wins = nearby.filter((index) => isExactFiveAfter(board, index, player));
  const winSet = new Set(wins);
  const blocks = nearby.filter((index) => (
    !winSet.has(index) && isExactFiveAfter(board, index, opponent)
  ));
  const forcing = new Set([...wins, ...blocks]);
  const remainder = nearby
    .filter((index) => !forcing.has(index))
    .sort((a, b) => tacticalScore(board, b, player) - tacticalScore(board, a, player) || a - b);
  return [...wins, ...blocks, ...remainder].slice(0, width);
}

function boardAfter(board: Stone[], index: number, player: Player) {
  const next = [...board];
  next[index] = player;
  return next;
}

function replyCandidates(
  board: Stone[],
  open: number[],
  player: Player,
  focus: number,
  baseline: number[],
) {
  const opponent = (1 - player) as Player;
  const nearby = nearbyMoves(board, open);
  const wins = nearby.filter((index) => isExactFiveAfter(board, index, player));
  const winSet = new Set(wins);
  const blocks = nearby.filter((index) => (
    !winSet.has(index) && isExactFiveAfter(board, index, opponent)
  ));
  const focusRow = Math.floor(focus / SIZE);
  const focusCol = focus % SIZE;
  const local = nearby.filter((index) => Math.max(
    Math.abs(Math.floor(index / SIZE) - focusRow),
    Math.abs(index % SIZE - focusCol),
  ) <= 2);
  const legal = new Set(open);
  const forcing = new Set([...wins, ...blocks]);
  const pool = [...new Set([
    ...wins,
    ...blocks,
    ...baseline.filter((index) => legal.has(index)),
    ...local,
  ])];
  const remainder = pool
    .filter((index) => !forcing.has(index))
    .sort((a, b) => tacticalScore(board, b, player) - tacticalScore(board, a, player) || a - b);
  return [...wins, ...blocks, ...remainder].slice(0, REPLY_WIDTH);
}

function replyValue(board: Stone[], reply: number, player: Player) {
  const opponent = (1 - player) as Player;
  if (isExactFiveAfter(board, reply, opponent)) return -1_000_000;

  const replyThreat = tacticalScore(board, reply, opponent);
  const leaf = boardAfter(board, reply, opponent);
  const open = leaf.flatMap((stone, index) => (stone === null ? [index] : []));
  if (open.length === 0) return 0;
  const candidates = nearbyMoves(leaf, open);
  const ownWins = candidates.filter((index) => isExactFiveAfter(leaf, index, player)).length;
  if (ownWins > 0) return 500_000 + 50_000 * (ownWins - 1);

  const opponentWins = candidates.filter((index) => (
    isExactFiveAfter(leaf, index, opponent)
  )).length;
  if (opponentWins >= 2) return -500_000;

  return -Math.floor(1.1 * replyThreat) - (opponentWins === 1 ? 100_000 : 0);
}

function chooseBotMove(board: Stone[], player: Player, kind: BotKind) {
  const open = board.flatMap((stone, index) => (stone === null ? [index] : []));
  if (kind === 'random') return open[Math.floor(Math.random() * open.length)];

  const candidates = nearbyMoves(board, open);
  const winning = candidates.filter((index) => isExactFiveAfter(board, index, player));
  if (winning.length > 0) return winning[Math.floor(Math.random() * winning.length)];

  if (kind === 'lookahead') {
    const roots = orderedCandidates(board, open, player, ROOT_WIDTH);
    const baselineReplies = orderedCandidates(
      board, open, (1 - player) as Player, REPLY_WIDTH,
    );
    const scored = roots.map((index) => {
      const afterMove = boardAfter(board, index, player);
      const replyOpen = open.filter((reply) => reply !== index);
      const replies = replyCandidates(
        afterMove, replyOpen, (1 - player) as Player, index, baselineReplies,
      );
      const worstReply = replies.length === 0
        ? 0
        : Math.min(...replies.map((reply) => replyValue(afterMove, reply, player)));
      return { index, score: tacticalScore(board, index, player) + worstReply };
    });
    const bestScore = Math.max(...scored.map(({ score }) => score));
    const best = scored.filter(({ score }) => score === bestScore);
    return best[Math.floor(Math.random() * best.length)].index;
  }

  const opponent = (1 - player) as Player;
  const blocks = candidates.filter((index) => isExactFiveAfter(board, index, opponent));
  if (blocks.length > 0) return blocks[Math.floor(Math.random() * blocks.length)];

  const scored = candidates.map((index) => ({ index, score: tacticalScore(board, index, player) }));
  const bestScore = Math.max(...scored.map(({ score }) => score));
  const best = scored.filter(({ score }) => score === bestScore);
  return best[Math.floor(Math.random() * best.length)].index;
}

export default function Home() {
  const [board, setBoard] = useState<Stone[]>(() => Array(CELLS).fill(null));
  const [human, setHuman] = useState<Player>(0);
  const [turn, setTurn] = useState<Player>(0);
  const [result, setResult] = useState<Result>(null);
  const [winningCells, setWinningCells] = useState<number[]>([]);
  const [lastMove, setLastMove] = useState<number | null>(null);
  const [botKind, setBotKind] = useState<BotKind>('neural');
  const [botError, setBotError] = useState<string | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const requestRef = useRef(0);

  const moveCount = useMemo(() => board.filter((stone) => stone !== null).length, [board]);
  const bot = (1 - human) as Player;
  const botThinking = result === null && turn === bot;

  const playMove = useCallback((index: number, player: Player) => {
    if (board[index] !== null || result !== null) return;
    const nextBoard = [...board];
    nextBoard[index] = player;
    const run = winningRun(nextBoard, index, player);

    setBoard(nextBoard);
    setLastMove(index);
    if (run.length === 5) {
      setWinningCells(run);
      setResult(player);
    } else if (nextBoard.every((stone) => stone !== null)) {
      setResult('draw');
    } else {
      setTurn((1 - player) as Player);
    }
  }, [board, result]);

  useEffect(() => () => {
    workerRef.current?.terminate();
    workerRef.current = null;
  }, []);

  useEffect(() => {
    if (result !== null || turn !== bot) return;
    if (botKind !== 'neural') {
      const timer = window.setTimeout(() => {
        const choice = chooseBotMove(board, bot, botKind);
        if (choice !== undefined) playMove(choice, bot);
      }, 420);
      return () => window.clearTimeout(timer);
    }

    // Neural bot: search in a Web Worker; ignore answers to outdated positions.
    let cancelled = false;
    const id = ++requestRef.current;
    const fallback = (message: string) => {
      setBotError(`Neural bot unavailable (${message}); Look-ahead played instead.`);
      const choice = chooseBotMove(board, bot, 'lookahead');
      if (choice !== undefined) playMove(choice, bot);
    };
    let worker = workerRef.current;
    if (!worker) {
      try {
        worker = new Worker(NEURAL_WORKER_URL, { type: 'module' });
        workerRef.current = worker;
      } catch (error) {
        fallback(error instanceof Error ? error.message : 'worker failed to start');
        return;
      }
    }
    const onMessage = (event: MessageEvent<BotResponse>) => {
      if (cancelled || event.data.id !== id) return;
      if ('error' in event.data) fallback(event.data.error);
      else playMove(event.data.action, bot);
    };
    const onError = (event: ErrorEvent) => {
      if (!cancelled) fallback(event.message || 'worker error');
    };
    worker.addEventListener('message', onMessage);
    worker.addEventListener('error', onError);
    const request: BotRequest = {
      id,
      board: board.map((stone) => (stone === null ? -1 : stone)),
      player: bot,
      ...neuralSetup(),
      netWorkerUrl: NEURAL_NET_WORKER_URL,
      fallbackModelUrl: NEURAL_FALLBACK_MODEL_URL,
      timeLimitMs: NEURAL_TIME_LIMIT_MS,
      maxSimulations: NEURAL_MAX_SIMULATIONS,
    };
    worker.postMessage(request);
    return () => {
      cancelled = true;
      worker?.removeEventListener('message', onMessage);
      worker?.removeEventListener('error', onError);
    };
  }, [board, bot, botKind, playMove, result, turn]);

  const reset = useCallback((nextHuman = human) => {
    setBoard(Array(CELLS).fill(null));
    setHuman(nextHuman);
    setTurn(0);
    setResult(null);
    setWinningCells([]);
    setLastMove(null);
  }, [human]);

  const status = result === 'draw'
    ? 'The board is full — draw.'
    : result !== null
      ? result === human
        ? 'You connected five. You win!'
        : 'The bot connected five.'
      : botThinking
        ? `${BOT_LABELS[botKind]} bot is ${botKind === 'neural' ? 'thinking' : 'choosing'}…`
        : turn === human
          ? 'Your move'
          : 'Bot to move';

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const reportError = (error: unknown) => console.warn('WebMCP registration failed', error);
    const tools: WebMcpTool[] = [
      {
        name: 'start_new_connect_five_game',
        title: 'Start a new Connect Five game',
        description: 'Clear the board and start a new game with the human playing Black or White.',
        inputSchema: {
          type: 'object',
          properties: { humanColor: { type: 'string', enum: ['black', 'white'] } },
          required: ['humanColor'],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute(input) {
          const humanColor = (input as { humanColor?: unknown })?.humanColor;
          if (humanColor !== 'black' && humanColor !== 'white') throw new Error('humanColor must be black or white.');
          reset(humanColor === 'black' ? 0 : 1);
          return { started: true, humanColor };
        },
      },
      {
        name: 'play_connect_five_move',
        title: 'Play a Connect Five move',
        description: 'Place the human player’s stone at one legal Go coordinate on the visible board.',
        inputSchema: {
          type: 'object',
          properties: { coordinate: { type: 'string', description: 'Go coordinate such as K10.' } },
          required: ['coordinate'],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute(input) {
          const value = (input as { coordinate?: unknown })?.coordinate;
          if (typeof value !== 'string') throw new Error('coordinate must be a string.');
          if (result !== null) throw new Error('The game is already over. Start a new game first.');
          if (turn !== human) throw new Error('Wait for the bot to finish its move.');
          const index = coordinateIndex(value);
          if (board[index] !== null) throw new Error(`${coordinate(index)} is occupied.`);
          playMove(index, human);
          return { played: coordinate(index), color: human === 0 ? 'black' : 'white' };
        },
      },
    ];

    for (const tool of tools) {
      try {
        void Promise.resolve(context.registerTool(tool, { signal: lifecycle.signal })).catch(reportError);
      } catch (error) {
        reportError(error);
      }
    }
    return () => lifecycle.abort();
  }, [board, human, playMove, reset, result, turn]);

  return (
    <main className="min-h-screen px-4 py-5 sm:px-7 sm:py-7">
      <div className="mx-auto max-w-[1180px]">
        <header className="mb-5 flex items-center justify-between gap-4">
          <div>
            <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-[#8a6240]">
              <span className="inline-block size-2 rounded-full bg-[#d2613c]" />
              15 × 15 Gomoku
            </div>
            <h1 className="font-heading text-2xl font-semibold tracking-[-0.04em] text-[#211a15] sm:text-3xl">Connect Five</h1>
          </div>
          <a className="inline-flex size-10 items-center justify-center rounded-full border border-[#d8c7ae] bg-[#fffaf0] text-[#594735] transition hover:-translate-y-0.5 hover:border-[#a9845c] hover:text-[#211a15]" href="https://github.com/erichong20/connectfive" aria-label="View source on GitHub">
            <Code2 className="size-4" />
          </a>
        </header>

        <section className="game-shell">
          <div className="board-panel">
            <div className="board-wrap" aria-label="15 by 15 Gomoku board">
              <div className="board" aria-label="Game board">
                {board.map((stone, index) => {
                  const isHumanTurn = turn === human && result === null && !botThinking;
                  const isWinning = winningCells.includes(index);
                  return (
                    <button
                      key={index}
                      type="button"
                      className={`intersection ${stone !== null ? 'occupied' : ''}`}
                      aria-label={`${coordinate(index)}${stone === 0 ? ', black stone' : stone === 1 ? ', white stone' : ', empty'}`}
                      disabled={!isHumanTurn || stone !== null}
                      onClick={() => playMove(index, human)}
                    >
                      {STAR_POINTS.has(index) && stone === null && <span className="star" />}
                      {stone !== null && (
                        <span className={`stone ${stone === 0 ? 'black' : 'white'} ${isWinning ? 'winner' : ''} ${lastMove === index ? 'last' : ''}`} />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="coordinate-note"><span>A</span><span>15 × 15</span><span>P</span></div>
          </div>

          <aside className="control-panel">
            <div className="status-card" aria-live="polite">
              <div className="status-icon">{turn === human ? <UserRound className="size-5" /> : <Bot className="size-5" />}</div>
              <div><p className="eyebrow">Game status</p><p className="status-copy">{status}</p></div>
            </div>

            <div className="control-section">
              <p className="control-label">Opponent</p>
              <div className="color-picker opponent-picker">
                {(['neural', 'lookahead', 'tactical', 'random'] as const).map((kind) => (
                  <button key={kind} type="button" className={botKind === kind ? 'selected' : ''} onClick={() => { setBotKind(kind); setBotError(null); reset(); }} aria-pressed={botKind === kind}>
                    {BOT_LABELS[kind]}
                  </button>
                ))}
              </div>
              <p className="hint">{BOT_HINTS[botKind]}</p>
              {botError && <output className="hint block">{botError}</output>}
            </div>

            <div className="control-section">
              <p className="control-label">Play as</p>
              <div className="color-picker">
                <button type="button" className={human === 0 ? 'selected' : ''} onClick={() => reset(0)} aria-pressed={human === 0}>
                  <span className="mini-stone black" /> Black
                </button>
                <button type="button" className={human === 1 ? 'selected' : ''} onClick={() => reset(1)} aria-pressed={human === 1}>
                  <span className="mini-stone white" /> White
                </button>
              </div>
              <p className="hint">Choose White and the bot makes the opening move.</p>
            </div>

            <div className="score-row">
              <div><span>Moves</span><strong>{moveCount}</strong></div>
              <div><span>Opponent</span><strong>{BOT_LABELS[botKind]}</strong></div>
            </div>

            <Button size="lg" className="h-11 w-full rounded-xl bg-[#2f4f3e] text-[#fffaf0] hover:bg-[#20392d]" onClick={() => reset()}>
              <RotateCcw data-icon="inline-start" /> New game
            </Button>

            <div className="rules-note">
              <CircleHelp className="mt-0.5 size-4 shrink-0" />
              <p>Place a stone on any open intersection. Exactly five in any direction wins; overlines do not.</p>
            </div>
          </aside>
        </section>

        <footer>Standard Gomoku · Black moves first · Exact five wins</footer>
      </div>
    </main>
  );
}
