'use client';

import { Bot, CircleHelp, Code2, RotateCcw, UserRound } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { Button } from '@/components/ui/button';

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

export default function Home() {
  const [board, setBoard] = useState<Stone[]>(() => Array(CELLS).fill(null));
  const [human, setHuman] = useState<Player>(0);
  const [turn, setTurn] = useState<Player>(0);
  const [result, setResult] = useState<Result>(null);
  const [winningCells, setWinningCells] = useState<number[]>([]);
  const [lastMove, setLastMove] = useState<number | null>(null);

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

  useEffect(() => {
    if (result !== null || turn !== bot) return;
    const timer = window.setTimeout(() => {
      const open = board.flatMap((stone, index) => (stone === null ? [index] : []));
      if (open.length > 0) {
        const choice = open[Math.floor(Math.random() * open.length)];
        playMove(choice, bot);
      }
    }, 420);
    return () => window.clearTimeout(timer);
  }, [board, bot, playMove, result, turn]);

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
        ? 'Random bot is choosing…'
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
              <div><span>Opponent</span><strong>Random</strong></div>
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
