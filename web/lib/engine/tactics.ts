// Port of PatternSearch.generate and PatternSearch.vcf from
// src/connectfive/pattern_search.py: exact tactics that guided MCTS uses
// before asking the network anything.

import { BLACK, FORCE_FLAG, FOUR_FLAG, PatternBoard, WHITE, WIN_FLAG } from './patterns';

export const QUIET = 0;
export const WIN = 1;
export const LOSS = 2;
export const FORCE_WIN = 3;
export const BLOCK = 4;
export const DEFEND = 5;

export type Generated = { kind: number; moves: number[] };

export class Tactics {
  vcfNodes = 0;
  private readonly vcfFailures = new Map<number, number>();

  constructor(private readonly board: PatternBoard) {}

  /** Classify the position for the side to move and list sensible moves. */
  generate(): Generated {
    const board = this.board;
    const me = board.turn;
    const opponent = me === BLACK ? WHITE : BLACK;
    const mine = board.flags[me];
    const theirs = board.flags[opponent];
    const myOrder = board.order[me];
    const theirOrder = board.order[opponent];

    const wins: number[] = [];
    const blocks: number[] = [];
    const forcing: number[] = [];
    const threats: number[] = [];
    for (const index of board.candidates) {
      if (mine[index] & WIN_FLAG) wins.push(index);
      if (theirs[index] & WIN_FLAG) blocks.push(index);
      if (mine[index] & FORCE_FLAG) forcing.push(index);
      if (theirs[index] & FORCE_FLAG) threats.push(index);
    }
    if (wins.length) return { kind: WIN, moves: [Math.min(...wins)] };
    if (blocks.length >= 2) return { kind: LOSS, moves: blocks.sort((a, b) => a - b) };
    if (blocks.length) return { kind: BLOCK, moves: blocks };
    if (forcing.length) {
      let best = forcing[0];
      for (const index of forcing) {
        if (myOrder[index] > myOrder[best] || (myOrder[index] === myOrder[best] && index < best)) best = index;
      }
      return { kind: FORCE_WIN, moves: [best] };
    }
    const byCombined = (a: number, b: number) =>
      myOrder[b] + theirOrder[b] - (myOrder[a] + theirOrder[a]) || a - b;
    if (threats.length) {
      const defence: number[] = [];
      for (const index of board.candidates) {
        if (theirs[index] & FOUR_FLAG || mine[index] & FOUR_FLAG) defence.push(index);
      }
      return { kind: DEFEND, moves: defence.sort(byCombined) };
    }
    return { kind: QUIET, moves: [...board.candidates].sort(byCombined) };
  }

  /** First move of a proven win by continuous fours, or null. */
  vcf(depth: number): number | null {
    const board = this.board;
    const me = board.turn;
    const opponent = me === BLACK ? WHITE : BLACK;
    const mine = board.flags[me];
    const theirs = board.flags[opponent];
    const myOrder = board.order[me];

    let firstWin = -1;
    const blocks: number[] = [];
    let fours: number[] = [];
    for (const index of board.candidates) {
      if (mine[index] & WIN_FLAG && (firstWin < 0 || index < firstWin)) firstWin = index;
      if (theirs[index] & WIN_FLAG) blocks.push(index);
      if (mine[index] & FOUR_FLAG) fours.push(index);
    }
    if (firstWin >= 0) return firstWin;
    if (depth <= 0) return null;
    const failed = this.vcfFailures.get(board.hash);
    if (failed !== undefined && failed >= depth) return null;
    if (blocks.length >= 2) return null;
    if (blocks.length) fours = fours.filter((index) => index === blocks[0]);
    fours.sort((a, b) => myOrder[b] - myOrder[a] || a - b);
    for (const move of fours) {
      this.vcfNodes += 1;
      board.play(move);
      const replies = board.winningCells(me);
      let proved = replies.length >= 2;
      if (replies.length === 1) {
        board.play(replies[0]);
        proved = this.vcf(depth - 1) !== null;
        board.undo();
      }
      board.undo();
      if (proved) return move;
    }
    this.vcfFailures.set(board.hash, depth);
    return null;
  }
}
