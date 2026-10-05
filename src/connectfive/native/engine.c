// Native core for guided MCTS: the exact-five pattern board, tactics/VCF, and a
// batched PUCT tree. A port of src/connectfive/{patterns,pattern_search,
// guided_search}.py (via web/lib/engine/*.ts). Python keeps the network: the
// search hands out batches of leaf positions as input features and receives
// policy logits and values back. Loaded with ctypes by native/__init__.py.
//
// Semantics match GuidedMCTS(batch_size=B) in Python: pick up to B leaves under
// virtual loss, back up tactical/terminal leaves at once, stop picking at a
// pending leaf, then evaluate and back up the pending ones in pick order.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 15
#define PAD 5
#define WIDTH (SIZE + 2 * PAD)
#define CELLS (WIDTH * WIDTH)
#define ACTIONS (SIZE * SIZE)
#define EMPTY 0
#define BLACK 1
#define WHITE 2
#define WALL 3

#define WIN_FLAG 1
#define FORCE_FLAG 2
#define FOUR_FLAG 4

enum { QUIET = 0, WIN = 1, LOSS = 2, FORCE_WIN = 3, BLOCK = 4, DEFEND = 5 };

static const int DIRECTIONS[4] = {1, WIDTH, WIDTH + 1, WIDTH - 1};
static int ACTION_TO_INDEX[ACTIONS];
static int INDEX_TO_ACTION[CELLS];
static int NEIGHBOURS[CELLS][24];
static int NEIGHBOUR_COUNT[CELLS];
static int LINE_CELLS[CELLS][4][11];
static int LINE_COUNT[CELLS][4];
static uint64_t ZOBRIST[3][CELLS];

static int32_t SUMMARY_ORDER[2401];
static uint8_t SUMMARY_FLAGS[2401];
static int8_t LEVEL_MEMO[177147];  // 3^11 windows: 0 empty, 1 own, 2 blocked
static int8_t LINE_MEMO[1048576];  // 4^10 raw neighbourhoods -> black * 8 + white
static int initialized = 0;

static int on_board(int index) { return index >= 0 && index < CELLS && INDEX_TO_ACTION[index] >= 0; }

static uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static int run_through_center(const int *w) {
  int run = 1;
  for (int i = PAD - 1; i >= 0 && w[i] == 1; i--) run++;
  for (int i = PAD + 1; i <= 2 * PAD && w[i] == 1; i++) run++;
  return run;
}

static int level(int *w) {
  int key = 0;
  for (int i = 0; i <= 2 * PAD; i++) key = key * 3 + w[i];
  if (LEVEL_MEMO[key] >= 0) return LEVEL_MEMO[key];
  int result = 0, run = run_through_center(w);
  if (run == 5) {
    result = 6;
  } else if (run < 5) {
    int empties[2 * PAD], n = 0, winning = 0;
    for (int i = 1; i < 2 * PAD; i++)
      if (w[i] == 0) empties[n++] = i;
    for (int k = 0; k < n; k++) {
      w[empties[k]] = 1;
      if (run_through_center(w) == 5) winning++;
      w[empties[k]] = 0;
    }
    if (winning >= 2) result = 5;
    else if (winning == 1) result = 4;
    else {
      int best = 0;
      for (int k = 0; k < n && best != 5; k++) {
        w[empties[k]] = 1;
        int value = level(w);
        if (value > best) best = value;
        w[empties[k]] = 0;
      }
      result = best == 5 ? 3 : best == 4 ? 2 : best == 3 ? 1 : 0;
    }
  }
  LEVEL_MEMO[key] = (int8_t)result;
  return result;
}

static const int TRANSLATE[3][4] = {{0, 0, 0, 0}, {0, 1, 2, 2}, {0, 2, 1, 2}};

static int line_levels(int key, const int *raw) {
  if (LINE_MEMO[key] >= 0) return LINE_MEMO[key];
  int packed = 0, w[2 * PAD + 1];
  for (int player = BLACK; player <= WHITE; player++) {
    for (int i = 0; i < PAD; i++) w[i] = TRANSLATE[player][raw[i]];
    w[PAD] = 1;
    for (int i = PAD; i < 2 * PAD; i++) w[i + 1] = TRANSLATE[player][raw[i]];
    packed = packed * 8 + level(w);
  }
  LINE_MEMO[key] = (int8_t)packed;
  return packed;
}

static void init_tables(void) {
  if (initialized) return;
  for (int i = 0; i < CELLS; i++) INDEX_TO_ACTION[i] = -1;
  for (int a = 0; a < ACTIONS; a++) {
    int index = (a / SIZE + PAD) * WIDTH + a % SIZE + PAD;
    ACTION_TO_INDEX[a] = index;
    INDEX_TO_ACTION[index] = a;
  }
  for (int index = 0; index < CELLS; index++) {
    NEIGHBOUR_COUNT[index] = 0;
    if (INDEX_TO_ACTION[index] < 0) continue;
    for (int dr = -2; dr <= 2; dr++)
      for (int dc = -2; dc <= 2; dc++) {
        int other = index + dr * WIDTH + dc;
        if ((dr || dc) && on_board(other)) NEIGHBOURS[index][NEIGHBOUR_COUNT[index]++] = other;
      }
    for (int d = 0; d < 4; d++) {
      LINE_COUNT[index][d] = 0;
      for (int k = -PAD; k <= PAD; k++) {
        int cell = index + k * DIRECTIONS[d];
        if (on_board(cell)) LINE_CELLS[index][d][LINE_COUNT[index][d]++] = cell;
      }
    }
  }
  static const int ORDER_W[7] = {0, 40, 300, 2500, 2000, 0, 0};
  for (int key = 0; key < 2401; key++) {
    int levels[4] = {key / 343 % 7, key / 49 % 7, key / 7 % 7, key % 7}, counts[7] = {0};
    int order = 0;
    for (int i = 0; i < 4; i++) {
      counts[levels[i]]++;
      order += ORDER_W[levels[i]];
    }
    int win = counts[6] > 0, fours = counts[5] + counts[4];
    int force = !win && (counts[5] > 0 || fours >= 2);
    if (win) order += 10000000;
    else if (force) order += 1000000;
    else if (counts[4] && counts[3]) order += 100000;
    else if (counts[3] >= 2) order += 50000;
    SUMMARY_ORDER[key] = order;
    SUMMARY_FLAGS[key] = (uint8_t)((win ? WIN_FLAG : 0) | (force ? FORCE_FLAG : 0) | (win || fours > 0 ? FOUR_FLAG : 0));
  }
  memset(LEVEL_MEMO, -1, sizeof LEVEL_MEMO);
  memset(LINE_MEMO, -1, sizeof LINE_MEMO);
  uint64_t seed = 20260930;
  for (int p = BLACK; p <= WHITE; p++)
    for (int i = 0; i < CELLS; i++) ZOBRIST[p][i] = splitmix64(&seed);
  initialized = 1;
}

// ----- board ------------------------------------------------------------------

typedef struct {
  int8_t cells[CELLS];
  int16_t near[CELLS];
  uint8_t levels[3][CELLS * 4];
  int32_t order[3][CELLS];
  uint8_t flags[3][CELLS];
  int candidates[CELLS], candidate_pos[CELLS], ncandidates;
  int moves[ACTIONS], nmoves;
  int turn;
  uint64_t hash;
  // Undo log: (player, offset, old level) triples; each move's start, -1 = recompute.
  int32_t *log;
  int log_len, log_cap;
  int move_start[ACTIONS];
} Board;

static void summarize(Board *b, int index, int player) {
  const uint8_t *l = &b->levels[player][index * 4];
  int key = ((l[0] * 7 + l[1]) * 7 + l[2]) * 7 + l[3];
  b->order[player][index] = SUMMARY_ORDER[key];
  b->flags[player][index] = SUMMARY_FLAGS[key];
}

static void add_candidate(Board *b, int index) {
  if (b->candidate_pos[index] >= 0) return;
  b->candidate_pos[index] = b->ncandidates;
  b->candidates[b->ncandidates++] = index;
}

static void remove_candidate(Board *b, int index) {
  int pos = b->candidate_pos[index];
  if (pos < 0) return;
  int last = b->candidates[--b->ncandidates];
  b->candidates[pos] = last;
  b->candidate_pos[last] = pos;
  b->candidate_pos[index] = -1;
}

static void log_push(Board *b, int player, int offset, int old) {
  if (b->log_len + 3 > b->log_cap) {
    b->log_cap = b->log_cap ? b->log_cap * 2 : 1 << 16;
    b->log = realloc(b->log, sizeof(int32_t) * b->log_cap);
  }
  b->log[b->log_len++] = player;
  b->log[b->log_len++] = offset;
  b->log[b->log_len++] = old;
}

static void refresh(Board *b, int index, int direction, int record) {
  int step = DIRECTIONS[direction], key = 0, raw[2 * PAD], slot = 0;
  for (int k = -PAD; k <= PAD; k++) {
    if (!k) continue;
    int value = b->cells[index + k * step];
    raw[slot++] = value;
    key = key * 4 + value;
  }
  int packed = line_levels(key, raw), offset = index * 4 + direction;
  int values[3] = {0, packed >> 3, packed & 7};
  for (int player = BLACK; player <= WHITE; player++) {
    if (b->levels[player][offset] != values[player]) {
      if (record) log_push(b, player, offset, b->levels[player][offset]);
      b->levels[player][offset] = (uint8_t)values[player];
      summarize(b, index, player);
    }
  }
}

static void refresh_lines(Board *b, int index, int record) {
  for (int d = 0; d < 4; d++)
    for (int k = 0; k < LINE_COUNT[index][d]; k++) {
      int cell = LINE_CELLS[index][d][k];
      if (b->cells[cell] == EMPTY) refresh(b, cell, d, record);
    }
}

static void board_init(Board *b) {
  int32_t *log = b->log;
  int cap = b->log_cap;
  memset(b, 0, sizeof *b);
  b->log = log;
  b->log_cap = cap;
  for (int i = 0; i < CELLS; i++) {
    b->cells[i] = WALL;
    b->candidate_pos[i] = -1;
  }
  for (int a = 0; a < ACTIONS; a++) b->cells[ACTION_TO_INDEX[a]] = EMPTY;
  for (int p = BLACK; p <= WHITE; p++)
    for (int a = 0; a < ACTIONS; a++) summarize(b, ACTION_TO_INDEX[a], p);
  b->turn = BLACK;
}

static void board_play(Board *b, int index) {
  int stone = b->turn;
  b->cells[index] = (int8_t)stone;
  b->hash ^= ZOBRIST[stone][index];
  b->moves[b->nmoves] = INDEX_TO_ACTION[index];
  remove_candidate(b, index);
  for (int k = 0; k < NEIGHBOUR_COUNT[index]; k++) {
    int n = NEIGHBOURS[index][k];
    b->near[n]++;
    if (b->cells[n] == EMPTY) add_candidate(b, n);
  }
  b->move_start[b->nmoves++] = b->log_len;
  refresh_lines(b, index, 1);
  b->turn = stone == BLACK ? WHITE : BLACK;
}

static void board_undo(Board *b) {
  int index = ACTION_TO_INDEX[b->moves[--b->nmoves]];
  int stone = b->cells[index];
  b->cells[index] = EMPTY;
  b->hash ^= ZOBRIST[stone][index];
  for (int k = 0; k < NEIGHBOUR_COUNT[index]; k++) {
    int n = NEIGHBOURS[index][k];
    if (!--b->near[n]) remove_candidate(b, n);
  }
  if (b->near[index]) add_candidate(b, index);
  int start = b->move_start[b->nmoves];
  while (b->log_len > start) {
    int old = b->log[--b->log_len], offset = b->log[--b->log_len], player = b->log[--b->log_len];
    b->levels[player][offset] = (uint8_t)old;
    summarize(b, offset >> 2, player);
  }
  b->turn = stone;
}

// ----- tactics ------------------------------------------------------------------

typedef struct {
  uint64_t key;  // 0 = empty slot
  int value;
} Entry;

typedef struct {
  Entry *slots;
  size_t cap, size;
} Table;

static int table_get(Table *t, uint64_t key, int *value) {
  if (!t->cap) return 0;
  key |= 1;
  for (size_t i = key & (t->cap - 1);; i = (i + 1) & (t->cap - 1)) {
    if (!t->slots[i].key) return 0;
    if (t->slots[i].key == key) {
      *value = t->slots[i].value;
      return 1;
    }
  }
}

static void table_put(Table *t, uint64_t key, int value) {
  if ((t->size + 1) * 2 > t->cap) {
    Entry *old = t->slots;
    size_t old_cap = t->cap;
    t->cap = old_cap ? old_cap * 2 : 1 << 12;
    t->slots = calloc(t->cap, sizeof(Entry));
    t->size = 0;
    for (size_t i = 0; i < old_cap; i++)
      if (old[i].key) table_put(t, old[i].key, old[i].value);
    free(old);
  }
  key |= 1;
  size_t i = key & (t->cap - 1);
  while (t->slots[i].key && t->slots[i].key != key) i = (i + 1) & (t->cap - 1);
  if (!t->slots[i].key) t->size++;
  t->slots[i].key = key;
  t->slots[i].value = value;
}

static void table_clear(Table *t) {
  if (t->slots) memset(t->slots, 0, t->cap * sizeof(Entry));
  t->size = 0;
}

typedef struct {
  Table failures;  // hash -> depth that failed
  Table wins;      // hash * 16 + depth -> winning move
} VcfCache;

static int combined_order(const Board *b, int me, int index) {
  int opponent = me == BLACK ? WHITE : BLACK;
  return b->order[me][index] + b->order[opponent][index];
}

static const Board *SORT_BOARD;
static int SORT_ME;

static int by_combined(const void *x, const void *y) {
  int a = *(const int *)x, c = *(const int *)y;
  int ka = combined_order(SORT_BOARD, SORT_ME, a), kc = combined_order(SORT_BOARD, SORT_ME, c);
  if (ka != kc) return ka > kc ? -1 : 1;
  return a - c;
}

static int by_own_order(const void *x, const void *y) {
  int a = *(const int *)x, c = *(const int *)y;
  int ka = SORT_BOARD->order[SORT_ME][a], kc = SORT_BOARD->order[SORT_ME][c];
  if (ka != kc) return ka > kc ? -1 : 1;
  return a - c;
}

static int by_index(const void *x, const void *y) { return *(const int *)x - *(const int *)y; }

// Port of PatternSearch.generate with an unlimited width. Returns the kind.
static int generate(const Board *b, int *moves, int *count) {
  int me = b->turn, opponent = me == BLACK ? WHITE : BLACK;
  const uint8_t *mine = b->flags[me], *theirs = b->flags[opponent];
  int win = -1, nblocks = 0, nforcing = 0, nthreats = 0, best_force = -1;
  int blocks[ACTIONS];
  for (int k = 0; k < b->ncandidates; k++) {
    int i = b->candidates[k];
    if (mine[i] & WIN_FLAG && (win < 0 || i < win)) win = i;
    if (theirs[i] & WIN_FLAG) blocks[nblocks++] = i;
    if (mine[i] & FORCE_FLAG) {
      nforcing++;
      if (best_force < 0 || b->order[me][i] > b->order[me][best_force] ||
          (b->order[me][i] == b->order[me][best_force] && i < best_force))
        best_force = i;
    }
    if (theirs[i] & FORCE_FLAG) nthreats++;
  }
  if (win >= 0) {
    moves[0] = win;
    *count = 1;
    return WIN;
  }
  if (nblocks >= 2) {
    qsort(blocks, nblocks, sizeof(int), by_index);
    memcpy(moves, blocks, sizeof(int) * nblocks);
    *count = nblocks;
    return LOSS;
  }
  if (nblocks == 1) {
    moves[0] = blocks[0];
    *count = 1;
    return BLOCK;
  }
  if (nforcing) {
    moves[0] = best_force;
    *count = 1;
    return FORCE_WIN;
  }
  int n = 0;
  if (nthreats) {
    for (int k = 0; k < b->ncandidates; k++) {
      int i = b->candidates[k];
      if (theirs[i] & FOUR_FLAG || mine[i] & FOUR_FLAG) moves[n++] = i;
    }
  } else {
    for (int k = 0; k < b->ncandidates; k++) moves[n++] = b->candidates[k];
  }
  SORT_BOARD = b;
  SORT_ME = me;
  qsort(moves, n, sizeof(int), by_combined);
  *count = n;
  return nthreats ? DEFEND : QUIET;
}

// Port of PatternSearch.vcf. Returns the first winning move, or -1.
static int vcf(Board *b, VcfCache *cache, int depth) {
  int me = b->turn, opponent = me == BLACK ? WHITE : BLACK;
  const uint8_t *mine = b->flags[me], *theirs = b->flags[opponent];
  int win = -1, nblocks = 0, block = -1, nfours = 0, fours[ACTIONS];
  for (int k = 0; k < b->ncandidates; k++) {
    int i = b->candidates[k];
    if (mine[i] & WIN_FLAG && (win < 0 || i < win)) win = i;
    if (theirs[i] & WIN_FLAG) {
      nblocks++;
      block = i;
    }
  }
  if (win >= 0) return win;
  if (depth <= 0) return -1;
  int value;
  if (table_get(&cache->failures, b->hash, &value) && value >= depth) return -1;
  if (table_get(&cache->wins, b->hash * 16 + (uint64_t)depth, &value)) return value;
  if (nblocks >= 2) return -1;
  for (int k = 0; k < b->ncandidates; k++) {
    int i = b->candidates[k];
    if (mine[i] & FOUR_FLAG && (nblocks == 0 || i == block)) fours[nfours++] = i;
  }
  SORT_BOARD = b;
  SORT_ME = me;
  qsort(fours, nfours, sizeof(int), by_own_order);
  for (int f = 0; f < nfours; f++) {
    int move = fours[f], replies[44], nreplies = 0;
    board_play(b, move);
    // No winning cells existed before this four; only its own lines changed.
    for (int d = 0; d < 4; d++)
      for (int k = 0; k < LINE_COUNT[move][d]; k++) {
        int cell = LINE_CELLS[move][d][k], seen = 0;
        if (b->cells[cell] != EMPTY || !(b->flags[me][cell] & WIN_FLAG)) continue;
        for (int r = 0; r < nreplies; r++) seen |= replies[r] == cell;
        if (!seen) replies[nreplies++] = cell;
      }
    int proved = nreplies >= 2;
    if (nreplies == 1) {
      board_play(b, replies[0]);
      proved = vcf(b, cache, depth - 1) >= 0;
      board_undo(b);
    }
    board_undo(b);
    if (proved) {
      table_put(&cache->wins, b->hash * 16 + (uint64_t)depth, move);
      return move;
    }
  }
  table_put(&cache->failures, b->hash, depth);
  return -1;
}

// ----- search tree -------------------------------------------------------------

typedef struct {
  double prior, value_sum, terminal;
  int visits, first_child, nchildren, move, pending, has_terminal;
} Node;

typedef struct {
  int path[ACTIONS + 1], length, leaf, kind, nmoves, moves[ACTIONS];
} Pending;

typedef struct {
  Board board;
  VcfCache vcf;
  Node *nodes;
  int nnodes, node_cap;
  double c_puct, virtual_loss;
  int top_k, leaf_vcf_depth, planes;
  int root_moves;  // board.nmoves at the root
  Pending *pending;
  int npending, pending_cap;
  int root_needs_eval, root_kind, root_nmoves, root_move_list[ACTIONS];
  int simulations;
  long vcf_limit;
} Search;

static int new_nodes(Search *s, int count) {
  if (s->nnodes + count > s->node_cap) {
    while (s->nnodes + count > s->node_cap) s->node_cap = s->node_cap ? s->node_cap * 2 : 1 << 14;
    s->nodes = realloc(s->nodes, sizeof(Node) * s->node_cap);
  }
  int first = s->nnodes;
  s->nnodes += count;
  memset(&s->nodes[first], 0, sizeof(Node) * count);
  return first;
}

static double node_q(const Node *n) { return n->visits ? n->value_sum / n->visits : 0.0; }

static void set_children(Search *s, int node, const int *moves, const double *priors, int count) {
  int first = new_nodes(s, count);
  for (int k = 0; k < count; k++) {
    s->nodes[first + k].move = moves[k];
    s->nodes[first + k].prior = priors[k];
  }
  s->nodes[node].first_child = first;
  s->nodes[node].nchildren = count;
}

static void set_terminal(Search *s, int node, double value, int move) {
  double one = 1.0;
  s->nodes[node].terminal = value;
  s->nodes[node].has_terminal = 1;
  if (move >= 0) set_children(s, node, &move, &one, 1);
}

// Tactics part of GuidedMCTS._expand_tactics. Returns 1 and the value if
// resolved; else 0 and leaves kind/moves for the network.
static int expand_tactics(Search *s, int node, double *value, int *kind, int *moves, int *nmoves) {
  Board *b = &s->board;
  if (b->nmoves == ACTIONS) {
    set_terminal(s, node, 0.0, -1);
    *value = 0.0;
    return 1;
  }
  *kind = generate(b, moves, nmoves);
  if (*kind == WIN || *kind == FORCE_WIN) {
    set_terminal(s, node, 1.0, moves[0]);
    *value = 1.0;
    return 1;
  }
  if (*kind == LOSS) {
    set_terminal(s, node, -1.0, moves[0]);
    *value = -1.0;
    return 1;
  }
  if (*kind == QUIET && s->leaf_vcf_depth) {
    int win = vcf(b, &s->vcf, s->leaf_vcf_depth);
    if (win >= 0) {
      set_terminal(s, node, 1.0, win);
      *value = 1.0;
      return 1;
    }
  }
  return 0;
}

static const float *SORT_LOGITS;

typedef struct {
  int move, position;
} Ranked;

static int by_logit(const void *x, const void *y) {
  const Ranked *a = x, *c = y;
  float la = SORT_LOGITS[INDEX_TO_ACTION[a->move]], lc = SORT_LOGITS[INDEX_TO_ACTION[c->move]];
  if (la != lc) return la > lc ? -1 : 1;
  return a->position - c->position;  // stable, as in Python
}

static double attach(Search *s, int node, int kind, int *moves, int nmoves, const float *logits, float value) {
  int count = nmoves;
  if (kind == QUIET) {
    Ranked ranked[ACTIONS];
    for (int k = 0; k < nmoves; k++) {
      ranked[k].move = moves[k];
      ranked[k].position = k;
    }
    SORT_LOGITS = logits;
    qsort(ranked, nmoves, sizeof(Ranked), by_logit);
    count = nmoves < s->top_k ? nmoves : s->top_k;
    for (int k = 0; k < count; k++) moves[k] = ranked[k].move;
  }
  double priors[ACTIONS], max = -INFINITY, total = 0.0;
  for (int k = 0; k < count; k++) {
    double l = logits[INDEX_TO_ACTION[moves[k]]];
    if (l > max) max = l;
  }
  for (int k = 0; k < count; k++) total += priors[k] = exp((double)logits[INDEX_TO_ACTION[moves[k]]] - max);
  for (int k = 0; k < count; k++) priors[k] /= total;
  set_children(s, node, moves, priors, count);
  return value;
}

static int select_child(Search *s, int node) {
  Node *n = &s->nodes[node];
  double root_visits = sqrt(n->visits > 1 ? n->visits : 1), fpu = -node_q(n) - 0.2, best_score = -INFINITY;
  int best = -1;
  for (int k = 0; k < n->nchildren; k++) {
    Node *child = &s->nodes[n->first_child + k];
    double q = child->visits ? -node_q(child) : fpu;
    double score = q + s->c_puct * child->prior * root_visits / (1 + child->visits);
    if (score > best_score) {
      best_score = score;
      best = n->first_child + k;
    }
  }
  return best;
}

static void backup(Search *s, const int *path, int length, double value) {
  for (int i = length - 1; i >= 0; i--) {
    s->nodes[path[i]].visits++;
    s->nodes[path[i]].value_sum += value;
    value = -value;
  }
}

static void encode(const Board *b, float *out, int planes) {
  int player = b->turn;
  float black_to_move = player == BLACK ? 1.0f : 0.0f;
  for (int a = 0; a < ACTIONS; a++) {
    int cell = b->cells[ACTION_TO_INDEX[a]];
    float *f = out + a * planes;
    f[0] = cell == player ? 1.0f : 0.0f;
    f[1] = cell != EMPTY && cell != player ? 1.0f : 0.0f;
    f[2] = black_to_move;
    if (planes == 4) f[3] = 1.0f;
  }
}

// ----- exported API ----------------------------------------------------------------

Search *cf_new(double c_puct, int top_k, int leaf_vcf_depth, double virtual_loss, int planes) {
  init_tables();
  Search *s = calloc(1, sizeof(Search));
  s->c_puct = c_puct;
  s->top_k = top_k;
  s->leaf_vcf_depth = leaf_vcf_depth;
  s->virtual_loss = virtual_loss;
  s->planes = planes;
  s->vcf_limit = 2000000;
  board_init(&s->board);
  return s;
}

void cf_free(Search *s) {
  free(s->board.log);
  free(s->nodes);
  free(s->pending);
  free(s->vcf.failures.slots);
  free(s->vcf.wins.slots);
  free(s);
}

void cf_clear_vcf_cache(Search *s) {
  table_clear(&s->vcf.failures);
  table_clear(&s->vcf.wins);
}

// Set the position (moves in play order) and expand the root.
// Returns: 0 needs the network for the root (features in `features`),
//          1 empty board (center), 2 forced/terminal root, 3 ready to search.
int cf_begin(Search *s, const int *actions, int count, float *features, uint64_t *hash) {
  if ((long)(s->vcf.failures.size + s->vcf.wins.size) > s->vcf_limit) cf_clear_vcf_cache(s);
  board_init(&s->board);
  for (int k = 0; k < count; k++) board_play(&s->board, ACTION_TO_INDEX[actions[k]]);
  s->root_moves = count;
  s->nnodes = 0;
  s->npending = 0;
  s->simulations = 0;
  s->root_needs_eval = 0;
  int root = new_nodes(s, 1);
  s->nodes[root].prior = 1.0;
  if (!s->board.ncandidates) {
    double one = 1.0;
    int center = ACTION_TO_INDEX[7 * SIZE + 7];
    set_children(s, root, &center, &one, 1);
    s->nodes[s->nodes[root].first_child].visits = 1;
    return 1;
  }
  double value;
  s->nodes[root].visits = 1;
  if (expand_tactics(s, root, &value, &s->root_kind, s->root_move_list, &s->root_nmoves)) {
    s->nodes[root].value_sum = value;
    return 2;
  }
  s->root_needs_eval = 1;
  encode(&s->board, features, s->planes);
  *hash = s->board.hash;
  return 0;
}

// Attach the root's network output. Returns 2 if the root is forced, else 3.
int cf_root_eval(Search *s, const float *logits, float value) {
  s->root_needs_eval = 0;
  s->nodes[0].value_sum = attach(s, 0, s->root_kind, s->root_move_list, s->root_nmoves, logits, value);
  return s->nodes[0].nchildren == 1 ? 2 : 3;
}

// Number of root children and their actions/priors (for root noise).
int cf_root_children(Search *s, int *actions, double *priors) {
  Node *root = &s->nodes[0];
  for (int k = 0; k < root->nchildren; k++) {
    actions[k] = INDEX_TO_ACTION[s->nodes[root->first_child + k].move];
    if (priors) priors[k] = s->nodes[root->first_child + k].prior;
  }
  return root->nchildren;
}

void cf_set_root_priors(Search *s, const double *priors) {
  Node *root = &s->nodes[0];
  for (int k = 0; k < root->nchildren; k++) s->nodes[root->first_child + k].prior = priors[k];
}

// Pick up to `size` leaves. Tactical leaves are backed up immediately; leaves
// needing the network get virtual loss and their features written to
// `features` (one block of 225 * planes floats each). Returns the number of
// network requests; `completed` receives the simulations finished meanwhile.
int cf_pick(Search *s, int size, float *features, uint64_t *hashes, int *completed) {
  Board *b = &s->board;
  *completed = 0;
  s->npending = 0;
  if (s->pending_cap < size) {
    s->pending_cap = size;
    s->pending = realloc(s->pending, sizeof(Pending) * size);
  }
  for (int pick = 0; pick < size; pick++) {
    Pending *p = &s->pending[s->npending];
    int node = 0, played = 0;
    p->length = 0;
    p->path[p->length++] = 0;
    while (s->nodes[node].nchildren && !s->nodes[node].has_terminal && !s->nodes[node].pending) {
      node = select_child(s, node);
      board_play(b, s->nodes[node].move);
      played++;
      p->path[p->length++] = node;
    }
    if (s->nodes[node].pending) {
      while (played--) board_undo(b);
      break;
    }
    double value = 0.0;
    int resolved = 1;
    if (s->nodes[node].has_terminal) value = s->nodes[node].terminal;
    else {
      resolved = expand_tactics(s, node, &value, &p->kind, p->moves, &p->nmoves);
      if (!resolved) {
        encode(b, features + (size_t)s->npending * ACTIONS * s->planes, s->planes);
        hashes[s->npending] = b->hash;
      }
    }
    while (played--) board_undo(b);
    if (resolved) {
      backup(s, p->path, p->length, value);
      s->simulations++;
      (*completed)++;
      continue;
    }
    p->leaf = node;
    s->nodes[node].pending = 1;
    for (int i = 0; i < p->length; i++) {
      s->nodes[p->path[i]].visits++;
      if (i) s->nodes[p->path[i]].value_sum += s->virtual_loss;
    }
    s->npending++;
  }
  return s->npending;
}

// Network results for the leaves of the last cf_pick, in order.
void cf_apply(Search *s, const float *logits, const float *values) {
  for (int k = 0; k < s->npending; k++) {
    Pending *p = &s->pending[k];
    for (int i = 0; i < p->length; i++) {
      s->nodes[p->path[i]].visits--;
      if (i) s->nodes[p->path[i]].value_sum -= s->virtual_loss;
    }
    s->nodes[p->leaf].pending = 0;
    double value = attach(s, p->leaf, p->kind, p->moves, p->nmoves, logits + (size_t)k * ACTIONS, values[k]);
    backup(s, p->path, p->length, value);
    s->simulations++;
  }
  s->npending = 0;
}

// Root statistics: visits per root child (actions in child order) and root q.
int cf_result(Search *s, int *actions, int *visits, double *root_q, int *simulations) {
  Node *root = &s->nodes[0];
  for (int k = 0; k < root->nchildren; k++) {
    actions[k] = INDEX_TO_ACTION[s->nodes[root->first_child + k].move];
    visits[k] = s->nodes[root->first_child + k].visits;
  }
  *root_q = node_q(root);
  *simulations = s->simulations;
  return root->nchildren;
}
