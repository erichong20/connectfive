"""JAX-native Connect Five environment."""

from connectfive.agents import Agent, LookaheadAgent, RandomAgent, TacticalAgent, make_agent
from connectfive.env import BOARD_SIZE, NUM_ACTIONS, ConnectFive, State
from connectfive.match import (
    GameRecord,
    MatchSummary,
    MoveDiagnostic,
    evaluate_agents,
    evaluate_agents_with_records,
    generate_opening,
    play_game,
)
from connectfive.search import (
    NegamaxAgent,
    SearchResult,
    SearchStats,
    ThreatSearchAgent,
)

__all__ = [
    "BOARD_SIZE",
    "NUM_ACTIONS",
    "Agent",
    "ConnectFive",
    "GameRecord",
    "LookaheadAgent",
    "MatchSummary",
    "MoveDiagnostic",
    "NegamaxAgent",
    "RandomAgent",
    "SearchResult",
    "SearchStats",
    "State",
    "TacticalAgent",
    "ThreatSearchAgent",
    "evaluate_agents",
    "evaluate_agents_with_records",
    "generate_opening",
    "make_agent",
    "play_game",
]
__version__ = "0.1.0"
