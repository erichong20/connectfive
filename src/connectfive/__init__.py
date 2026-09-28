"""JAX-native Connect Five environment."""

from connectfive.agents import Agent, RandomAgent, TacticalAgent, make_agent
from connectfive.env import BOARD_SIZE, NUM_ACTIONS, ConnectFive, State
from connectfive.match import GameRecord, MatchSummary, evaluate_agents, play_game

__all__ = [
    "BOARD_SIZE",
    "NUM_ACTIONS",
    "Agent",
    "ConnectFive",
    "GameRecord",
    "MatchSummary",
    "RandomAgent",
    "State",
    "TacticalAgent",
    "evaluate_agents",
    "make_agent",
    "play_game",
]
__version__ = "0.1.0"
