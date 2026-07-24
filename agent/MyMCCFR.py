from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agent.heuristic_agent import HeuristicPokerAgent
from agent.uct_agent import _EVSimulation
from poker_env import (
    BETTING_RULES_VERSION,
    Card,
    get_best_hand,
    get_public_betting_priority,
)


MCCFR_START_STREETS = ("6th", "7th_hidden")

