"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from enum import Enum
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import helper
import random
from collections import deque
import time

TIME_BUDGET = 0.8  # Time budget in seconds for each agent's step, to stay within this limit

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.max_depth = 4
        self.name = "Template Pacman"
        self.last_known_enemy_pos = None

        self.steps_since_seen = 0
        self.stale_threshold = 15

        self.recent_positions = []

        self.visited_cells = set()
        self._rng = random.Random(1234)

        self._search_start = 0.0

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        
        """
        Updates the memory of the Ghost's position. 
        If the Ghost is unseen, it explores the map. If the Ghost is nearby, 
        it triggers the Alpha-Beta pruning search to hunt it down.
        """

        # Update our memory when we actually see the enemy.
        # If they hide in the fog, we remember their last location for up to the stale_threshold steps.
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        target = enemy_position
        if target is None:
            if self.last_known_enemy_pos is not None and self.steps_since_seen <= self.stale_threshold:
                target = self.last_known_enemy_pos  # Hunt the ghost's last known location
            else:
                self.last_known_enemy_pos = None    # Ghost is completely lost

        self._update_visited(my_position, map_state)

        # If we have absolutely no idea where the enemy is, do not run Minimax.
        # Instead, head towards the nearest unvisited fog cell to expand our vision.
        if target is None:
            frontier_move = helper.nearest_unvisited_move(my_position, map_state, self.visited_cells)
            if frontier_move is not None and self._is_valid_move(my_position, frontier_move, map_state):
                return (frontier_move, 1)

            # If no unvisited cells are found, pick a random valid move.           
            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            self._rng.shuffle(moves)
            self.recent_positions.append(my_position)
            self.recent_positions = self.recent_positions[-4:]
            for move in moves:
                if self._is_valid_move(my_position, move, map_state):
                    next_pos = helper.get_result(my_position, (move, 1))
                    if next_pos not in self.recent_positions:
                        return (move, 1)
            for move in moves:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        self._search_start = time.perf_counter()

        # Initialize necessary variables
        v = -float('inf')      # Keep track of the max score
        move = None            # The action which is bounded to the max score
        alpha = -float('inf')  # Keep track of the maximum node of a level
        beta = float('inf')    # Keep track of the minimum node of a level

        # Evaluate the most promising moves first
        actions = self._seek_actions(my_position, map_state)
        ordered_actions = sorted(actions, key=lambda a: helper._manhattan(helper.get_result(my_position, a), target))

        for action in ordered_actions:
            next_pos = helper.get_result(my_position, action)
            score = self.min_value(next_pos, target, step_number + 1, map_state, 1, alpha, beta)
            if score > v:
                v = score
                move = action
            if self._time_up(): 
                break
        return move if move is not None else (Move.STAY, 1)

    # Helper methods

    def _update_visited(self, my_position, map_state):
        """
        Records the agent's current position and any known safe paths within 
        its line of sight to prevent redundant exploration.
        """
        self.visited_cells.add(my_position)
        rows, cols = np.where(map_state == 0)
        for r, c in zip(rows.tolist(), cols.tolist()):
            self.visited_cells.add((r, c))

    def _seek_actions(self, pos, map_state: np.ndarray):
        """
        Generates all possible valid moves for Pacman, considering its speed and the map's layout.
        Returns a list of (Move, steps) tuples.
        """
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            max_steps = self._max_valid_steps(pos, move, map_state, self.pacman_speed)

            for steps in range(1, max_steps + 1):
                actions.append((move, steps))

        return actions 

    def _hide_action(self, my_position, map_state: np.ndarray):
        """
        Generates 1-step moves. Used by Pacman to simulate and predict 
        what the Ghost might do during the Minimax lookahead.
        """
        ghost_action = [(Move.STAY, 0)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                ghost_action.append((move, 1)) 
        return ghost_action

    def max_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        """
        Represents Pacman's turn, trying to choose the move that 
        maximizes the utility score.
        """
        if helper.is_terminal(my_position, enemy_position, steps):
            return helper.get_utility(my_position, enemy_position, steps, depth)

        # STOPPING CONDITIONS:
        # Reached max depth limit.
        # Reached the strict 0.7s time limit.
        # FOG OF WAR: The simulated move steps into an unknown cell (-1).
        if depth > self.max_depth or self._time_up() or map_state[my_position[0], my_position[1]] == -1:
            return helper._evaluation_heuristic(my_position, enemy_position, map_state)

        v = -float('inf') # Keeps track of the highest score
        actions = self._seek_actions(my_position, map_state)
        ordered_actions = sorted(actions, key=lambda a: helper._manhattan(helper.get_result(my_position, a), enemy_position))

        for action in ordered_actions:
            next_pos = helper.get_result(my_position, action)
            v = max(v, self.min_value(next_pos, enemy_position, steps + 1, map_state, depth + 1, alpha, beta))

            if v >= beta: return v
            alpha = max(alpha, v)
            
            if self._time_up(): return v
            
        return v

    def min_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        """
        Represents the Ghost's turn, simulating the Ghost picking the move that 
        minimizes Pacman's score.
        """
        v = float('inf')
        
        actions = self._hide_action(enemy_position, map_state)
        ordered_actions = sorted(actions, key=lambda a: -helper._manhattan(my_position, helper.get_result(enemy_position, a)))

        for action in ordered_actions:
            next_enemy_pos = helper.get_result(enemy_position, action)

            if helper.is_terminal(my_position, next_enemy_pos, steps + 1):
                score = helper.get_utility(my_position, next_enemy_pos, steps + 1, depth)
            elif depth > self.max_depth or self._time_up() or map_state[next_enemy_pos[0], next_enemy_pos[1]] == -1:
                score = helper._evaluation_heuristic(my_position, next_enemy_pos, map_state)
            else:
                score = self.max_value(my_position, next_enemy_pos, steps + 1, map_state, depth + 1, alpha, beta)

            v = min(v, score)
            if v <= alpha: return v
            beta = min(beta, v)
            
            if self._time_up(): return v
            
        return v

    """
    --------------------------------- ///// -------------------------------------
    """

    # def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
    #     for move in moves:
    #         max_steps = min(self.pacman_speed, max(1, desired_steps))
    #         steps = self._max_valid_steps(pos, move, map_state, max_steps)
    #         if steps > 0:
    #             return (move, steps)
    #     return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        """
        Returns the maximum number of valid steps Pacman can take in the given direction,
        safely stops accumulating steps if it steps into the fog (-1).
        """
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            
            # Stop if we hit an explicitly known wall or go out of bounds
            if not helper.is_valid_position(next_pos, map_state):
                break
                
            steps += 1
            current = next_pos
            
            if map_state[current[0], current[1]] == -1:
                break
                
        return steps

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _time_up(self):
        """Checks if the internal search has exceeded the strict time budget."""
        return (time.perf_counter() - self._search_start) > TIME_BUDGET



class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught by minimizing Pacman's performance metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_depth = 4
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Simultaneous_Minimax_Ghost"
        self.last_known_enemy_pos = None

        self.steps_since_seen = 0
        self.stale_threshold = 15

        self.recent_positions = []
        self.visited_cells = set()
        self._rng = random.Random(5678)

        self._search_start = 0.0

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        
        """
        Explores if safe, but if Pacman 
        is detected, runs Alpha-Beta pruning to find the best escape route.
        """

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

        target = enemy_position
        if target is None:
            if self.last_known_enemy_pos is not None and self.steps_since_seen <= self.stale_threshold:
                target = self.last_known_enemy_pos
            else:
                self.last_known_enemy_pos = None

        self._update_visited(my_position, map_state)

        if target is None:
            frontier_move = helper.nearest_unvisited_move(my_position, map_state, self.visited_cells)
            if frontier_move is not None and self._is_valid_move(my_position, frontier_move, map_state):
                return frontier_move

            moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            self._rng.shuffle(moves)
            self.recent_positions.append(my_position)
            self.recent_positions = self.recent_positions[-4:]
            for move in moves:
                if self._is_valid_move(my_position, move, map_state):
                    next_pos = helper.get_result(my_position, (move, 1))
                    if next_pos not in self.recent_positions:
                        return move
            for move in moves:
                if self._is_valid_move(my_position, move, map_state):
                    return move
            return Move.STAY

        self._search_start = time.perf_counter()

        stay_score = None
        best_move_score = float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')

        actions = self._hide_action(my_position, map_state)
        ordered_actions = sorted(actions, key=lambda a: -helper._manhattan(helper.get_result(my_position, a), target))

        for action in ordered_actions:
            next_hide_pos = helper.get_result(my_position, action)
            score = self.max_value(next_hide_pos, target, step_number + 1, map_state, 1, alpha, beta)
            
            if action[0] == Move.STAY:
                stay_score = score
                continue
            
            if score < best_move_score:
                best_move_score = score
                best_move = action[0]

            if self._time_up():
                break

        if best_move is None:
            return Move.STAY

        STAY_MARGIN = 1.5
        if stay_score is not None and stay_score < best_move_score - STAY_MARGIN:
            return Move.STAY
        return best_move

    def _update_visited(self, my_position, map_state):
        # same as PacmanAgent's _update_visited
        self.visited_cells.add(my_position)
        rows, cols = np.where(map_state == 0)
        for r, c in zip(rows.tolist(), cols.tolist()):
            self.visited_cells.add((r, c))

    def _hide_action(self, my_position, map_state: np.ndarray):
        # same as PacmanAgent's _hide_action
        actions = [(Move.STAY, 1)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                actions.append((move, 1)) 
        return actions

    def _seek_actions(self, pos, map_state):
        # same as PacmanAgent's _seek_actions
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            current = pos
            for steps in range(1, self.pacman_speed + 1):
                delta_row, delta_col = move.value
                next_pos = (current[0] + delta_row, current[1] + delta_col)
                if not helper.is_valid_position(next_pos, map_state):
                    break
                actions.append((move, steps))
                current = next_pos
        return actions

    def max_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        """
        Assumes Pacman will play optimally to maximize its score. 
        """
        if helper.is_terminal(my_position, enemy_position, steps):
            return helper.get_utility(my_position, enemy_position, steps, depth)

        if depth > self.max_depth or self._time_up() or map_state[enemy_position[0], enemy_position[1]] == -1:
            return helper._evaluation_heuristic(my_position, enemy_position, map_state)

        v = -float('inf')
        
        actions = self._seek_actions(enemy_position, map_state)
        ordered_actions = sorted(actions, key=lambda a: helper._manhattan(my_position, helper.get_result(enemy_position, a)))

        for action in ordered_actions:
            next_enemy_pos = helper.get_result(enemy_position, action)
            v = max(v, self.min_value(my_position, next_enemy_pos, steps + 1, map_state, depth + 1, alpha, beta))

            if v >= beta: return v
            alpha = max(alpha, v)

            if self._time_up(): return v

        return v

    def min_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        """
        The Ghost tries to pick actions that minimize the evaluation score.
        """
        v = float('inf')

        # Evaluate the most promising moves first
        actions = self._hide_action(my_position, map_state)
        ordered_actions = sorted(actions, key=lambda a: -helper._manhattan(helper.get_result(my_position, a), enemy_position))

        for action in ordered_actions:
            next_hide_pos = helper.get_result(my_position, action)

            if helper.is_terminal(next_hide_pos, enemy_position, steps + 1):
                score = helper.get_utility(next_hide_pos, enemy_position, steps + 1, depth)
            elif depth > self.max_depth or self._time_up() or map_state[next_hide_pos[0], next_hide_pos[1]] == -1:
                score = helper._evaluation_heuristic(next_hide_pos, enemy_position, map_state)
            else:
                score = self.max_value(next_hide_pos, enemy_position, steps + 1, map_state, depth + 1, alpha, beta)

            v = min(v, score) 

            if v <= alpha: return v
            beta = min(beta, v)

            if self._time_up(): return v

        return v

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        # same as PacmanAgent's _is_valid_move
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return helper.is_valid_position(new_pos, map_state)
    
    def _time_up(self):
        # same as PacmanAgent's _time_up
        return (time.perf_counter() - self._search_start) > TIME_BUDGET
