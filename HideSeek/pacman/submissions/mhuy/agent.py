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
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
from collections import deque
import helper


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost

    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need

        self.map_size = (21, 21)
        self.global_map = np.full(self.map_size, -1)
        self.belief_state = np.ones(self.map_size) / (21 * 21)
        self.position_history = deque(maxlen=10)

        self.current_target = None
        self.last_move = None
        self.name = "Template Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.

        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)

        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        self.map_size = map_state.shape

        # 1. Update memory state cleanly
        visible = map_state != -1
        self.global_map[visible] = map_state[visible]

        # 2. Sync beliefs
        self._sync_belief_state(map_state, my_position)
        self.position_history.append(my_position)

        # 3. BULLETPROOF VISION & DISTANCE CHECK
        # is_ghost_visible = (
        #         enemy_position is not None
        #         and len(enemy_position) >= 2
        #         and enemy_position != (-1, -1)
        #         and enemy_position != ()
        # )

        if enemy_position:
            # Calculate the raw distance to the ghost
            dist = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])

            if dist <= 4:
                # CLOSE COMBAT: Use Matrix to intercept and prevent juking
                self.current_target = None
                action = self._matrix_search(my_position, enemy_position)
                print("Matrix")
                self.last_move = action[0]
                return action
            else:
                self.belief_state.fill(0.0)
                self.belief_state[enemy_position[0], enemy_position[1]] = 1.0
                self.current_target = enemy_position

                action = self._blind_search(my_position)
                print("Blind")
                self.last_move = action[0]
                return action
        else:
            # GHOST HIDDEN: Standard exploration sweep
            action = self._blind_search(my_position)
            print("Blind")
            self.last_move = action[0]
            return action

    # Helper methods (you can add more)
    def _sync_belief_state(self, map_state, my_pos):
        """Clears vision strictly using the environment mask, normalizes, and diffuses."""

        # 1. Environment-Driven Vision Clearing[cite: 1]
        visible_mask = map_state != -1
        self.belief_state[visible_mask] = 0.0

        # 2. Normalize BEFORE diffusion
        total = self.belief_state.sum()
        if total > 0:
            self.belief_state /= total
        else:
            self.belief_state = np.ones((21, 21)) / (21 * 21)
            self.belief_state[visible_mask] = 0.0
            if self.belief_state.sum() > 0:
                self.belief_state /= self.belief_state.sum()

        # 3. Diffuse probabilities outward
        new_belief = np.zeros_like(self.belief_state)
        for r in range(21):
            for c in range(21):
                if self.belief_state[r, c] > 0 and self.global_map[r, c] != 1:
                    passable_neighbors = []
                    for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        nr, nc = r + m.value[0], c + m.value[1]
                        if self.global_map[nr, nc] == -1:
                            passable_neighbors.append((nr, nc))

                    if passable_neighbors:
                        prob_share = self.belief_state[r, c] / len(passable_neighbors)
                        for nr, nc in passable_neighbors:
                            new_belief[nr, nc] += prob_share
                    else:
                        new_belief[r, c] += self.belief_state[r, c]

        self.belief_state = new_belief

    def _matrix_search(self, my_position, ghost_position):
        """Predicts simultaneous actions using minimax and directional momentum tiebreakers."""
        best_move = Move.STAY
        best_steps = 1
        min_max_dist = float('inf')
        best_tiebreaker_val = float('-inf')

        seek_actions = self._seek_actions(my_position)
        hide_action = self._hide_action(ghost_position)

        for p_move, p_steps, p_dest in seek_actions:
            max_dist_for_this_move = -1

            for g_dest in hide_action:
                dist = abs(p_dest[0] - g_dest[0]) + abs(p_dest[1] - g_dest[1])
                if dist > max_dist_for_this_move:
                    max_dist_for_this_move = dist

            is_reversal = False
            if self.last_move is not None:
                if (p_move == Move.UP and self.last_move == Move.DOWN) or \
                        (p_move == Move.DOWN and self.last_move == Move.UP) or \
                        (p_move == Move.LEFT and self.last_move == Move.RIGHT) or \
                        (p_move == Move.RIGHT and self.last_move == Move.LEFT):
                    is_reversal = True

            tiebreaker_val = (0 if is_reversal else 10) + (5 if p_move == self.last_move else 0) + p_steps

            if max_dist_for_this_move < min_max_dist:
                min_max_dist = max_dist_for_this_move
                best_move = p_move
                best_steps = p_steps
                best_tiebreaker_val = tiebreaker_val
            elif max_dist_for_this_move == min_max_dist:
                if tiebreaker_val > best_tiebreaker_val:
                    best_tiebreaker_val = tiebreaker_val
                    best_move = p_move
                    best_steps = p_steps

        return best_move, best_steps

    def _blind_search(self, my_position):
        distances, parents = self._get_safe_bfs_tree(my_position)

        # Replan strictly if target is missing, reached, unreachable, or completely empty
        if (self.current_target is None or
                my_position == self.current_target or
                self.current_target not in distances or
                self.belief_state[self.current_target] == 0.0):

            best_target = None
            best_score = -1

            # Unified Target Selection: Seamlessly blends Ghost-Hunting and Fog-Exploration
            for r in range(21):
                for c in range(21):
                    if (r, c) not in distances:
                        continue

                    dist = distances[(r, c)]
                    if dist == 0:
                        continue

                    prob = self.belief_state[r, c]

                    # Frontier Bonus: Check if this safe cell borders unobserved fog (-1)
                    has_fog_neighbor = False
                    for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                        nr, nc = r + m.value[0], c + m.value[1]
                        if 0 <= nr < 21 and 0 <= nc < 21 and self.global_map[nr, nc] == -1:
                            has_fog_neighbor = True
                            break
                    
                    if has_fog_neighbor:
                        prob += 0.5  # Encourages pathing toward dark corners/corridors

                    if prob > 0:
                        score = prob / (1.0 + dist * 0.5)

                        # Strict Directional Inertia: Mathematically forbids 180-degree flip-flops
                        if self.last_move is not None:
                            dr, dc = r - my_position[0], c - my_position[1]
                            is_behind = False
                            if self.last_move == Move.UP and dr > 0:
                                is_behind = True
                            elif self.last_move == Move.DOWN and dr < 0:
                                is_behind = True
                            elif self.last_move == Move.LEFT and dc > 0:
                                is_behind = True
                            elif self.last_move == Move.RIGHT and dc < 0:
                                is_behind = True

                            if is_behind:
                                score *= 0.0001  # Crushes the score of tiles behind Pacman

                        if score > best_score:
                            best_score = score
                            best_target = (r, c)

            # Fallback path mechanism
            if best_target is None:
                if len(distances) > 1:
                    best_target = max([k for k in distances.keys() if k != my_position],
                                      key=lambda k: distances[k])
                else:
                    best_target = my_position

            self.current_target = best_target

        # Reconstruct path
        path = []
        curr = self.current_target
        while curr in parents and parents[curr] is not None:
            prev, move = parents[curr]
            path.append(move)
            curr = prev
        path.reverse()

        if not path:
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = my_position[0] + m.value[0], my_position[1] + m.value[1]
                if self._is_confirmed_safe((nr, nc)):
                    return m, 1
            return Move.STAY, 1

        # SAFE SPEED BURST: Only double-step if the landing zone is verified safe.
        first_move = path[0]
        steps = 1
        if self.pacman_speed >= 2 and len(path) >= 2:
            if path[1] == first_move:
                land_r = my_position[0] + first_move.value[0] * 2
                land_c = my_position[1] + first_move.value[1] * 2
                if 0 <= land_r < 21 and 0 <= land_c < 21:
                    if self.global_map[land_r, land_c] == 0:
                        steps = 2

        return first_move, steps

    def _get_safe_bfs_tree(self, start):
        distances = {start: 0}
        parents = {start: None}
        queue = deque([start])

        while queue:
            curr = queue.popleft()
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (curr[0] + m.value[0], curr[1] + m.value[1])

                if nxt not in distances:
                    if self._is_confirmed_safe(nxt):
                        distances[nxt] = distances[curr] + 1
                        parents[nxt] = (curr, m)
                        queue.append(nxt)
        return distances, parents

    def _seek_actions(self, my_pos):
        pacman_options = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            p1 = (my_pos[0] + m.value[0], my_pos[1] + m.value[1])
            if self._is_confirmed_safe(p1):
                pacman_options.append((m, 1, p1))
                p2 = (p1[0] + m.value[0], p1[1] + m.value[1])
                if self._is_confirmed_safe(p2):
                    pacman_options.append((m, 2, p2))
        return pacman_options

    def _hide_action(self, ghost_pos):
        ghost_options = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            gp = (ghost_pos[0] + m.value[0], ghost_pos[1] + m.value[1])
            if 0 <= gp[0] < 21 and 0 <= gp[1] < 21 and self.global_map[gp] == 0:
                ghost_options.append(gp)
        return ghost_options

    def _is_confirmed_safe(self, pos):
        return (
            0 <= pos[0] < 21
            and 0 <= pos[1] < 21
            and self.global_map[pos] == 0
        )


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.max_depth = 4
        self.pacman_speed = 2
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos

        # --- 1. OPENING BIAS: Prioritize UP and RIGHT for the first 15 steps ---
        if step_number <= 15 and enemy_position is None:
            preferred_moves = [Move.UP, Move.RIGHT, Move.DOWN, Move.LEFT, Move.STAY]
            
            # Find the first preferred move that is valid and safe
            for move in preferred_moves:
                delta_row, delta_col = move.value
                next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
                if 0 <= next_pos[0] < 21 and 0 <= next_pos[1] < 21 and map_state[next_pos] == 0:
                    return move

        # 2. CLEAR VISION: Pacman is visible, use Minimax / adversarial search
        if enemy_position is not None:
            v = float('inf')
            best_move = Move.STAY
            alpha = -float('inf')
            beta = float('inf')

            for action in self._hide_action(my_position, map_state):
                move_enum, step_val = action
                next_hide_pos = self._result(my_position, action)
                score = self.max_value(next_hide_pos, enemy_position, step_number + 1, map_state, 1, alpha, beta)

                if score < v:
                    v = score
                    best_move = move_enum 

            return best_move if best_move is not None else Move.STAY

        # 3. BLIND MODE / SAFE EXPLORATION: Pacman is hidden in fog
        else:
            best_move = Move.STAY
            max_score = -float('inf')

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                delta_row, delta_col = move.value
                next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

                if 0 <= next_pos[0] < 21 and 0 <= next_pos[1] < 21 and map_state[next_pos] == 0:
                    score = 0
                    
                    if threat is not None:
                        dist_to_threat = abs(next_pos[0] - threat[0]) + abs(next_pos[1] - threat[1])
                        score += dist_to_threat * 10

                    open_neighbors = 0
                    for m in [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]:
                        nr, nc = next_pos[0] + m.value[0], next_pos[1] + m.value[1]
                        if 0 <= nr < 21 and 0 <= nc < 21 and map_state[nr, nc] == 0:
                            open_neighbors += 1
                    
                    score += open_neighbors * 5

                    if score > max_score:
                        max_score = score
                        best_move = move

            return best_move
    
    # Helper methods (you can add more)
    # def _find_shadow_move(self, my_position, threat_pos, map_state):
    #     best_move = Move.STAY
    #     max_score = -float('inf')

    #     for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
    #         delta_row, delta_col = move.value
    #         next_pos = (my_position[0] + delta_row, my_position[1] + delta_col)

    #         # Check if valid map position and not a wall
    #         if 0 <= next_pos[0] < 21 and 0 <= next_pos[1] < 21 and map_state[next_pos] == 0:
    #             dist = abs(next_pos[0] - threat_pos[0]) + abs(next_pos[1] - threat_pos[1])
                
    #             # Check if this move puts a wall or corner between Ghost and Pacman
    #             # (i.e., they are no longer aligned on a clear cross-hair ray)
    #             is_shaded = False
    #             if next_pos[0] != threat_pos[0] and next_pos[1] != threat_pos[1]:
    #                 is_shaded = True # Diagonals or offset positions break the cardinal cross ray
                
    #             # Scoring: Prioritize distance, give a massive bonus for being "shaded" behind a corner
    #             score = dist + (50 if is_shaded else 0)

    #             if score > max_score:
    #                 max_score = score
    #                 best_move = move

    #     return best_move
    
    def _evaluation_heuristic(self, my_position, enemy_position, map_state):
        # --- FIX 2: O(1) Lookup instead of O(BFS) calculation ---
        # Return negative distance because we want to minimize distance to ghost (or maximize -dist)
        return -self.dist_map[my_position[0], my_position[1]]

    # Return a list of every possible moves for ghost
    def _hide_action(self, my_position, map_state: np.ndarray):
        actions = [(Move.STAY, 0)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                actions.append((move, 1)) # Ghost always moves with 1 step
        return actions

    # Return a list of every possible moves for pacman
    def _seek_actions(self, pos, map_state):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            current = pos
            for steps in range(1, self.pacman_speed + 1):
                delta_row, delta_col = move.value
                next_pos = (current[0] + delta_row, current[1] + delta_col)
                if not self._is_valid_position(next_pos, map_state):
                    break
                actions.append((move, steps))
                current = next_pos
        return actions

    # Check if the game ends
    def _terminal(self, my_position, enemy_position, steps):
        if helper._manhattan(my_position, enemy_position) < 2:
            return True
        if steps > 200:
            return True
        return False

    # The transition model, but this time my_position is the ghost.
    def _result(self, my_position, action):
        move, step = action
        delta_row, delta_col = move.value
        return (my_position[0] + delta_row * step, my_position[1] + delta_col * step)

    # Gives out score (Same with pacman)
    def _utility(self, my_position, enemy_position, steps, depth):
        if helper._manhattan(my_position, enemy_position) < 2:
            return 1000 - depth
        return -1000

    # Same with pacman, but this time my_position is the ghost.
    def max_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):

        # If any transition model reaches the terminal state, gives out the scores
        if self._terminal(my_position, enemy_position, steps):
            return self._utility(my_position, enemy_position, steps, depth)

        # If depth is reached before terminal, calculate heuristically instead
        if depth > self.max_depth:
            return helper._evaluation_heuristic(my_position, enemy_position, map_state)

        v = -float('inf')
        for action in self._seek_actions(enemy_position, map_state):
            # Predicting pacman's move
            next_enemy_pos = self._result(enemy_position, action)
            v = max(v, self.min_value(my_position, next_enemy_pos, steps + 1, map_state, depth + 1, alpha, beta))

            # If we know that this moves is higher than Beta
            # Which means in the next loop, the opponent is already having a better move already.
            # We prune the rest away (Stop the recursive loop).

            if v >= beta: return v
            alpha = max(alpha, v)
        return v

    def min_value(self, my_position, enemy_position, steps, map_state, depth, alpha, beta):
        v = float('inf')
        for action in self._hide_action(my_position, map_state):
            next_hide_pos = self._result(my_position, action)

            # Same with pacman agent's min_value, but this time my_position = ghost's position.
            # We also prevent the guarding situation here, read pacman's agent comments for more details.

            if self._terminal(next_hide_pos, enemy_position, steps + 1):
                score = self._utility(next_hide_pos, enemy_position, steps + 1, depth)
            elif depth > self.max_depth:
                score = helper._evaluation_heuristic(next_hide_pos, enemy_position, map_state)
            else:
                score = self.max_value(next_hide_pos, enemy_position, steps + 1, map_state, depth + 1, alpha, beta)

            v = min(v, score) # Pick best move.

            # If we know that this moves is lower than Alpha
            # Which means in the next loop, the opponent is already having a better move already.
            # We prune the rest away (Stop the recursive loop).

            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0
