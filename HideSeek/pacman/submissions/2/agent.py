import sys
import heapq
import random
from pathlib import Path
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

#bfs helper function to calculate distances from a starting position
def _bfs_distances(map_state, start):
    dist = {start: 0}
    queue = deque([start])
    height, width = map_state.shape
    while queue:
        pos = queue.popleft()
        for move in (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT):
            dr, dc = move.value
            npos = (pos[0] + dr, pos[1] + dc)
            r, c = npos
            if 0 <= r < height and 0 <= c < width and map_state[r, c] == 0 and npos not in dist:
                dist[npos] = dist[pos] + 1
                queue.append(npos)
    return dist

class PacmanAgent(BasePacmanAgent):
    """
    Seeker agent using plain A* search (Manhattan-distance heuristic).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A* Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            # No information at all — explore randomly
            return self._explore(my_position, map_state)

        if target == my_position:
            return (Move.STAY, 1)

        path = self._a_star(my_position, target, map_state)

        if not path:
            # No path found (shouldn't happen on a connected map)
            return self._explore(my_position, map_state)

        first_move = path[0]

        run_length = 1
        for m in path[1:self.pacman_speed]:
            if m != first_move:
                break
            run_length += 1

        steps = self._max_valid_steps(my_position, first_move, map_state, run_length)

        return (first_move, steps)

    def _a_star(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        Standard A* search with a Manhattan-distance heuristic.
        """
        if start == goal:
            return []

        def h(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        open_heap = []
        counter = 0  # tie-breaker so heap tuples are always comparable
        heapq.heappush(open_heap, (h(start), counter, start, []))
        best_g = {start: 0}

        while open_heap:
            _, _, current, path = heapq.heappop(open_heap)

            if current == goal:
                return path

            g = best_g[current]

            for move in MOVES:
                dr, dc = move.value
                nxt = (current[0] + dr, current[1] + dc)

                if not self._is_valid_position(nxt, map_state):
                    continue

                new_g = g + 1
                if nxt not in best_g or new_g < best_g[nxt]:
                    best_g[nxt] = new_g
                    counter += 1
                    heapq.heappush(open_heap,
                                   (new_g + h(nxt), counter, nxt, path + [move]))

        return []

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = list(MOVES)
        random.shuffle(all_moves)

        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)

        return (Move.STAY, 1)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray,
                          desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            nxt = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(nxt, map_state):
                break
            steps += 1
            current = nxt
        return steps


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Improved Simple Evasion Ghost"
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            return self._random_move(my_position, map_state)
        
        best_move = self._improved_maximize_distance(my_position, threat, map_state)
        return best_move if best_move else Move.STAY
    
    def _improved_maximize_distance(self, my_pos: tuple, threat_pos: tuple, map_state: np.ndarray) -> Move:
        """
        Strategy: chooses move that maximizes TRUE path distance from the threat with a
        mobility tiebreaker to avoid dead ends.
        """
        # BFS once from the threat's position gives real distance to every
        # reachable cell, including my current neighbors.
        dist_map = _bfs_distances(map_state, threat_pos)
 
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        candidates = []
 
        for i, move in enumerate(moves):
            if move == Move.STAY:
                new_pos = my_pos
            else:
                delta_row, delta_col = move.value
                new_pos = (my_pos[0] + delta_row, my_pos[1] + delta_col)

            #delta_row, delta_col = move.value
            #new_pos = (my_pos[0] + delta_row, my_pos[1] + delta_col)
 
            if not self._is_valid_position(new_pos, map_state):
                continue

            distance = dist_map.get(new_pos, len(dist_map) + 1)
            open_spaces = self._count_open_neighbors(new_pos, map_state)
            #candidates.append((-distance, -open_spaces, i, move, new_pos))
            candidates.append((-1 * (distance * 10 + open_spaces), i, move, new_pos))  # Higher score is better
 
        if not candidates:
            return None
 
        candidates.sort()
        return candidates[0][2]
    
    def _count_open_neighbors(self, pos: tuple, map_state: np.ndarray) -> int:
        """Count how many adjacent cells are open (not walls)."""
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            neighbor_pos = (pos[0] + delta_row, pos[1] + delta_col)
            if self._is_valid_position(neighbor_pos, map_state):
                count += 1
        return count
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _random_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        import random
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move
        
        return Move.STAY
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0