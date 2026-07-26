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

import collections
import sys
from pathlib import Path
import time
from collections import deque
from enum import Enum
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

# Uses for 2 agents
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def bfs_maze_distance(start, goal, map_state):
    if start == goal:
        return 0
    
    queue = deque([(start, 0)])
    visited = {start}
    
    height, width = map_state.shape
    
    while queue:
        curr, dist = queue.popleft()
        
        if manhattan_distance(curr, goal) < 2:
            return dist + 1
            
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr[0] + dr, curr[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                if map_state[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
    return 1000

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.max_depth = 4
        self.name = "AIO's Pacman"
        self.last_known_enemy_pos = None
        self.start_time = 0
        self.TIME_LIMIT = 0.9
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):

        if not hasattr(self, 'known_map') or self.known_map is None:
            self.known_map = np.copy(map_state)

        else:
            visible_cells = (map_state != -1)
            self.known_map[visible_cells] = map_state[visible_cells]

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        self.start_time = time.time()
        target = self.last_known_enemy_pos
        
        if target is None or my_position == target:
            target = self._find_nearest_unexplored(my_position)

            if target is None:
                return (Move.STAY, 1)

            return self._bfs_move_towards(my_position, target, is_pacman=True)

        self.dist_map = self._precompute_distances(target, self.known_map)

        best_val = -float('inf')
        best_action = None
        alpha = -float('inf')
        beta = float('inf')

        actions = self._seek_actions(my_position, self.known_map)
        actions.sort(key=lambda a: self.dist_map[self._result(my_position, a)[0], self._result(my_position, a)[1]])

        for action in actions:
            next_pos = self._result(my_position, action)
            # Kích hoạt não Minimax
            val = self.min_value(next_pos, target, step_number + 1, depth=1, alpha=alpha, beta=beta)
            
            if val > best_val:
                best_val = val
                best_action = action
            
            alpha = max(alpha, best_val)
            
            # Cắt tỉa nếu sắp hết thời gian (dưới 0.9s)
            if time.time() - self.start_time > self.TIME_LIMIT:
                break

        return best_action if best_action else (Move.STAY, 1)
    
    # --- Các hàm Minimax của Pacman ---

    def max_value(self, pacman_pos, ghost_pos, steps, depth, alpha, beta):
        if self._terminal(pacman_pos, ghost_pos, steps):
            return self._utility(pacman_pos, ghost_pos, steps, depth)

        if depth >= self.max_depth or (time.time() - self.start_time > self.TIME_LIMIT):
            return -self.dist_map[pacman_pos[0], pacman_pos[1]]

        v = -float('inf')
        for action in self._seek_actions(pacman_pos, self.known_map):
            next_pac = self._result(pacman_pos, action)
            v = max(v, self.min_value(next_pac, ghost_pos, steps + 1, depth + 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, pacman_pos, ghost_pos, steps, depth, alpha, beta):
        if self._terminal(pacman_pos, ghost_pos, steps):
            return self._utility(pacman_pos, ghost_pos, steps, depth)

        if depth >= self.max_depth or (time.time() - self.start_time > self.TIME_LIMIT):
            return -self.dist_map[pacman_pos[0], pacman_pos[1]]

        v = float('inf')
        for action in self._hide_action(ghost_pos, self.known_map):
            next_ghost = self._result(ghost_pos, action)
            v = min(v, self.max_value(pacman_pos, next_ghost, steps + 1, depth + 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
            
    # Helper methods (you can add more)
    """
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None
    """
    def _seek_actions(self, pos, map_state: np.ndarray):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            max_steps = self._max_valid_steps(pos, move, map_state, self.pacman_speed)
            for steps in range(1, max_steps + 1):
                actions.append((move, steps))
        return actions

    def _hide_action(self, pos, map_state: np.ndarray):
        actions = [(Move.STAY, 0)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(pos, move, map_state):
                actions.append((move, 1))
        return actions

    def _terminal(self, my_position, enemy_position, steps):
        if manhattan_distance(my_position, enemy_position) < 2: return True
        if steps > 200: return True
        return False

    def _result(self, pos, action):
        move, step = action
        dr, dc = move.value
        return (pos[0] + dr * step, pos[1] + dc * step)

    def _utility(self, my_position, enemy_position, steps, depth):
        if manhattan_distance(my_position, enemy_position) < 2:
            return 1000 - depth
        return -1000

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _find_nearest_unexplored(self, start):
        """
        Dùng BFS loang ra để tìm ô '0' (an toàn) gần nhất 
        mà nằm sát vách với một ô '-1' (sương mù).
        """
        rows, cols = self.known_map.shape
        queue = collections.deque([start])
        visited = set([start])
        
        while queue:
            r, c = queue.popleft()
            
            # Kiểm tra 4 hướng xung quanh ô hiện tại
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Nếu thấy sương mù (-1) sát bên, thì ô hiện tại (r, c) là mép rìa an toàn
                    if self.known_map[nr, nc] == -1:
                        return (r, c) 
            
            # Nếu chưa thấy sương mù, tiếp tục loang ra các ô an toàn (0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if self.known_map[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return None # Đã khám phá hết map

    def _bfs_move_towards(self, start, target, is_pacman=True):
        """
        Đi từng bước một hướng về target dựa trên mảng khoảng cách BFS.
        """
        if start == target or target is None:
            return (Move.STAY, 1) if is_pacman else Move.STAY
            
        # Gọi lại hàm precompute thần thánh để đo khoảng cách từ target tới mọi nơi
        dist_map = self._precompute_distances(target, self.known_map)
        
        best_move = Move.STAY
        best_dist = float('inf')
        
        # Thử 4 hướng đi từ vị trí hiện tại
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(start, move, self.known_map):
                delta_row, delta_col = move.value
                nr, nc = start[0] + delta_row, start[1] + delta_col
                
                # Chọn bước đi làm khoảng cách tới target giảm nhiều nhất
                if dist_map[nr, nc] < best_dist:
                    best_dist = dist_map[nr, nc]
                    best_move = move
                    
        return (best_move, 1) if is_pacman else best_move

    def _bfs_run_away(self, start, threat):
        """
        (Dành riêng cho Ghost) Tìm bước đi hợp lệ để tránh xa threat nhất có thể.
        """
        if threat is None:
            return Move.STAY
            
        dist_map = self._precompute_distances(threat, self.known_map)
        
        best_move = Move.STAY
        best_dist = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(start, move, self.known_map):
                delta_row, delta_col = move.value
                nr, nc = start[0] + delta_row, start[1] + delta_col
                
                # Chọn bước đi làm khoảng cách tới threat TĂNG lên cao nhất
                if dist_map[nr, nc] > best_dist and dist_map[nr, nc] != 100:
                    best_dist = dist_map[nr, nc]
                    best_move = move
                    
        return best_move

    def _precompute_distances(self, goal, map_state):
        rows, cols = map_state.shape
        # Khởi tạo mảng khoảng cách với giá trị lớn (1000)
        distances = np.full((rows, cols), 1000)
        queue = collections.deque([(goal, 0)])
        distances[goal] = 0

        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and map_state[nr, nc] == 0:
                    if distances[nr, nc] == 1000:
                        distances[nr, nc] = dist + 1
                        queue.append(((nr, nc), dist + 1))
        return distances
    
class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.max_depth = 4
        self.name = "AIO's Ghost"
        self.last_known_enemy_pos = None
        self.start_time = 0
        self.TIME_LIMIT = 0.9
    
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
        if not hasattr(self, 'known_map') or self.known_map is None:
            self.known_map = np.copy(map_state)
        else:
            visible_cells = (map_state != -1)
            self.known_map[visible_cells] = map_state[visible_cells]
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        self.start_time = time.time()  
        threat = self.last_known_enemy_pos
        
        if threat is None:
            if self.last_known_enemy_pos is not None:
                return self._bfs_run_away(my_position, self.last_known_enemy_pos)
            else:
                target = self._find_nearest_unexplored(my_position)
                if target:
                    return self._bfs_move_towards(my_position, target, is_pacman=False)
                return Move.STAY

        self.dist_map = self._precompute_distances(threat, self.known_map)

        # MIN player at first 
        best_val = float('inf')
        best_action = Move.STAY
        alpha = -float('inf')
        beta = float('inf')

        actions = self._hide_action(my_position, self.known_map)
        # Move Ordering: ưu tiên các ô cách xa Pacman nhất
        actions.sort(key=lambda a: self.dist_map[self._result(my_position, a)[0], self._result(my_position, a)[1]], reverse=True)

        for action in actions:
            next_hide_pos = self._result(my_position, action)
            val = self.max_value(next_hide_pos, threat, step_number + 1, depth=1, alpha=alpha, beta=beta)
            
            if val < best_val:
                best_val = val
                best_action = action[0] 
                
            beta = min(beta, best_val)
            
            if time.time() - self.start_time > self.TIME_LIMIT:
                break
        """
        TG = time.time() - self.start_time
        print(f"Step {step_number} - Ghost thinks: {TG:.4f} s")
        """
        return best_action
            
    # --- Các hàm Minimax của Ghost ---

    def max_value(self, ghost_pos, pacman_pos, steps, depth, alpha, beta):
        if self._terminal(pacman_pos, ghost_pos, steps):
            return self._utility(pacman_pos, ghost_pos, steps, depth)

        if depth >= self.max_depth or (time.time() - self.start_time > self.TIME_LIMIT):
            return -self.dist_map[ghost_pos[0], ghost_pos[1]]

        v = -float('inf')
        for action in self._seek_actions(pacman_pos, self.known_map):
            next_pac = self._result(pacman_pos, action)
            v = max(v, self.min_value(ghost_pos, next_pac, steps + 1, depth + 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, ghost_pos, pacman_pos, steps, depth, alpha, beta):
        if self._terminal(pacman_pos, ghost_pos, steps):
            return self._utility(pacman_pos, ghost_pos, steps, depth)

        if depth >= self.max_depth or (time.time() - self.start_time > self.TIME_LIMIT):
            return -self.dist_map[ghost_pos[0], ghost_pos[1]]

        v = float('inf')
        for action in self._hide_action(ghost_pos, self.known_map):
            next_ghost = self._result(ghost_pos, action)
            v = min(v, self.max_value(next_ghost, pacman_pos, steps + 1, depth + 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    # Helper methods (you can add more)
    def _seek_actions(self, pos, map_state: np.ndarray):
        actions = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            max_steps = self._max_valid_steps(pos, move, map_state, self.pacman_speed)
            for steps in range(1, max_steps + 1):
                actions.append((move, steps))
        return actions

    def _hide_action(self, pos, map_state: np.ndarray):
        actions = [(Move.STAY, 0)]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(pos, move, map_state):
                actions.append((move, 1))
        return actions

    def _terminal(self, pacman_pos, ghost_pos, steps):
        if manhattan_distance(pacman_pos, ghost_pos) < 2: return True
        if steps > 200: return True
        return False

    def _result(self, pos, action):
        move, step = action
        dr, dc = move.value
        return (pos[0] + dr * step, pos[1] + dc * step)

    def _utility(self, pacman_pos, ghost_pos, steps, depth):
        if manhattan_distance(pacman_pos, ghost_pos) < 2:
            return 1000 - depth
        return -1000

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state): break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    def _find_nearest_unexplored(self, start):
        """
        Dùng BFS loang ra để tìm ô '0' (an toàn) gần nhất 
        mà nằm sát vách với một ô '-1' (sương mù).
        """
        rows, cols = self.known_map.shape
        queue = collections.deque([start])
        visited = set([start])
        
        while queue:
            r, c = queue.popleft()
            
            # Kiểm tra 4 hướng xung quanh ô hiện tại
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Nếu thấy sương mù (-1) sát bên, thì ô hiện tại (r, c) là mép rìa an toàn
                    if self.known_map[nr, nc] == -1:
                        return (r, c) 
            
            # Nếu chưa thấy sương mù, tiếp tục loang ra các ô an toàn (0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if self.known_map[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return None # Đã khám phá hết map

    def _bfs_move_towards(self, start, target, is_pacman=True):
        """
        Đi từng bước một hướng về target dựa trên mảng khoảng cách BFS.
        """
        if start == target or target is None:
            return (Move.STAY, 1) if is_pacman else Move.STAY
            
        # Gọi lại hàm precompute thần thánh để đo khoảng cách từ target tới mọi nơi
        dist_map = self._precompute_distances(target, self.known_map)
        
        best_move = Move.STAY
        best_dist = float('inf')
        
        # Thử 4 hướng đi từ vị trí hiện tại
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(start, move, self.known_map):
                delta_row, delta_col = move.value
                nr, nc = start[0] + delta_row, start[1] + delta_col
                
                # Chọn bước đi làm khoảng cách tới target giảm nhiều nhất
                if dist_map[nr, nc] < best_dist:
                    best_dist = dist_map[nr, nc]
                    best_move = move
                    
        return (best_move, 1) if is_pacman else best_move

    def _bfs_run_away(self, start, threat):
        """
        (Dành riêng cho Ghost) Tìm bước đi hợp lệ để tránh xa threat nhất có thể.
        """
        if threat is None:
            return Move.STAY
            
        dist_map = self._precompute_distances(threat, self.known_map)
        
        best_move = Move.STAY
        best_dist = -1
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(start, move, self.known_map):
                delta_row, delta_col = move.value
                nr, nc = start[0] + delta_row, start[1] + delta_col
                
                # Chọn bước đi làm khoảng cách tới threat TĂNG lên cao nhất
                if dist_map[nr, nc] > best_dist and dist_map[nr, nc] != 100:
                    best_dist = dist_map[nr, nc]
                    best_move = move
                    
        return best_move

    def _precompute_distances(self, goal, map_state):
        rows, cols = map_state.shape
        # Khởi tạo mảng khoảng cách với giá trị lớn (1000)
        distances = np.full((rows, cols), 1000)
        queue = collections.deque([(goal, 0)])
        distances[goal] = 0

        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and map_state[nr, nc] == 0:
                    if distances[nr, nc] == 1000:
                        distances[nr, nc] = dist + 1
                        queue.append(((nr, nc), dist + 1))
        return distances