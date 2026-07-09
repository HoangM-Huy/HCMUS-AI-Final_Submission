import numpy as np
from collections import deque
from environment import Move

def _evaluation_heuristic(my_position, enemy_position, map_state):
    """
    Evaluates the current state by combining the shortest safe path distance 
    and the mobility score. Higher mobility means more escape routes.
    """
    distance_score = -_bfs_maze_distance(my_position, enemy_position, map_state)

    my_freedom = _degree_of_freedom(my_position, map_state)
    enemy_freedom = _degree_of_freedom(enemy_position, map_state)
    mobility_score = 0.5 * (my_freedom - enemy_freedom)

    return distance_score + mobility_score

def _degree_of_freedom(pos, map_state):
    """
    Counts the number of directly adjacent safe cells (0). 
    Used to measure how easily an agent can move without getting trapped.
    """
    row, col = pos
    height, width = map_state.shape
    count = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width and map_state[nr, nc] == 0:
            count += 1
    return count

def _bfs_maze_distance(start, goal, map_state):
    """
    Computes the shortest walking distance to the goal using BFS, strictly 
    stepping on known safe cells (0). Falls back to Manhattan distance if 
    the goal is hidden in the fog.
    """
    if start == goal:
        return 0
    queue = [(start, 0)]
    visited = {start}
    while len(queue) > 0:
        curr, dist = queue.pop(0)
        if abs(curr[0] - goal[0]) + abs(curr[1] - goal[1]) < 2:
            return dist + 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr[0] + dr, curr[1] + dc
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1]:
                if map_state[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))
    
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def _manhattan(my_position, enemy_position):
    """Calculates the standard Manhattan distance between two grid points."""
    return abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])

_DIRS = [(-1, 0, Move.UP), (1, 0, Move.DOWN), (0, -1, Move.LEFT), (0, 1, Move.RIGHT)]

def nearest_unvisited_move(my_position, map_state, visited_cells):
    """
    Uses BFS to find the best initial move that leads to the nearest unvisited 
    cell. Returns None if all cells have been visited or are unreachable.
    """
    height, width = map_state.shape
    seen = {my_position}
    queue = deque([(my_position, None)])
    while queue:
        pos, first_move = queue.popleft()
        for dr, dc, mv in _DIRS:
            nr, nc = pos[0] + dr, pos[1] + dc
            npos = (nr, nc)
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if map_state[nr, nc] == 1 or npos in seen:
                continue
            seen.add(npos)
            move = first_move if first_move is not None else mv
            if npos not in visited_cells:
                return move
            queue.append((npos, move))
    return None

def is_valid_position(pos, map_state):
    """Check if a position is valid (not a wall and within bounds)."""
    row, col = pos
    height, width = map_state.shape
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    return map_state[row, col] != 1

def is_terminal(my_position, enemy_position, steps):
    """Check if the game ends"""
    if _manhattan(my_position, enemy_position) < 2:
        return True
    if steps > 200:
        return True
    return False

def get_result(my_position, action):
    """The transition model"""
    move, step = action
    delta_row, delta_col = move.value
    return (my_position[0] + delta_row * step, my_position[1] + delta_col * step)

def get_utility(my_position, enemy_position, steps, depth):
    """Gives out scores for the agents"""
    if _manhattan(my_position, enemy_position) < 2:
        return 1000 - depth
    return -1000