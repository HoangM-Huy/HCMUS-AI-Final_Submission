import collections

def _evaluation_heuristic(my_position, enemy_position, map_state):
    # FIX: Calculate actual shortest maze walking distance instead of Manhattan distance
    return -_bfs_maze_distance(my_position, enemy_position, map_state)


def _bfs_maze_distance(start, goal, map_state):
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
    return 100

def _manhattan(my_position, enemy_position):
    return abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])

def merge_observation(known_map, map_state):
    """
    Update the agent's persistent memory of the maze.
    """
    mask = map_state != -1
    known_map[mask] = map_state[mask]
    return known_map

def bfs_next_move(start, goal, known_map, moves):
    """
    Finds the next move strictly traversing known safe cells (0).
    """
    if start == goal:
        return None
    rows, cols = known_map.shape
    visited = {start}
    queue = collections.deque([(start, None)])
    while queue:
        (r, c), first_move = queue.popleft()
        for move in moves:
            dr, dc = move.value
            nr, nc = r + dr, c + dc
            if (nr, nc) == goal:
                return first_move if first_move is not None else move
            
            # Only walk on known '0' cells
            if 0 <= nr < rows and 0 <= nc < cols and known_map[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), first_move if first_move is not None else move))
    return None

def nearest_frontier(start, known_map, avoid=None):
    """
    Find the closest known open cell (0) that borders an unknown cell (-1).
    """
    rows, cols = known_map.shape
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    avoid = avoid or set()
    visited = {start}
    queue = collections.deque([start])
    
    while queue:
        r, c = queue.popleft()
        
        # Check if this safe cell borders the fog
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and known_map[nr, nc] == -1:
                if (r, c) != start and (r, c) not in avoid:
                    return (r, c) # Return the safe cell on the border
                    
        # Continue searching through safe cells
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and known_map[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return None