"""
=============================================================================
BITS Pilani - WILPD - MTech AIML
Course: AIMLCZG557/AECLZG557
Assignment 1 - PS10: Grid Travel Agent
=============================================================================
"""



import heapq
import time
import tracemalloc
import sys
import os

# ---------------------------------------------------------------------------
# Cell type constants
# ---------------------------------------------------------------------------
EMPTY       = 'E'
KNIGHT      = 'K'
QUEEN       = 'Q'
WATER       = 'W'   # obstacle — reward +5 when adjacent
FOREST      = 'F'   # obstacle — reward +5 when adjacent
FIRE        = 'X'   # obstacle — penalty -5 when adjacent
WILD_ANIMAL = 'A'   # obstacle — penalty -5 when adjacent
MOUNTAIN    = 'M'   # obstacle — penalty -3 when adjacent

# ---------------------------------------------------------------------------
# Cells the knight is ALLOWED to step on
# ---------------------------------------------------------------------------
WALKABLE_CELLS = {EMPTY, KNIGHT, QUEEN}

# ---------------------------------------------------------------------------
# Cells that are IMPASSABLE obstacles (knight cannot enter these)
# Reward/penalty is applied when the knight is ADJACENT to these cells,
# not when standing on them.
# ---------------------------------------------------------------------------
OBSTACLE_CELLS = {WATER, FOREST, FIRE, WILD_ANIMAL, MOUNTAIN}

# ---------------------------------------------------------------------------
# Adjacency reward/penalty lookup
# Applied when the knight stands next to one of these obstacle cells.
# ---------------------------------------------------------------------------
ADJACENCY_REWARD = {
    WATER:        +5,
    FOREST:       +5,
    FIRE:         -5,
    WILD_ANIMAL:  -5,
    MOUNTAIN:     -3,
    EMPTY:         0,
    KNIGHT:        0,
    QUEEN:         0,
}

# ---------------------------------------------------------------------------
# Display symbols for grid ASCII output
# ---------------------------------------------------------------------------
CELL_DISPLAY = {
    EMPTY:       '  .  ',
    KNIGHT:      '  K  ',
    QUEEN:       '  Q  ',
    WATER:       '  W  ',
    FOREST:      '  F  ',
    FIRE:        '  X  ',
    WILD_ANIMAL: '  A  ',
    MOUNTAIN:    '  M  ',
}


# ===========================================================================
# Data Structure 1: Priority Queue (Min-Heap) — OPEN list for A*
# ===========================================================================

class PriorityQueue:
    """
    Min-heap based priority queue used as the OPEN list in A*.

    Each element is stored internally as (priority, item).
    The heap invariant ensures the element with the smallest priority
    is always at the top (index 0).

    Capacity is bounded to prevent unbounded memory growth.
    Raises OverflowError on insert when full.
    Raises IndexError on extract/peek when empty.
    """

    def __init__(self, capacity=100000):
        """
        Initialise an empty priority queue.

        Parameters:
            capacity (int): Maximum number of elements allowed.
        """
        self._heap     = []
        self._capacity = capacity
        self._size     = 0

    def is_empty(self):
        """Return True if the queue contains no elements."""
        return self._size == 0

    def is_full(self):
        """Return True if the queue has reached its maximum capacity."""
        return self._size >= self._capacity

    def insert(self, priority, item):
        """
        Insert item with the given priority into the heap.

        Parameters:
            priority: Comparable value used to order elements.
            item    : The payload to store alongside the priority.

        Raises:
            OverflowError: If the queue is at full capacity.
        """
        if self.is_full():
            raise OverflowError(
                f"[PriorityQueue ERROR] Cannot insert: queue is at full "
                f"capacity ({self._capacity}). No more elements can be added."
            )
        heapq.heappush(self._heap, (priority, item))
        self._size += 1

    def extract_min(self):
        """
        Remove and return the (priority, item) pair with the lowest priority.

        Returns:
            Tuple (priority, item).

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError(
                "[PriorityQueue ERROR] Cannot extract_min: queue is empty. "
                "No elements to remove."
            )
        priority, item = heapq.heappop(self._heap)
        self._size -= 1
        return priority, item

    def peek(self):
        """
        Return the (priority, item) of the minimum element WITHOUT removing it.

        Returns:
            Tuple (priority, item).

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError(
                "[PriorityQueue ERROR] Cannot peek: queue is empty. "
                "No elements to inspect."
            )
        return self._heap[0]

    def __len__(self):
        """Return the current number of elements in the queue."""
        return self._size


# ===========================================================================
# Data Structure 2: Closed Set (Hash Set) — CLOSED list for A*
# ===========================================================================

class ClosedSet:
    """
    Hash-set for tracking visited (expanded) nodes in A*.

    Using a hash set gives O(1) average-case insertion and lookup,
    which is critical for A* performance since the closed-set check
    occurs at every node expansion.

    Raises TypeError if a non-tuple state is added — all states in this
    problem are (row, col) tuples, so anything else indicates a bug.
    """

    def __init__(self):
        """Initialise an empty closed set."""
        self._data = set()

    def add(self, state):
        """
        Add state to the closed set, marking it as visited.

        Parameters:
            state: A tuple representing the agent's state.

        Raises:
            TypeError: If state is not a tuple.
        """
        if not isinstance(state, tuple):
            raise TypeError(
                f"[ClosedSet ERROR] State must be a tuple, "
                f"got {type(state).__name__}. Only tuple states are valid."
            )
        self._data.add(state)

    def contains(self, state):
        """
        Return True if state has already been visited/expanded.

        Parameters:
            state: A tuple representing the agent's state.
        """
        return state in self._data

    def size(self):
        """Return the total number of visited states."""
        return len(self._data)

    def is_empty(self):
        """Return True if no states have been visited yet."""
        return len(self._data) == 0


# ===========================================================================
# Grid Class
# ===========================================================================

class Grid:
    """
    Represents the 2-D grid environment for the knight agent.

    Key design decisions:
        - Obstacle cells (Water, Forest, Fire, Wild Animal, Mountain) are
          IMPASSABLE — get_neighbors() never returns them as valid moves.
        - Reward/penalty is earned by ADJACENCY: when the knight stands on
          a walkable cell next to an obstacle cell, get_adjacency_reward()
          sums up the rewards/penalties of all neighbouring obstacle cells.
        - The grid is stored as a deep copy to prevent mutation of the
          original input data.
    """

    def __init__(self, grid_data):
        """
        Initialise the grid from a 2-D list of cell-type strings.

        Parameters:
            grid_data (list[list[str]]): 2-D array of cell type codes.
        """
        self._grid = [row[:] for row in grid_data]   # deep copy
        self.rows  = len(self._grid)
        self.cols  = len(self._grid[0]) if self.rows > 0 else 0

    def get_cell(self, row, col):
        """
        Return the cell type string at position (row, col).

        Raises:
            IndexError: If (row, col) is outside grid boundaries.
        """
        if not self.in_bounds(row, col):
            raise IndexError(
                f"[Grid ERROR] Cell ({row},{col}) is out of bounds. "
                f"Grid size is {self.rows} x {self.cols}."
            )
        return self._grid[row][col]

    def set_cell(self, row, col, cell_type):
        """
        Overwrite the cell at (row, col) with cell_type.

        Raises:
            IndexError: If (row, col) is outside grid boundaries.
        """
        if not self.in_bounds(row, col):
            raise IndexError(
                f"[Grid ERROR] Cannot set cell ({row},{col}): out of bounds. "
                f"Grid size is {self.rows} x {self.cols}."
            )
        self._grid[row][col] = cell_type

    def in_bounds(self, row, col):
        """Return True if (row, col) lies within the grid boundaries."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def find_cell(self, cell_type):
        """
        Return (row, col) of the first occurrence of cell_type.
        Returns None if cell_type is not found anywhere in the grid.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                if self._grid[r][c] == cell_type:
                    return (r, c)
        return None

    def get_neighbors(self, row, col):
        """
        Return a list of valid positions reachable from (row, col) in one step.

        Only WALKABLE cells (Empty, Knight, Queen) are included.
        Obstacle cells (Water, Forest, Fire, Wild Animal, Mountain) are
        EXCLUDED — the knight cannot enter them under any circumstances.

        Movement is 4-directional only: Up, Down, Left, Right.
        No diagonal movement is allowed.

        Parameters:
            row, col (int): Current position of the knight.

        Returns:
            list of (row, col) tuples for valid next positions.
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:   # Up Down Left Right
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and self._grid[nr][nc] in WALKABLE_CELLS:
                neighbors.append((nr, nc))
        return neighbors

    def get_adjacency_reward(self, row, col):
        """
        Compute the net reward/penalty earned by the knight standing at (row, col).

        The knight earns reward by being ADJACENT to obstacle cells —
        not by stepping on them (obstacle cells are impassable).
        All 4 cardinal neighbours are checked; their ADJACENCY_REWARD
        values are summed.

        Example:
            If the knight is at (r, c) and (r, c-1) is a Forest (+5)
            and (r, c+1) is a Fire (-5), the adjacency reward is 0.

        Parameters:
            row, col (int): Knight's current position.

        Returns:
            int: Net reward for standing at this position.
        """
        reward = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc):
                reward += ADJACENCY_REWARD.get(self._grid[nr][nc], 0)
        return reward

    def display(self, path=None):
        """
        Print an ASCII representation of the grid.

        Cells on the given path (excluding Knight and Queen start/end)
        are marked with '*' to show the route taken.

        Parameters:
            path (list of (row,col)): Optional path to highlight.
        """
        path_set = set(path) if path else set()
        print("     " + "  ".join(f"C{c}" for c in range(self.cols)))
        sep = "    +" + "------+" * self.cols
        print(sep)
        for r in range(self.rows):
            row_str = f" R{r} |"
            for c in range(self.cols):
                cell = self._grid[r][c]
                if (r, c) in path_set and cell not in (KNIGHT, QUEEN):
                    row_str += '  *  |'
                else:
                    row_str += CELL_DISPLAY.get(cell, '  ?  ') + '|'
            print(row_str)
            print(sep)
        print()


# ===========================================================================
# Heuristic Function
# ===========================================================================

def manhattan_distance(pos, goal):
    """
    Admissible and consistent heuristic: Manhattan distance.

    h(n) = |row_n - row_goal| + |col_n - col_goal|

    Since each move costs exactly 1 (uniform step cost), and Manhattan
    distance gives the minimum number of moves needed ignoring obstacles,
    it never overestimates the true cost — making it admissible.

    It is also consistent: h(n) <= 1 + h(n') for any neighbour n',
    because moving one step changes Manhattan distance by at most 1.
    A consistent heuristic guarantees A* never re-expands a node,
    ensuring both optimality and completeness.

    Parameters:
        pos  (tuple): Current (row, col) position.
        goal (tuple): Goal (row, col) position.

    Returns:
        int: Manhattan distance from pos to goal.
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


# ===========================================================================
# A* Search Algorithm
# ===========================================================================

def astar_search(grid, start, goal):
    """
    A* Search to find the optimal path from start to goal.

    Cost Model:
        step_cost = 1 for every valid move (uniform).
        Reward/penalty is accumulated separately via adjacency — it does
        NOT affect the A* search cost (g-values stay non-negative,
        keeping the Manhattan heuristic admissible and consistent).

        f(n) = g(n) + h(n)
        g(n) = number of steps from start to n
        h(n) = Manhattan distance from n to goal

    Why reward does not enter the g-cost:
        The assignment objective is to find the SHORTEST path (minimum
        cells) while maximising reward. Treating reward as negative cost
        would make g-values decrease, breaking heuristic admissibility.
        Instead, A* finds the shortest valid path, and adjacency reward
        is computed as a separate metric reported alongside path length.

    Parameters:
        grid  (Grid): The grid environment.
        start (tuple): (row, col) of the knight's starting position.
        goal  (tuple): (row, col) of the queen's position.

    Returns:
        path           (list): Ordered list of (row,col) from start to goal,
                               or None if no path exists.
        net_reward     (int) : Total adjacency reward along the path
                               (excluding the start cell).
        nodes_expanded (int) : Number of nodes popped from the OPEN list.

    Raises:
        ValueError: If start or goal is outside grid bounds.
    """
    if not grid.in_bounds(start[0], start[1]):
        raise ValueError(
            f"[A* ERROR] Start position {start} is outside the grid "
            f"({grid.rows} x {grid.cols})."
        )
    if not grid.in_bounds(goal[0], goal[1]):
        raise ValueError(
            f"[A* ERROR] Goal position {goal} is outside the grid "
            f"({grid.rows} x {grid.cols})."
        )

    open_list  = PriorityQueue(capacity=100000)   # OPEN list  (min-heap)
    closed_set = ClosedSet()                       # CLOSED list (hash-set)
    g_cost     = {start: 0}                        # best known g-cost per state
    counter    = 0                                 # tie-breaking counter

    h0 = manhattan_distance(start, goal)
    open_list.insert(h0, (counter, start, 0, [start]))

    nodes_expanded = 0

    while not open_list.is_empty():
        f, (_, current, g, path) = open_list.extract_min()
        nodes_expanded += 1

        # ---- Goal test ------------------------------------------------
        if current == goal:
            # Compute adjacency reward for every step AFTER the start cell.
            # The start cell itself earns no reward (the knight is already there).
            net_reward = sum(
                grid.get_adjacency_reward(r, c) for r, c in path[1:]
            )
            return path, net_reward, nodes_expanded

        # Skip stale entries — a node may be in the heap multiple times
        # if a cheaper path to it was found after it was first inserted.
        # The first time a node is popped (i.e. expanded) is always with
        # the lowest cost, so any later pops are discarded here.
        if closed_set.contains(current):
            continue
        closed_set.add(current)

        # ---- Expand neighbours ----------------------------------------
        for neighbor in grid.get_neighbors(current[0], current[1]):
            if closed_set.contains(neighbor):
                continue

            # Uniform step cost: every walkable move costs exactly 1.
            # This keeps g-values non-negative and the heuristic admissible.
            step_cost = 1
            new_g     = g + step_cost

            # Only insert/update if this path to neighbor is better
            # than any previously known path to it.
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                h_new            = manhattan_distance(neighbor, goal)
                f_new            = new_g + h_new
                counter         += 1
                new_path         = path + [neighbor]

                if open_list.is_full():
                    print(
                        "[WARNING] Open list is at full capacity. "
                        "Some paths cannot be explored; result may be suboptimal."
                    )
                    break

                open_list.insert(f_new, (counter, neighbor, new_g, new_path))

    # If the while loop exits without returning, no path exists.
    return None, 0, nodes_expanded


# ===========================================================================
# Input / Output Helpers
# ===========================================================================

def read_input_file(filepath):
    """
    Parse inputPS10.txt.

    Expected format:
        # optional comment lines (lines starting with #)
        START: row,col
        cell,cell,cell,...      <- grid row 0
        cell,cell,cell,...      <- grid row 1
        ...

    Cell codes: E, K, Q, W, F, X, A, M
    The START line specifies the knight's fixed starting position.

    Parameters:
        filepath (str): Path to the input file.

    Returns:
        grid  (Grid): Parsed grid object.
        start (tuple): (row, col) knight starting position.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError       : If the file content is malformed.
    """
    valid_cells = set(ADJACENCY_REWARD.keys())

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[Input ERROR] Input file not found: '{filepath}'. "
            f"Please ensure the file exists in the current directory."
        )

    with open(filepath, 'r') as f:
        raw_lines = f.readlines()

    # Remove blank lines and comment lines
    lines = [
        l.strip() for l in raw_lines
        if l.strip() and not l.strip().startswith('#')
    ]

    if not lines:
        raise ValueError(
            "[Input ERROR] Input file contains no valid content "
            "(all lines are blank or comments)."
        )

    # Parse START line and grid rows in a single pass
    start      = None
    grid_lines = []

    for line in lines:
        if line.upper().startswith("START:"):
            parts = line.split(":", 1)[1].strip().split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"[Input ERROR] START line must be 'START: row,col'. "
                    f"Got: '{line}'"
                )
            try:
                start = (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                raise ValueError(
                    f"[Input ERROR] START coordinates must be integers. "
                    f"Got: '{line}'"
                )
        else:
            grid_lines.append(line)

    if start is None:
        raise ValueError(
            "[Input ERROR] No 'START: row,col' line found in input file."
        )

    if not grid_lines:
        raise ValueError(
            "[Input ERROR] No grid rows found in input file."
        )

    # Parse each grid row into a list of cell codes
    grid_data = []
    for line_num, line in enumerate(grid_lines, 1):
        cells = [c.strip().upper() for c in line.split(',')]
        for cell in cells:
            if cell not in valid_cells:
                raise ValueError(
                    f"[Input ERROR] Unknown cell code '{cell}' on grid "
                    f"line {line_num}. Valid codes: {sorted(valid_cells)}"
                )
        grid_data.append(cells)

    # Validate that all rows have the same number of columns
    row_lengths = [len(r) for r in grid_data]
    if len(set(row_lengths)) != 1:
        raise ValueError(
            f"[Input ERROR] Grid rows have unequal lengths: {row_lengths}. "
            f"All rows must have the same number of columns."
        )

    grid = Grid(grid_data)

    # Validate that the start position falls within the grid
    if not grid.in_bounds(start[0], start[1]):
        raise ValueError(
            f"[Input ERROR] START position {start} is outside the grid "
            f"bounds ({grid.rows} x {grid.cols})."
        )

    # Validate that the start cell is walkable
    start_cell = grid.get_cell(start[0], start[1])
    if start_cell not in WALKABLE_CELLS:
        raise ValueError(
            f"[Input ERROR] START position {start} is an obstacle cell "
            f"('{start_cell}'). The knight cannot start on an obstacle."
        )

    return grid, start


def write_output_file(filepath, path, net_reward, nodes_expanded,
                      start, goal, elapsed_ms, peak_mem_kb, grid):
    """
    Write the A* search results to outputPS10.txt.

    Parameters:
        filepath       (str)  : Output file path.
        path           (list) : List of (row,col) tuples forming the path,
                                or None if no path was found.
        net_reward     (int)  : Total adjacency reward along the path.
        nodes_expanded (int)  : Number of nodes expanded during search.
        start          (tuple): Knight's start position.
        goal           (tuple): Queen's position.
        elapsed_ms     (float): Search time in milliseconds.
        peak_mem_kb    (float): Peak memory usage in kilobytes.
        grid           (Grid) : The grid (used to look up cell types).
    """
    # Human-readable names for each cell type
    cell_names = {
        EMPTY:       'Empty',
        KNIGHT:      'Knight (start)',
        QUEEN:       'Queen (goal)',
        WATER:       'Water  (+5 adj)',
        FOREST:      'Forest (+5 adj)',
        FIRE:        'Fire   (-5 adj)',
        WILD_ANIMAL: 'Wild Animal (-5 adj)',
        MOUNTAIN:    'Mountain (-3 adj)',
    }

    with open(filepath, 'w') as f:
        f.write("=" * 62 + "\n")
        f.write("  GRID TRAVEL AGENT - A* SEARCH OUTPUT\n")
        f.write("=" * 62 + "\n\n")
        f.write(f"  Grid Size             : {grid.rows} x {grid.cols}\n")
        f.write(f"  Start Position        : {start}\n")
        f.write(f"  Goal  (Queen)         : {goal}\n\n")

        if path:
            # Path as a comma-separated list of coordinates
            path_str = ", ".join(str(p) for p in path)
            f.write(f"Path taken by the knight:\n{path_str}\n\n")
            f.write(f"Number of cells traversed : {len(path)}\n")
            f.write(f"The cost of the path is   : {net_reward}\n\n")

            # Detailed step-by-step breakdown
            f.write("Detailed Path Breakdown:\n")
            f.write(f"  {'Cell':<12} {'Type':<22} {'Adj. Reward'}\n")
            f.write(f"  {'-'*12} {'-'*22} {'-'*12}\n")

            for i, pos in enumerate(path):
                r, c       = pos
                cell_type  = grid.get_cell(r, c)
                cname      = cell_names.get(cell_type, 'Unknown')

                if i == 0:
                    reward_str = "(start — no reward)"
                else:
                    adj_reward = grid.get_adjacency_reward(r, c)
                    if adj_reward > 0:
                        reward_str = f"+{adj_reward}"
                    elif adj_reward < 0:
                        reward_str = str(adj_reward)
                    else:
                        reward_str = "0"

                f.write(f"  {str(pos):<12} {cname:<22} {reward_str}\n")

            f.write(f"\n  TOTAL NET REWARD: {net_reward}\n\n")
        else:
            f.write("  No path found from start to goal.\n\n")

        f.write("=" * 62 + "\n")
        f.write("  COMPLEXITY (Measured during execution)\n")
        f.write("=" * 62 + "\n")
        f.write(f"  Nodes Expanded  : {nodes_expanded}\n")
        f.write(f"  Time Taken      : {elapsed_ms:.4f} ms\n")
        f.write(f"  Peak Memory     : {peak_mem_kb:.4f} KB\n")
        f.write("=" * 62 + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    """
    Entry point for the Grid Travel Agent.

    Workflow:
        1. Read grid and start position from inputPS10.txt.
        2. Locate the Queen (goal) on the grid.
        3. Display the initial grid.
        4. Run A* search while measuring time and memory.
        5. Print complexity metrics, path, and reward to stdout.
        6. Write results to outputPS10.txt.
    """
    print("=" * 62)
    print("  BITS Pilani WILPD - MTech AIML")
    print("  Assignment 1 - PS10: Grid Travel Agent")
    print("  A* Search Algorithm")
    print("=" * 62)
    print()

    input_file  = "inputPS10.txt"
    output_file = "outputPS10.txt"

    # ------------------------------------------------------------------
    # Step 1: Load grid and start position from input file.
    # ------------------------------------------------------------------
    print(f"[Info] Reading input from '{input_file}' ...")
    try:
        grid, start = read_input_file(input_file)
        print(f"[Info] Grid loaded      : {grid.rows} x {grid.cols}")
        print(f"[Info] Start position   : {start}")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print(f"        Please place '{input_file}' in the same directory and re-run.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[ERROR] Malformed input file: {e}")
        print(f"        Please fix '{input_file}' and re-run.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Locate the Queen (goal position).
    # ------------------------------------------------------------------
    goal = grid.find_cell(QUEEN)
    if goal is None:
        print("[ERROR] No Queen cell (Q) found in the grid. "
              "Please add a Q cell to the input file.")
        sys.exit(1)
    print(f"[Info] Goal (Queen)     : {goal}")

    # Mark the start cell as KNIGHT on the grid for display purposes.
    # If there was already a K elsewhere (e.g. decorative), clear it first.
    existing_knight = grid.find_cell(KNIGHT)
    if existing_knight and existing_knight != start:
        grid.set_cell(existing_knight[0], existing_knight[1], EMPTY)
    if grid.get_cell(start[0], start[1]) != QUEEN:
        grid.set_cell(start[0], start[1], KNIGHT)

    # ------------------------------------------------------------------
    # Step 3: Display the initial grid.
    # ------------------------------------------------------------------
    print()
    print("Initial Grid:")
    print("Legend: K=Knight(start)  Q=Queen(goal)  .=Empty(walkable)")
    print("        W=Water(+5 adj)   F=Forest(+5 adj)")
    print("        X=Fire(-5 adj)    A=Wild Animal(-5 adj)")
    print("        M=Mountain(-3 adj)")
    print("Note  : W, F, X, A, M are IMPASSABLE obstacles.")
    print("        Reward/penalty is earned by standing ADJACENT to them.\n")
    grid.display()

    # ------------------------------------------------------------------
    # Step 4: Run A* search and measure time + space complexity.
    # ------------------------------------------------------------------
    print("Running A* Search ...\n")

    tracemalloc.start()
    t0 = time.perf_counter()

    path, net_reward, nodes_expanded = astar_search(grid, start, goal)

    t1 = time.perf_counter()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_ms  = (t1 - t0) * 1000
    peak_mem_kb = peak_mem / 1024

    # ------------------------------------------------------------------
    # Step 5a: Print complexity (measured, not theoretical).
    # ------------------------------------------------------------------
    print("=" * 62)
    print("  COMPLEXITY  (Measured during execution — not theoretical)")
    print("=" * 62)
    print(f"  Grid size         : {grid.rows} x {grid.cols} "
          f"= {grid.rows * grid.cols} cells")
    print(f"  Nodes expanded    : {nodes_expanded}")
    print(f"  Measured Time     : {elapsed_ms:.4f} ms")
    print(f"  Peak Memory       : {peak_mem_kb:.4f} KB  ({peak_mem} bytes)")
    print()
    print("  [Note] Theoretical bounds: Time O(b^d), Space O(b^d)")
    print("         where b = branching factor (~4) and d = path depth.")
    print("         The heuristic prunes many branches, so actual")
    print("         nodes expanded << theoretical worst case.")
    print("=" * 62)
    print()

    # ------------------------------------------------------------------
    # Step 5b: Print results.
    # ------------------------------------------------------------------
    print("=" * 62)
    print("  RESULTS")
    print("=" * 62)

    if path:
        path_str = ", ".join(str(p) for p in path)
        print(f"\nPath taken by the knight:\n{path_str}\n")
        print(f"Number of cells traversed : {len(path)}")
        print(f"The cost of the path is   : {net_reward}")
    else:
        print("\nNo path found from start to goal.")
        print("The Queen may be unreachable due to surrounding obstacles.")

    print(f"\nNodes expanded during search: {nodes_expanded}")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Step 5c: Display grid with path highlighted, and breakdown table.
    # ------------------------------------------------------------------
    if path:
        print("\nGrid with path highlighted  (* = cells visited by knight):")
        grid.display(path=path)

        print("Detailed Path Breakdown:")
        print("(Adjacency reward = net reward earned by standing at each cell,")
        print(" based on obstacle cells adjacent to that position.)\n")
        print(f"  {'Cell':<12} {'Type':<22} {'Adj. Reward'}")
        print(f"  {'-'*12} {'-'*22} {'-'*12}")

        cell_names = {
            EMPTY:       'Empty',
            KNIGHT:      'Knight (start)',
            QUEEN:       'Queen (goal)',
            WATER:       'Water  (+5 adj)',
            FOREST:      'Forest (+5 adj)',
            FIRE:        'Fire   (-5 adj)',
            WILD_ANIMAL: 'Wild Animal (-5 adj)',
            MOUNTAIN:    'Mountain (-3 adj)',
        }

        for i, pos in enumerate(path):
            r, c      = pos
            cell_type = grid.get_cell(r, c)
            cname     = cell_names.get(cell_type, 'Unknown')

            if i == 0:
                reward_str = "(start — no reward)"
            else:
                adj_reward = grid.get_adjacency_reward(r, c)
                if adj_reward > 0:
                    reward_str = f"+{adj_reward}"
                elif adj_reward < 0:
                    reward_str = str(adj_reward)
                else:
                    reward_str = "0"

            print(f"  {str(pos):<12} {cname:<22} {reward_str}")

        print(f"\n  TOTAL NET REWARD: {net_reward}")

    # ------------------------------------------------------------------
    # Step 6: Write results to output file.
    # ------------------------------------------------------------------
    write_output_file(
        output_file, path, net_reward, nodes_expanded,
        start, goal, elapsed_ms, peak_mem_kb, grid
    )
    print(f"\n[Output] Results written to '{output_file}'.")
    print("[Done]   Program completed successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
