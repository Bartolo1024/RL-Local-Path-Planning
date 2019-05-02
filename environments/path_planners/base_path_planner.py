import heapq
import cell

class GridGraphPathPlanner(object):
    def __init__(self, grid_height, grid_width):
        self.opened = []
        heapq.heapify(self.opened)
        self.closed = set()
        self.cells = []
        self.grid_height = grid_height
        self.grid_width = grid_width

    def init_grid(self, walls):
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if (x, y) in walls:
                    reachable = False
                else:
                    reachable = True
                self.cells.append(cell.Cell(x, y, reachable))

    def set_task(self, begin, end):
        self.start = self.get_cell(*begin)
        self.end = self.get_cell(*end)

    def get_cell(self, x, y):
        return self.cells[x * self.grid_height + y]

    def get_adjacent_cells(self, cell):
        cells = []
        if cell.x < self.grid_width - 1:
            cells.append(self.get_cell(cell.x + 1, cell.y))
        if cell.x > 0:
            cells.append(self.get_cell(cell.x - 1, cell.y))
        if cell.y < self.grid_width - 1:
            cells.append(self.get_cell(cell.x, cell.y + 1))
        if cell.y > 0:
            cells.append(self.get_cell(cell.x, cell.y - 1))
        return cells

    def get_path(self):
        cell = self.end
        path = [(cell.x, cell.y)]
        while cell.parent is not self.start:
            cell = cell.parent
            path.append((cell.x, cell.y))
        path.append((self.start.x, self.start.y))
        path.reverse()
        return path

    def update_cell(self, adj, cell):
        adj.g = cell.g + 10
        adj.h = self.get_heuristic(adj)
        adj.parent = cell
        adj.f = adj.h + adj.g

    def solve(self):
        raise NotImplementedError('base path planner is an abstract class')