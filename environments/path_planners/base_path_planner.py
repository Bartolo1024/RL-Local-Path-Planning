import heapq
import cell
import abc

class GridGraphPathPlanner(object):
    def __init__(self, grid_height, grid_width):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.reset()

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

    @abc.abstractmethod
    def solve(self):
        pass

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

    @abc.abstractmethod
    def get_path(self):
        pass

    def reset(self):
        self.opened = []
        heapq.heapify(self.opened)
        self.closed = set()
        self.cells = []
