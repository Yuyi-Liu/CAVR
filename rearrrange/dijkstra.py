import numpy as np
from collections import deque

def bfs(grid, start, end):
    m, n = grid.shape
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == end:
            return dist
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and (nx, ny) not in visited and grid[nx][ny] == 1:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

def floyd(grid, points):
    n = len(points)
    d = np.full((n, n), 10000000, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                d[i][j] = 0
            else:
                start, end = points[i], points[j]
                dist = bfs(grid, start, end)
                if dist != -1:
                    d[i][j] = dist
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d

if __name__ == '__main__':
    grid = np.array([[1, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1],
                    [0, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 1]])
    points = [(0, 0), (1, 0), (2, 2), (3, 2), (4, 1)]
    distances = floyd(grid, points)
    print(distances)

