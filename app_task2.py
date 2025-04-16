import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import heapq

# A* search helper functions
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(grid, start, goal):
    neighbors = [(0,1), (1,0), (0,-1), (-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            tentative_g_score = gscore[current] + 1
            # Check boundaries
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                if grid[neighbor[0]][neighbor[1]] == 1:
                    continue  # obstacle cell
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return False

# Streamlit App Layout
st.title("Goal-Oriented Agent: Grid Path Finder")
st.write("This interactive app simulates an agent navigating a grid to reach a goal using the A* search algorithm.")

# Parameters
grid_size = st.slider("Select grid size", min_value=5, max_value=20, value=10)
obstacle_prob = st.slider("Obstacle probability", min_value=0.0, max_value=1.0, value=0.2)

# Generate grid with obstacles (0 = free, 1 = obstacle)
grid = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        if np.random.rand() < obstacle_prob:
            grid[i, j] = 1

# Set start and goal positions
start = (0, 0)
goal = (grid_size - 1, grid_size - 1)
grid[start[0], start[1]] = 0  # ensure start is free
grid[goal[0], goal[1]] = 0    # ensure goal is free

st.write("**Start:**", start, " **Goal:**", goal)

# Find path using A* algorithm
path = astar(grid, start, goal)

# Display the grid and the path
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(grid, cmap="binary")
if path:
    x_coords = [p[1] for p in path]
    y_coords = [p[0] for p in path]
    ax.plot(x_coords, y_coords, color="red", linewidth=2, marker="o")
    st.success("Path found!")
else:
    st.error("No path found. Try lowering the obstacle probability!")
ax.set_title("Grid World with Obstacles and Path")
st.pyplot(fig)
