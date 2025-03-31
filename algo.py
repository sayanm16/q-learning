import streamlit as st
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Maze representation
maze = np.array([
    ["Start", "Cheese", "Empty"],
    ["Empty", "Tom", "Goal"]
])

rewards = {
    "Empty": 0,
    "Cheese": 1,
    "Goal": 10,
    "Tom": -10
}

# Possible paths Jerry can take
paths = [
    [(0,0), (0,1), (1,1), (1,2)],  # Path with cheese but caught by Tom
    [(0,0), (0,1), (0,2), (1,2)],  # Path directly to Goal without getting caught
    [(0,0), (1,0), (1,2)],          # Path avoiding cheese, directly to Goal
]

# Function to draw the maze
def draw_maze(path_taken):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    
    colors = {"Start": "lightblue", "Cheese": "yellow", "Empty": "white", "Tom": "red", "Goal": "green"}
    for i in range(2):
        for j in range(3):
            ax.add_patch(patches.Rectangle((j, 1-i), 1, 1, color=colors[maze[i, j]], ec='black'))
            ax.text(j+0.5, 1-i+0.5, maze[i, j], ha='center', va='center', fontsize=10, weight='bold')
    
    # Animate Jerry's movement
    for step, (x, y) in enumerate(path_taken):
        ax.plot(y+0.5, 1-x+0.5, 'o', color='brown', markersize=10, label='Jerry' if step == 0 else "")
        st.pyplot(fig)
        time.sleep(1)

# Streamlit UI
st.title("Jerry's Maze Adventure")
st.write("Watch Jerry navigate the maze to get the cheese and reach home!")

selected_path = st.selectbox("Select a path", ["Path 1 (Cheese, but caught by Tom)", "Path 2 (Cheese, safely home)", "Path 3 (No cheese, directly home)"])
path_index = {"Path 1 (Cheese, but caught by Tom)": 0, "Path 2 (Cheese, safely home)": 1, "Path 3 (No cheese, directly home)": 2}[selected_path]

if st.button("Start Animation"):
    draw_maze(paths[path_index])
