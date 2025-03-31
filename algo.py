import streamlit as st
import time
import numpy as np
import pandas as pd

# Maze setup
maze_states = ["Maze Entry", "Cheese", "Tom", "Empty", "Empty", "Empty", "GOAL (Home)"]
rewards = [0, 1, -10, 0, 0, 0, 10]
state_positions = {
    0: (1, 1),   # Maze Entry
    1: (1, 2),   # Cheese
    2: (1, 3),   # Tom
    3: (2, 1),   # Empty
    4: (2, 2),   # Empty
    5: (2, 3),   # Empty
    6: (3, 2)    # GOAL (Home)
}

# Initialize Q-table
q_table = np.zeros((len(maze_states), len(maze_states)))
learning_rate = 0.1
discount_rate = 0.99
episodes = 1000

def get_possible_actions(current_state):
    """Return possible next states from current state"""
    if current_state == 0:  # Maze Entry
        return [1, 3]
    elif current_state == 1:  # Cheese
        return [0, 2, 4]
    elif current_state == 2:  # Tom
        return []  # Episode ends
    elif current_state == 3:  # Empty 1
        return [0, 4, 6]
    elif current_state == 4:  # Empty 2
        return [1, 3, 5]
    elif current_state == 5:  # Empty 3
        return [4, 6]
    elif current_state == 6:  # GOAL
        return []  # Episode ends
    return []

def choose_action(current_state, epsilon=0.1):
    """Epsilon-greedy action selection"""
    possible_actions = get_possible_actions(current_state)
    if not possible_actions:
        return None
    
    if np.random.random() < epsilon:
        return np.random.choice(possible_actions)
    else:
        return np.argmax(q_table[current_state, possible_actions])

def visualize_maze(current_state, path, total_reward, step):
    """Create a visual representation of the maze"""
    maze_visual = np.full((4, 4), "", dtype=object)
    
    # Fill in the maze elements
    for state, (row, col) in state_positions.items():
        if state == current_state:
            maze_visual[row, col] = "🐭"
        elif maze_states[state] == "Tom":
            maze_visual[row, col] = "🐱"
        elif maze_states[state] == "Cheese":
            maze_visual[row, col] = "🧀"
        elif maze_states[state] == "GOAL (Home)":
            maze_visual[row, col] = "🏠"
        else:
            maze_visual[row, col] = "⬜"
    
    # Mark visited path
    for visited_state in path[:-1]:
        row, col = state_positions[visited_state]
        maze_visual[row, col] = "🟩"
    
    # Display the maze
    st.write(f"### Step {step + 1} - Current Reward: {total_reward}")
    for row in maze_visual:
        cols = st.columns(4)
        for i, cell in enumerate(row):
            cols[i].write(cell)
    
    # Display current state info
    st.write(f"**Current State:** {maze_states[current_state]}")
    st.write(f"**Reward this step:** {rewards[current_state]}")
    st.write("---")

def train_agent():
    """Train the agent using Q-learning"""
    for episode in range(episodes):
        current_state = 0  # Start at maze entry
        path = [current_state]
        done = False
        step = 0
        
        while not done and step < 5:
            action = choose_action(current_state)
            if action is None:
                break
                
            next_state = action
            reward = rewards[next_state]
            
            # Q-learning update
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + discount_rate * q_table[next_state, best_next_action]
            td_error = td_target - q_table[current_state, next_state]
            q_table[current_state, next_state] += learning_rate * td_error
            
            current_state = next_state
            path.append(current_state)
            step += 1
            
            # Check if episode ended
            if current_state == 2 or current_state == 6 or step >= 5:
                done = True
    
    return q_table

def run_episode(q_table):
    """Run a single episode using the trained Q-table"""
    current_state = 0  # Start at maze entry
    path = [current_state]
    total_reward = 0
    step = 0
    
    visualize_maze(current_state, path, total_reward, step)
    time.sleep(1)
    
    while step < 5:
        possible_actions = get_possible_actions(current_state)
        if not possible_actions:
            break
            
        # Choose best action from Q-table
        action = np.argmax(q_table[current_state, possible_actions])
        next_state = possible_actions[action]
        
        total_reward += rewards[next_state]
        current_state = next_state
        path.append(current_state)
        step += 1
        
        visualize_maze(current_state, path, total_reward, step)
        time.sleep(1)
        
        # Check termination conditions
        if current_state == 2:  # Tom
            st.error("Oh no! Jerry got caught by Tom! Game Over!")
            break
        elif current_state == 6:  # Home
            st.success("Yay! Jerry made it home with the cheese!")
            break
        elif step >= 5:
            st.warning("Jerry took too many steps! Episode ended.")
            break

# Streamlit app
st.title("🐭 Jerry's Maze Adventure")
st.write("""
Jerry needs to navigate the maze to get the cheese (🧀) and reach home (🏠) 
while avoiding Tom the cat (🐱). Watch as Jerry learns the best path!
""")

# Train the agent
if st.button("Train Jerry"):
    st.write("Training Jerry... (this might take a moment)")
    q_table = train_agent()
    st.success("Training complete!")
    st.write("Optimal Q-table:")
    st.dataframe(pd.DataFrame(q_table, 
                             index=maze_states, 
                             columns=maze_states).round(2))

if st.button("Run Episode"):
    try:
        if 'q_table' not in globals():
            st.warning("Please train Jerry first!")
        else:
            st.write("Running an episode...")
            run_episode(q_table)
    except:
        st.error("Please train Jerry first by clicking 'Train Jerry'!")

# Display maze legend
st.sidebar.title("Maze Legend")
st.sidebar.write("🐭 - Jerry (current position)")
st.sidebar.write("🧀 - Cheese (+1 reward)")
st.sidebar.write("🐱 - Tom (-10 penalty)")
st.sidebar.write("🏠 - Home (+10 reward)")
st.sidebar.write("🟩 - Visited path")
st.sidebar.write("⬜ - Empty space (0 reward)")

# Display parameters
st.sidebar.title("Parameters")
st.sidebar.write(f"Learning rate: {learning_rate}")
st.sidebar.write(f"Discount rate: {discount_rate}")
st.sidebar.write("Max steps per episode: 5")
