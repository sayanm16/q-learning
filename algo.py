import streamlit as st
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

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

# Initialize Q-table with correct dimensions
num_states = len(maze_states)
q_table = np.zeros((num_states, num_states))
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
        return possible_actions[np.argmax(q_table[current_state, possible_actions])]

def create_maze_image(current_state, path, total_reward, step, jerry_pos=None):
    """Create a visual representation of the maze with animation frame"""
    img = Image.new('RGB', (800, 600), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    cell_size = 100
    padding = 50
    
    # Draw cells
    for state, (row, col) in state_positions.items():
        x = col * cell_size + padding
        y = row * cell_size + padding
        
        # Cell background
        if state in path:
            draw.rectangle([x, y, x+cell_size, y+cell_size], fill=(200, 255, 200))
        else:
            draw.rectangle([x, y, x+cell_size, y+cell_size], fill=(240, 240, 240))
        
        # Cell border
        draw.rectangle([x, y, x+cell_size, y+cell_size], outline=(0, 0, 0), width=2)
        
        # State content
        if state == current_state and jerry_pos is None:
            draw.text((x+40, y+40), "🐭", font=None, fill=(0, 0, 0))
        elif maze_states[state] == "Tom":
            draw.text((x+40, y+40), "🐱", font=None, fill=(0, 0, 0))
        elif maze_states[state] == "Cheese":
            draw.text((x+40, y+40), "🧀", font=None, fill=(0, 0, 0))
        elif maze_states[state] == "GOAL (Home)":
            draw.text((x+40, y+40), "🏠", font=None, fill=(0, 0, 0))
    
    # Draw animated Jerry if position is specified
    if jerry_pos is not None:
        x_pos, y_pos = jerry_pos
        draw.text((x_pos, y_pos), "🐭", font=None, fill=(0, 0, 0))
    
    # Add info text
    draw.text((padding, 10), f"Step: {step + 1}", fill=(0, 0, 0))
    draw.text((padding, 30), f"Total Reward: {total_reward}", fill=(0, 0, 0))
    draw.text((padding, 50), f"Current State: {maze_states[current_state]}", fill=(0, 0, 0))
    draw.text((padding, 70), f"Reward: {rewards[current_state]}", fill=(0, 0, 0))
    
    return img

def animate_movement(start_state, end_state, path, total_reward, step):
    """Create animation frames for movement between states"""
    frames = []
    
    # Get positions
    start_row, start_col = state_positions[start_state]
    end_row, end_col = state_positions[end_state]
    
    cell_size = 100
    padding = 50
    
    # Calculate pixel positions
    start_x = start_col * cell_size + padding + 40
    start_y = start_row * cell_size + padding + 40
    end_x = end_col * cell_size + padding + 40
    end_y = end_row * cell_size + padding + 40
    
    # Create frames for smooth movement
    steps = 10
    for i in range(steps + 1):
        progress = i / steps
        current_x = start_x + (end_x - start_x) * progress
        current_y = start_y + (end_y - start_y) * progress
        frame = create_maze_image(start_state, path, total_reward, step, (current_x, current_y))
        frames.append(frame)
    
    return frames

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
            possible_next_actions = get_possible_actions(next_state)
            if possible_next_actions:
                best_next_action = np.argmax(q_table[next_state, possible_next_actions])
                td_target = reward + discount_rate * q_table[next_state, possible_next_actions[best_next_action]]
            else:
                td_target = reward
                
            td_error = td_target - q_table[current_state, next_state]
            q_table[current_state, next_state] += learning_rate * td_error
            
            current_state = next_state
            path.append(current_state)
            step += 1
            
            # Check if episode ended
            if current_state == 2 or current_state == 6 or step >= 5:
                done = True
    
    # Create Q-table DataFrame with proper labels
    q_df = pd.DataFrame(q_table, 
                       index=maze_states,
                       columns=maze_states)
    return q_df

def run_episode(q_table):
    """Run a single episode using the trained Q-table with animations"""
    current_state = 0  # Start at maze entry
    path = [current_state]
    total_reward = 0
    step = 0
    
    # Create placeholder for animation
    animation_placeholder = st.empty()
    
    # Show initial state
    img = create_maze_image(current_state, path, total_reward, step)
    animation_placeholder.image(img, caption=f"Step {step + 1}")
    time.sleep(1)
    
    while step < 5:
        possible_actions = get_possible_actions(current_state)
        if not possible_actions:
            break
            
        # Choose best action from Q-table
        action = possible_actions[np.argmax(q_table[current_state, possible_actions])]
        next_state = action
        
        # Create movement animation
        frames = animate_movement(current_state, next_state, path, total_reward, step)
        for frame in frames:
            animation_placeholder.image(frame, caption=f"Step {step + 1}")
            time.sleep(0.05)
        
        total_reward += rewards[next_state]
        current_state = next_state
        path.append(current_state)
        step += 1
        
        # Show final position after movement
        img = create_maze_image(current_state, path, total_reward, step)
        animation_placeholder.image(img, caption=f"Step {step + 1}")
        time.sleep(0.5)
        
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
st.title("🐭 Jerry's Animated Maze Adventure")
st.write("""
Jerry needs to navigate the maze to get the cheese (🧀) and reach home (🏠) 
while avoiding Tom the cat (🐱). Watch Jerry move smoothly through the maze!
""")

# Initialize session state for Q-table
if 'q_df' not in st.session_state:
    st.session_state.q_df = None

# Train the agent
if st.button("Train Jerry"):
    st.write("Training Jerry... (this might take a moment)")
    with st.spinner("Training in progress..."):
        st.session_state.q_df = train_agent()
    st.success("Training complete!")
    st.write("Optimal Q-table:")
    st.dataframe(st.session_state.q_df.round(2))

if st.button("Run Animated Episode"):
    if st.session_state.q_df is not None:
        st.write("Running animated episode...")
        # Convert DataFrame back to numpy array for the episode
        q_table_array = st.session_state.q_df.to_numpy()
        run_episode(q_table_array)
    else:
        st.warning("Please train Jerry first by clicking 'Train Jerry'!")

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
