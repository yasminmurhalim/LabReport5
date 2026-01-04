import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="ReLU Visualizer")

# Title and Intro
st.title("Activation Function Visualizer")
st.header("1. Rectified Linear Unit (ReLU)")

st.markdown("""
The **ReLU** function is the most widely used activation function in deep learning. 
It outputs the input directly if it is positive, otherwise, it outputs zero.
""")

# Sidebar Controls
st.sidebar.header("Input Settings")
x_range = st.sidebar.slider(
    "Select X-axis Range", 
    min_value=5, 
    max_value=50, 
    value=10,
    help="Adjust how wide the graph view is."
)

# Data Generation
# Generate 100 points between -x_range and +x_range
x = np.linspace(-x_range, x_range, 400)

# Calculate ReLU: f(x) = max(0, x)
y = np.maximum(0, x)

# Visualization (Plotly) 
fig = go.Figure()

# Add the main ReLU line
fig.add_trace(go.Scatter(
    x=x, 
    y=y,
    mode='lines',
    name='ReLU Output',
    line=dict(color='#00CC96', width=4)
))

# Add a dashed reference line for y=0
fig.add_trace(go.Scatter(
    x=[min(x), max(x)], 
    y=[0, 0],
    mode='lines',
    name='Zero Baseline',
    line=dict(color='gray', width=1, dash='dash')
))

# Layout updates for a clean look
fig.update_layout(
    title="ReLU Function: f(x) = max(0, x)",
    xaxis_title="Input (x)",
    yaxis_title="Output (Activation)",
    template="plotly_dark",  # Dark mode graph
    height=500
)

# Render the chart in the app
st.plotly_chart(fig, use_container_width=True)

# --- Mathematical Explanation Section ---
with st.expander("See Mathematical Properties"):
    st.latex(r'''
        f(x) = \begin{cases} 
          0 & \text{if } x < 0 \\
          x & \text{if } x \ge 0 
       \end{cases}
    ''')
    st.write("""
    * **Linearity:** It is linear for all positive values.
    * **Sparsity:** It outputs true zero for negative values, allowing models to be sparse (efficient).
    * **Gradient:** The gradient is 1 for $x > 0$ and 0 for $x < 0$.
    """)