import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import streamlit as st
from torchdiffeq import odeint

st.set_page_config(page_title="🕸️ Neural ODE Demo", layout="centered")
st.title("🕸️ Neural ODE – Continuous Dynamics Modeling")

# ------------------------------
# Neural ODE Function Definition
# ------------------------------
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t, y):
        return self.net(y)

# ------------------------------
# Generate Synthetic Data
# ------------------------------
def get_data():
    t = torch.linspace(0, 25, 500)
    # Shape: (500, 2)
    y_true = torch.stack((torch.sin(t), torch.cos(t)), dim=1)
    return t, y_true

# ------------------------------
# Train Neural ODE
# ------------------------------
def train_neural_ode():
    func = ODEFunc()
    optimizer = optim.Adam(func.parameters(), lr=0.01)
    t, y_true = get_data()
    y0 = y_true[0].unsqueeze(0)

    for epoch in range(300):
        optimizer.zero_grad()
        y_pred = odeint(func, y0, t).squeeze(1)
        loss = torch.mean((y_pred - y_true) ** 2)
        loss.backward()
        optimizer.step()

    return func, t, y0, y_true

# ------------------------------
# Streamlit UI
# ------------------------------
with st.spinner("🔄 Training Neural ODE... please wait (~10s)"):
    func, t, y0, y_true = train_neural_ode()

with torch.no_grad():
    y_pred = odeint(func, y0, t).squeeze(1)

# ------------------------------
# Plot Results
# ------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(y_true[:, 0], y_true[:, 1], 'g-', label='True Trajectory')
ax.plot(y_pred[:, 0], y_pred[:, 1], 'r--', label='Neural ODE Prediction')
ax.set_xlabel("X(t)")
ax.set_ylabel("Y(t)")
ax.set_title("Learned Continuous Dynamics")
ax.legend()
st.pyplot(fig)

# ------------------------------
# Explanation
# ------------------------------
st.markdown("""
### 🧠 Neural ODE Concept:
Instead of stacking discrete layers like in traditional neural networks, Neural ODEs learn a **continuous transformation** of data using differential equations.

**Applications:**
- Time-series forecasting 📈  
- Physics simulations ⚙️  
- Continuous normalizing flows 🌊  

✅ **Key Takeaway:** Neural ODEs combine deep learning with continuous-time modeling — allowing smoother, more memory-efficient learning.
""")

