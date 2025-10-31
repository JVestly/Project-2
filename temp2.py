import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Generate XOR-like data ---
torch.manual_seed(0)
n_samples = 200
X = torch.randn(n_samples, 2)
y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)  # XOR pattern (1 if same sign)

# --- 2. Define model ---
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# --- 3. Helper: make meshgrid for visualization ---
def plot_decision_boundary(model, X, y, ax):
    # Create grid of points
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    # Get model predictions
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, preds, cmap='coolwarm', alpha=0.7)
    ax.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap='coolwarm', edgecolor='k')
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

# --- 4. Training and visualization ---
epochs = 300
snapshots = [0, 10, 50, 150, 300]
fig, axes = plt.subplots(1, len(snapshots), figsize=(15, 3))

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if epoch in snapshots:
        ax = axes[snapshots.index(epoch)]
        plot_decision_boundary(model, X, y, ax)
        ax.set_title(f"Epoch {epoch}\nLoss: {loss.item():.3f}")

plt.tight_layout()
plt.show()
