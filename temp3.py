import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# --- 1. Make XOR-like data ---
torch.manual_seed(0)
n = 400
X = torch.randn(n, 2)
y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. Helper to plot decision boundary ---
def plot_decision_boundary(model, X, y, ax, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)
    ax.contourf(xx, yy, preds, cmap='coolwarm', alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='coolwarm', edgecolor='k', s=20)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

# --- 3. Try different hidden layer sizes ---
hidden_sizes = [4, 16, 64, 256]
fig, axes = plt.subplots(len(hidden_sizes), 2, figsize=(8, 10))

for i, h in enumerate(hidden_sizes):
    model = nn.Sequential(
        nn.Linear(2, h),
        nn.ReLU(),
        nn.Linear(h, 1),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # Train
    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        train_acc = ((model(X_train) > 0.5) == y_train).float().mean().item()
        test_acc = ((model(X_test) > 0.5) == y_test).float().mean().item()

    # Plot train/test boundaries
    plot_decision_boundary(model, X_train, y_train, axes[i, 0],
                           f"{h} neurons\nTrain acc={train_acc:.2f}")
    plot_decision_boundary(model, X_test, y_test, axes[i, 1],
                           f"{h} neurons\nTest acc={test_acc:.2f}")

axes[0, 0].set_ylabel("Training data")
axes[0, 1].set_ylabel("Testing data")
plt.suptitle("Overfitting vs Generalization (Train/Test Boundaries)", fontsize=14)
plt.tight_layout()
plt.show()
