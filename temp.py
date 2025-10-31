import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- 1. Generate toy data ---
torch.manual_seed(42)
X = torch.randn(100, 2)            # 100 samples, 2 features
y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)  # XOR-like pattern

# --- 2. Define the MLP ---
model = nn.Sequential(
    nn.Linear(2, 8),   # input → hidden
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# --- 3. Define loss and optimizer ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# --- 4. Train and record loss ---
losses = []
weight_snapshots = []

for epoch in range(300):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    # Save weights from first layer every 50 epochs
    if epoch % 50 == 0:
        W = model[0].weight.data.clone().numpy()
        weight_snapshots.append(W)

# --- 5. Plot loss curve ---
plt.figure(figsize=(6, 4))
plt.plot(losses, label="Training Loss", color='purple')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

# --- 6. Visualize how weights evolve ---
fig, axes = plt.subplots(1, len(weight_snapshots), figsize=(12, 3))
for i, W in enumerate(weight_snapshots):
    im = axes[i].imshow(W, cmap='coolwarm', aspect='auto')
    axes[i].set_title(f"Epoch {i*50}")
    axes[i].set_xlabel("Hidden units")
    axes[i].set_ylabel("Input features")
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.suptitle("Evolution of First Layer Weights")
plt.show()
