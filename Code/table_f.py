import pandas as pd

df = pd.DataFrame({
    "Activation function": ["Sigmoid", "reLU", "Leaky reLU", "ELU", "GELU", "tanh"],
    "Optimizer": ["RMS", "ADAM", "ADAM", "RMS", "RMS", "ADAM"],
    "Accuracy": ["0.977","0.977", "0.98", "0.977", "0.975", "0.964"],
    "Time (s)": ["6.22s", "4.59s", "3.38s", "7.93s", "4.85s", "1.51s"],
    "(Hidden, nodes)": ["(3,64)", "(2,128)", "(2,64)", "(2,128)", "(2,128)", "(1,64)"]
})

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 2))
ax.axis("off")

# Create table
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 loc="center")

# Center all text horizontally and vertically
for key, cell in table.get_celld().items():
    cell.set_text_props(ha='center', va='center')

import os

save_dir = os.path.join(os.path.dirname(os.getcwd()), "Figures")
save_path = os.path.join(save_dir, "Act_OPTClass.png")
fig.savefig(save_path, dpi=300, bbox_inches='tight')