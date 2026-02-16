import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=Warning)


# GAT, Graph Attention Networks

# 1. library
import torch

find_links = f"https://data.pyg.org/whl/torch-{torch.__version__}.html"

!pip install -q \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f $find_links

# seed set
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Installation Complete.")

# GAT, Graph Attention Networks

# 1. library
import torch

find_links = f"https://data.pyg.org/whl/torch-{torch.__version__}.html"

!pip install -q \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    -f $find_links

# seed set
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Installation Complete.")

# 1. Edge Feature Construction

# setup: Adjacent Matrix 
# Diagonal Symmetry means Undirected Graph

A = np.array([
    [1, 1, 1, 1], # 0 node
    [1, 1, 0, 0], # 1 node
    [1, 0, 1, 1], # 2 node
    [1, 0, 1, 1]  # 3 node
])
A

import networkx as nx

G = nx.from_numpy_array(A)

plt.figure(figsize=(3, 3))
nx.draw(G, 
        with_labels=True, 
        node_color='lightgreen', 
        node_size=500, 
        font_weight='bold', 
        edge_color='gray',
        linewidths=2)

plt.title("Graph from Adjacency Matrix with Self-loops")
plt.show()

# setup: node feature matrix
np.random.seed(1)

X = np.random.uniform(-1, 1, (4, 4)) #(low, high (nodes, features))
X


