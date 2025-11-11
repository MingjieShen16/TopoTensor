import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceImage

import os
print("Current working directory:", os.getcwd())

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Changed working directory to script location:", os.getcwd())

# -------------------- 1) Load LOBSTER data --------------------
message_path = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
orderbook_path = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"

msg_cols = ["Time", "Type", "OrderID", "Size", "Price", "Direction"]
messages = pd.read_csv(message_path, names=msg_cols)

book_cols = []
for i in range(1,11):
    book_cols +=[f"Bid{i}_Price", f"Bid{i}_Vol"]

for i in range(1,11):
    book_cols +=[f"Ask{i}_Price", f"Ask{i}_Vol"]

orderbook = pd.read_csv(orderbook_path, names=book_cols)

assert len(messages) == len(orderbook), "Message and orderbook files misaligned"
print(f"Loaded {len(orderbook)} events for AMZN Level-10 book.")

# -------------------- 2) Compute inferred order flow --------------------
def infer_order_flow(book_t, book_t1):
    price_cols = [c for c in book_t.index if "Price" in c]
    vol_cols = [c for c in book_t.index if "Vol" in c]

    prices = book_t[price_cols].values
    vols_t = book_t[vol_cols].values
    vols_t1 = book_t1[vol_cols].values

    delta_vols = vols_t1 - vols_t

    mask = np.abs(delta_vols) > 1e-9
    #prices = prices[mask]
    #delta_vols_masked = delta_vols[mask]

    n_levels = len(prices)//2
    side_signs = np.array([1] * n_levels + [-1] * n_levels)[:len(prices)]
    #side_signs_masked = side_signs[mask]

    signed_dv = delta_vols*side_signs
    #signed_dv = delta_vols_masked * side_signs_masked

    return np.column_stack([prices,signed_dv])

# -------------------- 3) Build TopoTensor function  --------------------
from functions import build_topotensor_from_diagram

# -------------------- 4) Compute persistence diagrams --------------------
vr = VietorisRipsPersistence(homology_dimensions=[1])

tensors = []
metas = []

for i in range(0,len(orderbook)-1,50):
    pc = infer_order_flow(orderbook.iloc[i], orderbook.iloc[i+1])
    if len(pc) < 3:
        continue
    diagrams = vr.fit_transform([pc])
    pd = diagrams[0]
    if pd.shape[1]>=3:
        h1_points = pd[pd[:,2]==1][:,:2]
    else:
        h1_points = pd
    
    persists = h1_points[:,1] - h1_points[:,0]
    h1_points = h1_points[persists > 1e-9]

    tensor, meta =build_topotensor_from_diagram(h1_points)
    tensors.append(tensor)
    metas.append(meta)

print(f"Built {len(tensors)} TopoTensors from LOB snapshots.")

# -------------------- 5) Visualize one example --------------------
if tensors:
    example = tensors[0]
    meta = metas[0]

    n_channels = example.shape[0]
    if n_channels == 0:
        print(f"Example TopoTensor is empty (n_channels=0), meta={meta}")
    else:
        cols = min(6,n_channels)
        rows = int(np.ceil(n_channels/cols))
        plt.figure(figsize=(3*cols, 2.6*rows))
        for i in range(n_channels):
            plt.subplot(rows, cols, i+1)
            plt.imshow(example[i], origin='lower', interpolation='nearest')
            plt.title(f'Channel {i+1}')
            plt.axis('off')

    plt.suptitle(f"TopoTensor for LOB Flow (M_eff={meta['M_eff']}, n_clusters={meta['n_clusters']}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()
else:
    print("No valid persistennce diagrams found.")


