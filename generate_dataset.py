import os
import numpy as np
import pandas as pd

np.random.seed(42)

# --- Normal traffic (5 samples): low packets, longer duration, few SYN flags ---
normal = {
    "Packet_Count":   np.random.randint(50, 500, size=5),
    "Byte_Count":     np.random.uniform(1000, 10000, size=5),
    "Duration":       np.random.uniform(30, 300, size=5),
    "Source_Bytes":   np.random.uniform(500, 5000, size=5),
    "Dest_Bytes":     np.random.uniform(500, 5000, size=5),
    "Same_Srv_Rate":  np.random.uniform(0.7, 1.0, size=5),
    "Diff_Srv_Rate":  np.random.uniform(0.0, 0.2, size=5),
    "SYN_Flag_Count": np.random.randint(0, 10, size=5),
    "ACK_Flag_Count": np.random.randint(50, 200, size=5),
}

# --- DDoS attack traffic (5 samples): very high packets, short duration, high SYN ---
attack = {
    "Packet_Count":   np.random.randint(8000, 50000, size=5),
    "Byte_Count":     np.random.uniform(80000, 500000, size=5),
    "Duration":       np.random.uniform(0.01, 2.0, size=5),
    "Source_Bytes":   np.random.uniform(40000, 250000, size=5),
    "Dest_Bytes":     np.random.uniform(100, 1000, size=5),
    "Same_Srv_Rate":  np.random.uniform(0.9, 1.0, size=5),
    "Diff_Srv_Rate":  np.random.uniform(0.0, 0.05, size=5),
    "SYN_Flag_Count": np.random.randint(300, 1000, size=5),
    "ACK_Flag_Count": np.random.randint(0, 20, size=5),
}

df_normal = pd.DataFrame(normal)
df_normal["Label"] = "Normal"

df_attack = pd.DataFrame(attack)
df_attack["Label"] = "Attack"

df = pd.concat([df_normal, df_attack], ignore_index=True)

print(df.to_string(index=True))
print(f"\nShape: {df.shape}")

os.makedirs("data", exist_ok=True)
primary_output = os.path.join("data", "dummy_data.csv")
legacy_output = os.path.join("data", "dummy_dataset.csv")
df.to_csv(primary_output, index=False)
df.to_csv(legacy_output, index=False)
print(f"\nSaved to {primary_output}")
print(f"Saved to {legacy_output}")
