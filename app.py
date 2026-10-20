import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# 1. Simulate hourly power consumption data
np.random.seed(42)
date_rng = pd.date_range(start='2026-01-01', periods=720, freq='h')

# Convert hours to a NumPy array to ensure mutability
hours = date_rng.hour.to_numpy() 
base_load = 2.0 + 1.5 * np.sin(2 * np.pi * hours / 24)
noise = np.random.normal(0, 0.2, len(date_rng))
consumption = base_load + noise

# Inject artificial anomalies (now works because consumption is a mutable array)
consumption[150:155] = 8.5  # Sudden heavy spike
consumption[400:405] = 0.1  # Sudden drop (outage/tamper)

df = pd.DataFrame({'consumption': consumption}, index=date_rng)
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# 2. Preprocessing & Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['consumption', 'hour', 'dayofweek']])

# 3. Unsupervised Clustering: Grouping consumption behaviors
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_features)

# 4. Unsupervised Anomaly Detection: Finding unusual power profiles
iso_forest = IsolationForest(contamination=0.02, random_state=42)
df['anomaly'] = iso_forest.fit_predict(scaled_features) 

# Summary of detected anomalies
anomalies = df[df['anomaly'] == -1]
print(f"Total anomaly periods detected: {len(anomalies)}")
print(anomalies.head())