import numpy as np
import pandas as pd
from sklearn.cluster import Birch, DBSCAN, KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# || clustering

clusters = [
    (KMeans, dict()),
    (Birch, dict(n_clusters=5)),
    (DBSCAN, dict(n_jobs=-1)),
]

clustering_metrics = {
    'nmi': normalized_mutual_info_score,
    'ami': adjusted_mutual_info_score,
}


def clustering(x_train: np.ndarray, y_train: np.ndarray) -> pd.DataFrame:
    score = []
    for clu, args in clusters:
        cluster = clu(**args)
        y_pred = cluster.fit_predict(x_train)
        score.append({'cluster': clu.__name__})
        for name, metric in clustering_metrics.items():
            score[-1][name] = metric(y_train, y_pred)
    return pd.DataFrame(score)
