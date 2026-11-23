"""Item text -> fixed vector (frozen encoder). Falls back to TF-IDF + SVD."""
from __future__ import annotations

import numpy as np

from ..utils import log


def embed_texts(texts: list[str], model_name: str, device: str = "cuda",
                batch_size: int = 256) -> np.ndarray:
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        model = SentenceTransformer(model_name, device=dev)
        X = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
        log(f"[text] encoded {len(texts):,} texts with {model_name} -> {X.shape}")
        return X.astype(np.float32)
    except Exception as e:
        log(f"[text] sentence-transformers unavailable ({e!r}); using TF-IDF+SVD")
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        tf = TfidfVectorizer(max_features=50000, min_df=2, sublinear_tf=True)
        M = tf.fit_transform(texts)
        k = max(2, min(128, M.shape[1] - 1))
        X = TruncatedSVD(n_components=k, random_state=0).fit_transform(M)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X.astype(np.float32)
