# train_bandit.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle, os, random


CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",

    # LinUCB hyperparameters
    "alpha": 1.5,        
    "lambda_reg": 1.0,  

    # Reward processing
    "reward_scale": 1.0,        
    "normalize_reward": True,    # map reward to [0,1]

    # Training
    "epochs": 3,          # Offline RL passes over the dataset
    "shuffle_each_epoch": True,

    # Embeddings
    "normalize_embeddings": True,

    # File output
    "save_path": "model_state_rl.pkl"
}


class LinUCBBandit:
    def __init__(self, dim, alpha=1.0, lambda_reg=1.0):
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # A = λI (regularization)
        self.A = lambda_reg * np.eye(dim)     # d×d matrix
        self.b = np.zeros((dim, 1))           # d×1 vector

    def update(self, x, reward):
        """Standard LinUCB parameter update."""
        self.A += x @ x.T
        self.b += reward * x

    def predict_theta(self):
        """Return learned parameter vector."""
        return np.linalg.inv(self.A) @ self.b



def scale_reward(r, normalize=True, scale=1.0):
    """Optionally normalize reward to [0,1] and scale."""
    if normalize:
        # Min max scaling
        r_min, r_max = 0.0, 5.0 if r > 1 else 1.0
        r = (r - r_min) / (r_max - r_min + 1e-8)

    return r * scale


def train_bandit(csv_path="data/bugs.csv"):
    print("\n============= TRAINING LINUCB BANDIT =============\n")

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "relevance_score" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'relevance_score' columns")

    print("📌 Loaded dataset with", len(df), "bugs")

    print("📌 Loading SentenceTransformer model:", CONFIG["embedding_model"])
    embedder = SentenceTransformer(CONFIG["embedding_model"])

    print("📌 Encoding text...")
    embs = embedder.encode(df["text"].tolist(), convert_to_numpy=True).astype(np.float32)

    if CONFIG["normalize_embeddings"]:
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms

    dim = embs.shape[1]
    print("📌 Embedding dimension:", dim)

    bandit = LinUCBBandit(
        dim=dim,
        alpha=CONFIG["alpha"],
        lambda_reg=CONFIG["lambda_reg"],
    )

    print("\n📌 Starting training...")
    for epoch in range(CONFIG["epochs"]):
        print(f"\n🔄 Epoch {epoch+1}/{CONFIG['epochs']}")

        indices = list(range(len(df)))
        if CONFIG["shuffle_each_epoch"]:
            random.shuffle(indices)

        for i in indices:
            x = embs[i].reshape(-1, 1)
            reward = scale_reward(
                df.iloc[i]["relevance_score"],
                normalize=CONFIG["normalize_reward"],
                scale=CONFIG["reward_scale"]
            )
            bandit.update(x, reward)

    print("\n📌 Training completed.")

    with open(CONFIG["save_path"], "wb") as f:
        pickle.dump(bandit, f)

    print(f"\n📦 Saved trained RL state → {CONFIG['save_path']}")
    print("\n==================================================\n")


if __name__ == "__main__":
    train_bandit()
