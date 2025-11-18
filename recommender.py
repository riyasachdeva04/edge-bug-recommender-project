# recommender.py
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import faiss, os, pickle

class LinUCBBandit:
    def __init__(self, dim, alpha=1.0):
        self.alpha = alpha   # exploration factor
        self.A = np.eye(dim) # d x d matrix
        self.b = np.zeros((dim, 1)) # d x 1 vector

    def get_reward_estimate(self, x):
        A_inv = np.linalg.inv(self.A)
        theta = A_inv @ self.b
        mean = float(theta.T @ x)
        uncertainty = float(self.alpha * np.sqrt(x.T @ A_inv @ x))
        return mean + uncertainty

    def update(self, x, reward):
        self.A += x @ x.T
        self.b += reward * x


class BugRecommender:
    def __init__(self, data_path="data/bugs.csv", alpha=1.0):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.data = pd.read_csv(data_path)

        # Encode bug descriptions
        self.embs = self.model.encode(
            self.data["text"].tolist(), 
            convert_to_numpy=True
        ).astype(np.float32)

        faiss.normalize_L2(self.embs)

        # FAISS index for similarity search
        self.index = faiss.IndexFlatIP(self.embs.shape[1])
        self.index.add(self.embs)

        # RL contextual bandit
        self.bandit = LinUCBBandit(dim=self.embs.shape[1], alpha=alpha)

        # Load previous RL state if exists
        if os.path.exists("model_state_rl.pkl"):
            with open("model_state_rl.pkl", "rb") as f:
                self.bandit = pickle.load(f)

    def get_bug(self, bug_id: int):
        row = self.data.iloc[bug_id]
        return {
            "id": bug_id,
            "title": row.get("title", ""),
            "body": row.get("body", ""),
            "text": row.get("text", "")
        }

    def recommend_by_id(self, bug_id: int, topk=5):
        q_emb = self.embs[bug_id].reshape(-1, 1)  # column vector
        q_norm = q_emb.reshape(1, -1)
        faiss.normalize_L2(q_norm)

        # get similar items first
        _, I = self.index.search(q_norm, 20)

        scored = []
        for idx in I[0]:
            if idx == bug_id:
                continue

            x = self.embs[idx].reshape(-1, 1)
            score = self.bandit.get_reward_estimate(x)

            scored.append((score, idx))

        # pick top-K RL actions
        scored.sort(reverse=True)
        scored = scored[:topk]

        results = []
        for score, idx in scored:
            row = self.data.iloc[idx]
            results.append({
                "id": int(idx),
                "title": row.get("title", ""),
                "body": row.get("body", ""),
                "score": float(score)
            })
        return results

    def update_feedback(self, original_bug_id: int, rec_id: int, reward: int):
        x = self.embs[rec_id].reshape(-1, 1)

        # RL update
        self.bandit.update(x, reward)

        # Save state
        with open("model_state_rl.pkl", "wb") as f:
            pickle.dump(self.bandit, f)
