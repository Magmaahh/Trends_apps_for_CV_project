import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import random
import sys

# Add project root to path to allow imports from data_preparation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_preparation.config import DATASET_OUTPUT, DEVICE
from data_preparation.model import VideoEmbeddingAdapter

MODELS_DIR = os.path.join(DATASET_OUTPUT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- DATASET ---
class TripletDataset(Dataset):
    def __init__(self, embeddings_dict, num_triplets=100000):
        """
        embeddings_dict: {speaker_id: [vector1, vector2, ...]}
        """
        self.embeddings_dict = embeddings_dict
        self.speakers = list(embeddings_dict.keys())
        self.num_triplets = num_triplets
        
    def __len__(self):
        return self.num_triplets
    
    def __getitem__(self, idx):
        # 1. Select Anchor Speaker
        anchor_spk = random.choice(self.speakers)
        
        # 2. Select Positive (same speaker)
        anchor_idx = random.randint(0, len(self.embeddings_dict[anchor_spk]) - 1)
        positive_idx = random.randint(0, len(self.embeddings_dict[anchor_spk]) - 1)
        
        anchor_vec = self.embeddings_dict[anchor_spk][anchor_idx]
        positive_vec = self.embeddings_dict[anchor_spk][positive_idx]
        
        # 3. Select Negative (different speaker)
        negative_spk = random.choice(self.speakers)
        while negative_spk == anchor_spk:
            negative_spk = random.choice(self.speakers)
            
        negative_idx = random.randint(0, len(self.embeddings_dict[negative_spk]) - 1)
        negative_vec = self.embeddings_dict[negative_spk][negative_idx]
        
        return (torch.tensor(anchor_vec, dtype=torch.float32),
                torch.tensor(positive_vec, dtype=torch.float32),
                torch.tensor(negative_vec, dtype=torch.float32))

def load_all_embeddings(limit_per_speaker=500):
    print("Loading embeddings...")
    data = {}
    speaker_dirs = glob.glob(os.path.join(DATASET_OUTPUT, "video_embeddings_s*"))
    
    for d in tqdm(speaker_dirs):
        speaker_id = os.path.basename(d).split("_")[-1] # video_embeddings_s1 -> s1
        npz_files = glob.glob(os.path.join(d, "*.npz"))
        
        if not npz_files: continue
        
        vectors = []
        # Load a subset of files to save memory/time
        random.shuffle(npz_files)
        for f in npz_files[:50]: # Load 50 files per speaker
            try:
                loaded = np.load(f)
                # Flatten all phoneme vectors into a single list
                for k in loaded.files:
                    vecs = loaded[k]
                    if len(vecs.shape) == 2:
                        vectors.extend(vecs)
            except:
                pass
        
        if vectors:
            # Limit total vectors per speaker
            if len(vectors) > limit_per_speaker:
                vectors = random.sample(vectors, limit_per_speaker)
            data[speaker_id] = vectors
            
    print(f"Loaded {len(data)} speakers.")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    # 1. Load Data
    embeddings_map = load_all_embeddings()
    if len(embeddings_map) < 2:
        print("Error: Need at least 2 speakers to train.")
        return

    dataset = TripletDataset(embeddings_map, num_triplets=50000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Setup Model
    model = VideoEmbeddingAdapter().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    
    # 3. Train
    print(f"Starting training for {args.epochs} epochs...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            
            optimizer.zero_grad()
            
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            
            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")
        
    # 4. Save
    save_path = os.path.join(MODELS_DIR, "adapter.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
