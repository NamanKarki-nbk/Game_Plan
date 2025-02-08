
import os
import cv2
import numpy as np
import torch
import pickle
import gzip
from torchvision.ops import nms
from transformers import AutoProcessor, SiglipVisionModel
from umap import UMAP
from sklearn.cluster import KMeans
from more_itertools import chunked
from typing import List

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
BATCH_SIZE = 32

class TeamAssigner:
    def __init__(self, device: str = 'cpu', video_path=None):
        self.device = device
        self.model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = UMAP(n_components=3)  
        self.clustering_model = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
        self.player_team_mapping = {}  # Stores track_id -> team mapping
        self.player_feature_cache = {}  # Stores track_id -> feature vector
        
        # Generate a unique stub path for team assignments based on the video filename
        if video_path:
            video_name = os.path.splitext(os.path.basename(video_path))[0]  
            self.stub_path = f"stubs/{video_name}_team_stubs.pkl.gz"
        else:
            self.stub_path = None

        self.load_team_assignments()  # Load previous team assignments if available

    def save_team_assignments(self):
        """Save team assignments and features to a stub file."""
        if self.stub_path:
            with gzip.open(self.stub_path, "wb") as f:
                pickle.dump(
                    {
                        "player_team_mapping": self.player_team_mapping,
                        "player_feature_cache": self.player_feature_cache
                    },
                    f
                )
            print(f"📂 Saved team assignments to {self.stub_path}")

    def load_team_assignments(self):
        """Load team assignments and features from a stub file if available."""
        if self.stub_path and os.path.exists(self.stub_path):
            with gzip.open(self.stub_path, "rb") as f:
                data = pickle.load(f)
                self.player_team_mapping = data.get("player_team_mapping", {})
                self.player_feature_cache = data.get("player_feature_cache", {})
            print(f"✅ Loaded team assignments from {self.stub_path}")
        else:
            print("⚠️ No previous team assignments found. Starting fresh.")

    def apply_nms(self, bboxes, scores, iou_threshold=0.5):
        """Applies Non-Maximum Suppression (NMS) to filter overlapping player detections."""
        if len(bboxes) == 0:
            return [], []
        
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep_indices = nms(bboxes_tensor, scores_tensor, iou_threshold)
        return [bboxes[i] for i in keep_indices], [scores[i] for i in keep_indices]

    def extract_player_crops(self, frame, player_bboxes, scores):
        """Extract the top half of player crops after applying NMS."""
        player_bboxes, scores = self.apply_nms(player_bboxes, scores, iou_threshold=0.5)
        crops = []
        
        for bbox in player_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            mid_y = y1 + (y2 - y1) // 2
            if mid_y > y1:
                top_half_crop = frame[y1:mid_y, x1:x2]
                if top_half_crop.shape[0] > 0 and top_half_crop.shape[1] > 0:
                    crops.append(top_half_crop)
        return crops

    def extract_features(self, player_ids: List[int], crops: List[np.ndarray]) -> np.ndarray:
        """Extract features only for new players and reuse existing ones."""
        features = []
        new_players = []

        for pid, crop in zip(player_ids, crops):
            if pid in self.player_feature_cache:
                features.append(self.player_feature_cache[pid])
            else:
                new_players.append((pid, crop))

        if new_players:
            new_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for _, crop in new_players]
            batches = chunked(new_crops, BATCH_SIZE)
            
            with torch.no_grad():
                for batch in batches:
                    inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                    
                    for (pid, _), feature in zip(new_players, embeddings):
                        self.player_feature_cache[pid] = feature
                        features.append(feature)

        return np.array(features) if features else np.array([])

    def reduce_dimensionality(self, features: np.ndarray) -> np.ndarray:
        """Reduce feature vectors to 3D using UMAP."""
        if features.shape[0] == 0:
            return np.array([])
        
        return self.reducer.fit_transform(features)

    def assign_teams_by_track_id(self, player_ids, reduced_features, reassign=False):
        """Assign stable team labels by persisting player assignments across frames."""
    
        if len(reduced_features) < 2:
            return np.array([])  # No clustering possible

        # Trim player_ids if they exceed feature count
        if len(player_ids) > len(reduced_features):
            print(f"⚠️ Mismatch: {len(player_ids)} player IDs but only {len(reduced_features)} feature vectors.")
            player_ids = player_ids[:len(reduced_features)]

        if reassign and not self.player_team_mapping:  # Only reassign if no saved assignment exists
            new_labels = self.clustering_model.fit_predict(reduced_features)
        
            # Maintain previous assignments for existing players
            for player_id, label in zip(player_ids, new_labels):
                self.player_team_mapping[player_id] = label

        # Assign labels based on stored team mapping
        assigned_labels = []
        for player_id in player_ids:
            if player_id in self.player_team_mapping:
                assigned_labels.append(self.player_team_mapping[player_id])
            else:
                # Assign a new player to the least occupied team for balance
                team_counts = np.bincount(list(self.player_team_mapping.values()), minlength=2)
                new_label = 0 if team_counts[0] <= team_counts[1] else 1
                self.player_team_mapping[player_id] = new_label
                assigned_labels.append(new_label)

        return np.array(assigned_labels)



