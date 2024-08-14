from scene.gaussian_model import GaussianModel
from utils.sh_utils import RGB2SH
import numpy as np
from torch import nn
from eval.openclip_encoder import OpenCLIPNetwork
from autoencoder.model import Autoencoder
import torch

class LangSplat_scene:
    def __init__(self, ply_path, feature_path, ae_ckpt_path, encoder_hidden_dims, decoder_hidden_dims):
        # Load Gaussian Model and point cloud
        self.gm = GaussianModel(3)
        self.gm.load_ply(ply_path)

        # Load and normalize language features
        lang_features = torch.from_numpy(np.load(feature_path))
        language_feature_norm = lang_features / (lang_features.norm(dim=-1, keepdim=True) + 1e-9)
        self.language_feature_final = language_feature_norm.reshape(language_feature_norm.shape[0], language_feature_norm.shape[1])

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CLIP model
        self.clip_model = OpenCLIPNetwork(self.device)

        # Initialize Autoencoder and load checkpoint
        self.model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(self.device)
        checkpoint = torch.load(ae_ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Decode visual features
        self.language_feature_final = self.language_feature_final.to(self.device)
        self.visual_feat = self.model.decode(self.language_feature_final)

        self.obj_center = []

    def query_3d(self, query_text):
        # Encode the query text into feature vector using CLIP model
        text_features = self.clip_model.encode_text(query_text, self.device).float()
        text_feature_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-9)

        # Calculate similarities between visual features and text feature
        similarities = torch.mm(self.visual_feat, text_feature_norm.T)
        threshold = 0.25
        matching_gaussians = similarities > threshold
        obj_center = torch.zeros(1, 3)

        if torch.any(matching_gaussians):
            # Recolor matching Gaussians
            colors = self.gm._features_dc.clone()
            gm_centers = self.gm.get_xyz.clone()
            obj_centers = gm_centers[matching_gaussians.squeeze()]
            obj_center = obj_centers.mean(dim=0)

            # Define the RGB color as a torch tensor
            rgb_tensor = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            sh_tensor = RGB2SH(rgb_tensor)
            colors[torch.tensor(matching_gaussians)] = sh_tensor.unsqueeze(0)

            # Update the Gaussian model with the new colors
            self.gm._features_dc = nn.Parameter(colors)

            # Save the updated point cloud
            self.gm.save_ply(f"colored_pc/teatime_3_{query_text}.ply")

        return obj_center


# # Example usage:
# ply_path = "output/teatime_3/point_cloud/iteration_30000/point_cloud.ply"
# feature_path = "lerf_ovs/teatime/teatime_3_features.npy"
# ae_ckpt_path = "autoencoder/ckpt/teatime/best_ckpt.pth"
# encoder_hidden_dims = [256, 128, 64, 32, 3]
# decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]
#
# LangSplat_model = LangSplat_scene(ply_path, feature_path, ae_ckpt_path, encoder_hidden_dims, decoder_hidden_dims)
# query_text = "a white bear"
# object_center = LangSplat_model.query_3d(query_text)
# print(object_center)



