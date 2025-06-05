'''
Source:
https://huggingface.co/timm/vit_base_patch14_dinov2.lvd142m
'''
import timm
import torch
from PIL import Image
from tqdm import tqdm
import time

class DinoV2Featurizer:
    def __init__(self, device='cuda'):
        """
        Initialize the DINOv2 featurizer.
        """
        self.device = device
        self.model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).to(device).eval()  

        # Get model-specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.name = 'dinov2'

    def preprocess_image(self, pil_image):
        """
        Preprocess the image for DINOv2.
        """
        return self.transforms(pil_image).unsqueeze(0).to(self.device)

    def preprocess_images(self, pil_images):
        """
        Preprocess list of images for DINOv2.
        """
        prep_imgs_list = []

        for img in tqdm(pil_images, desc='preproc images...'):
            prep_img = self.preprocess_image(img)
            prep_imgs_list.append(prep_img)

        prep_imgs_tensor = torch.cat(prep_imgs_list)
        return prep_imgs_tensor

    def encode_image(self, image, batch_size=32):
        """
        Encode a single image or a batch of images using the DINOv2 model.
        """
        with torch.no_grad():
            # Process the images in batches and concatenate the features
            features = torch.cat([
                self.model.forward_features(image[i:i + batch_size])
                for i in tqdm(range(0, image.size(0), batch_size), desc="Encoding batches")
            ], dim=0)

            # Optionally, forward through the classification head (pre_logits=True gives feature embeddings)
            output = self.model.forward_head(features, pre_logits=True)

        return output
