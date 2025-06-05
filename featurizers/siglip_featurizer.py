'''
Source:
https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384
'''
import timm
import torch
from PIL import Image
from tqdm import tqdm
import time

class SigLIPFeaturizer:
    def __init__(self, device='cuda'):
        """
        Initialize the SigLIP featurizer.
        """
        self.device = device
        self.model = timm.create_model(
            'vit_so400m_patch14_siglip_384',
            pretrained=True,
            num_classes=0,  # Remove classifier nn.Linear
        ).to(device).eval() 

        # Get model-specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.name = 'siglip'

    def preprocess_image(self, pil_image):
        """
        Preprocess the image for SigLIP.
        """
        return self.transforms(pil_image).unsqueeze(0).to(self.device)

    def preprocess_images(self, pil_images):
        """
        Preprocess list of images for SigLIP.
        """
        prep_imgs_list = []

        for img in tqdm(pil_images, desc='preproc images...'):
            prep_img = self.preprocess_image(img)
            prep_imgs_list.append(prep_img)

        prep_imgs_tensor = torch.cat(prep_imgs_list)
        return prep_imgs_tensor

    def encode_image(self, image, batch_size=32):
        """
        Encode a single image or a batch of images using the SigLIP model.
        """
        with torch.no_grad():
            # Process the images in batches and concatenate the output
            output = torch.cat([
                self.model(image[i:i + batch_size])
                for i in tqdm(range(0, image.size(0), batch_size), desc="Encoding batches")
            ], dim=0)

        return output

