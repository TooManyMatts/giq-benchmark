'''
Source:
https://huggingface.co/timm/deit3_base_patch16_224.fb_in1k
'''
import timm
import torch
from PIL import Image
from tqdm import tqdm
import time

class DeiTIIIFeaturizer:
    def __init__(self, device='cuda'):
        """
        Initialize the DeiT III featurizer.
        """
        self.device = device
        self.model = timm.create_model(
            'deit3_base_patch16_224.fb_in1k',
            pretrained=True,
            num_classes=0,  # Remove classifier nn.Linear
        ).to(device).eval()  

        # Get model-specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.name = 'deit_iii'

    def preprocess_image(self, pil_image):
        """
        Preprocess the image for DeiT III.
        """
        return self.transforms(pil_image).unsqueeze(0).to(self.device)

    def preprocess_images(self, pil_images):
        """
        Preprocess list of images for DeiT III.
        """
        prep_imgs_list = []

        for img in tqdm(pil_images, desc='preproc images...'):
            prep_img = self.preprocess_image(img)
            prep_imgs_list.append(prep_img)

        prep_imgs_tensor = torch.cat(prep_imgs_list)
        return prep_imgs_tensor

    def encode_image(self, image):
        """
        Encode a single image or a batch of images using the DeiT III model.
        """
        with torch.no_grad():
            # The model.forward_features method outputs the intermediate features
            features = self.model.forward_features(image)
            # Optionally, forward through the classification head (pre_logits=True gives feature embeddings)
            output = self.model.forward_head(features, pre_logits=True)
        return output

