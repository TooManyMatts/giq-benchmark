'''
Source:
https://huggingface.co/timm/vit_base_patch32_224.sam_in1k
'''
import timm
from ipdb import set_trace
import torch
from PIL import Image
from tqdm import tqdm
import time

class SAMFeaturizer:
    def __init__(self, device='cuda'):
        """
        Initialize the SAM featurizer.
        """
        self.device = device
        self.model = timm.create_model(
            #'samvit_huge_patch16.sa1b',
            'vit_base_patch32_224.sam_in1k',
            pretrained=True,
            num_classes=0,  # Remove classifier nn.Linear
        ).to(device).eval()  

        # Get model-specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.name = 'sam'

    def preprocess_image(self, pil_image):
        """
        Preprocess the image for SAM.
        """
        return self.transforms(pil_image).unsqueeze(0).to(self.device)

    def preprocess_images(self, pil_images):
        """
        Preprocess list of images for SAM.
        """
        prep_imgs_list = []

        for img in tqdm(pil_images, desc='preproc images...'):
            prep_img = self.preprocess_image(img)
            prep_imgs_list.append(prep_img)

        prep_imgs_tensor = torch.cat(prep_imgs_list)
        return prep_imgs_tensor

    def encode_image(self, image, batch_size=32):
        """
        Encode a single image or a batch of images using the SAM model,
        performing forward pass and forward head for each batch to minimize memory usage.
        """
        outputs = []  # Store batch outputs
        
        with torch.no_grad():
            # Process the images in batches
            for i in tqdm(range(0, image.size(0), batch_size), desc="Encoding batches"):
                # Get batch of images
                batch = image[i:i + batch_size]
                
                # Perform forward pass for this batch
                features = self.model.forward_features(batch)
                
                # Perform forward through the classification head
                output = self.model.forward_head(features, pre_logits=True)
                
                # Store the result
                outputs.append(output)

        # Concatenate all outputs at the end
        return torch.cat(outputs, dim=0)
