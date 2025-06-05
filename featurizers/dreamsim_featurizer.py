'''
Source:
https://github.com/ssundaram21/dreamsim
'''
import torch
from PIL import Image
from tqdm import tqdm
import time
from dreamsim import dreamsim  

class DreamSimFeaturizer:
    def __init__(self, device='cuda'):
        """
        Initialize the DreamSim featurizer.
        """
        self.device = device
        self.model, self.preprocess = dreamsim(pretrained=True)
        self.model.to(device).eval()  
        self.name = 'dreamsim'

    def preprocess_image(self, pil_image):
        """
        Preprocess the image for DreamSim.
        """
        return self.preprocess(pil_image).to(self.device)

    def preprocess_images(self, pil_images):
        """
        Preprocess list of images for DreamSim.
        """
        prep_imgs_list = []

        for img in tqdm(pil_images, desc='preproc images...'):
            prep_img = self.preprocess_image(img)
            prep_imgs_list.append(prep_img)

        prep_imgs_tensor = torch.cat(prep_imgs_list)
        return prep_imgs_tensor

    def encode_image(self, image, batch_size=32):
        """
        Encode a single image or a batch of images using the DreamSim model.
        """
        outputs = []  # Store batch outputs

        with torch.no_grad():
            # Process the images in batches
            for i in tqdm(range(0, image.size(0), batch_size), desc="Encoding batches"):
                # Get batch of images
                batch = image[i:i + batch_size]

                # Perform embedding for this batch
                embedding = self.model.embed(batch)

                # Store the result
                outputs.append(embedding)

        # Concatenate all outputs at the end
        return torch.cat(outputs, dim=0)

