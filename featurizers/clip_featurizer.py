import clip
import torch
from PIL import Image
from tqdm import tqdm

class ClipFeaturizer:
    def __init__(self, device='cuda'):
        self.device = 'cuda'
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.name = 'clip'        

    def load_model(self):
        """
        Load the CLIP model.
        """
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def preprocess_image(self, pil_image):
        """
        Preprocess the image for CLIP.
        """
        return self.preprocess(pil_image).to(self.device)

    def preprocess_images(self, pil_images):
        """
        Preprocess list of images for CLIP.
        """
        prep_imgs_list = []

        for img in tqdm(pil_images,desc='preproc images...'):
            prep_img = self.preprocess_image(img)
            prep_imgs_list.append(prep_img)

        prep_imgs_tensor = torch.stack(prep_imgs_list)
        return prep_imgs_tensor

    def encode_image(self, image):
        """
        Encode a single image or a batch of images using the CLIP model.
        """
        with torch.no_grad():
            return self.model.encode_image(image)


