import timm
import torch
from PIL import Image
from tqdm import tqdm
import time
from ipdb import set_trace

class ConvNextFeaturizer:
    ''' 
    Source:
    https://huggingface.co/timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k 
    '''
    def __init__(self, device='cuda'):
        """
        Initialize the ConvNeXT featurizer.
        """
        self.device = device
        self.model = timm.create_model(
            'convnext_xxlarge.clip_laion2b_soup_ft_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).to(device).eval()  

        # Get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.name = 'convnext'

    def preprocess_image(self, pil_image):
        """
        Preprocess the image for ConvNeXT.
        """
        return self.transforms(pil_image).unsqueeze(0).to(self.device)

    def preprocess_images(self, pil_images):
        """
        Preprocess list of images for ConvNeXT.
        """
        prep_imgs_list = []

        for img in tqdm(pil_images, desc='preproc images...'):
            prep_img = self.preprocess_image(img)
            prep_imgs_list.append(prep_img)

        prep_imgs_tensor = torch.cat(prep_imgs_list)
        return prep_imgs_tensor

    def encode_image(self, image, batch_size=32):
        """
        Encode a single image or a batch of images using the ConvNeXT model.
        """
        with torch.no_grad():
            features = torch.cat([
                self.model.forward_features(image[i:i + batch_size])
                for i in tqdm(range(0, image.size(0), batch_size),
                    desc="Encoding batches")
            ], dim=0)

            output = self.model.forward_head(features, pre_logits=True)

        return output

