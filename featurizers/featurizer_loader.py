from featurizers.clip_featurizer import ClipFeaturizer
from featurizers.convnext_featurizer import ConvNextFeaturizer
from featurizers.deit_iii_featurizer import DeiTIIIFeaturizer
from featurizers.dino_featurizer import DinoFeaturizer
from featurizers.dinov2_featurizer import DinoV2Featurizer
from featurizers.mae_featurizer import MaeFeaturizer
from featurizers.sam_featurizer import SAMFeaturizer
from featurizers.siglip_featurizer import SigLIPFeaturizer
from featurizers.dreamsim_featurizer import DreamSimFeaturizer

def load_featurizer(model_name):
    if model_name == 'clip':
        model = ClipFeaturizer()
    elif model_name == 'convnext':
        model = ConvNextFeaturizer()
    elif model_name == 'deit_iii':
        model = DeiTIIIFeaturizer()
    elif model_name == 'dino':
        model = DinoFeaturizer()
    elif model_name == 'dinov2':
        model = DinoV2Featurizer()
    elif model_name == 'dreamsim':
        model = DreamSimFeaturizer()
    elif model_name == 'mae':
        model = MaeFeaturizer()
    elif model_name == 'sam':
        model = SAMFeaturizer()
    elif model_name == 'siglip':
        model = SigLIPFeaturizer()
    else:
        raise ValueError(f"Model {model_name} not recognized")
    
    return model
