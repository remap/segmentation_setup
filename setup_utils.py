import sys
from pathlib import Path
current_dir = str(Path(__file__).parent)+'/'
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
sys.path.insert(0, current_dir+'Detic/')
sys.path.insert(0, current_dir+'Detic/datasets/')
sys.path.insert(0, current_dir+'Detic/detic')
sys.path.insert(0, current_dir+'Detic/detic/modeling/utils')
sys.path.insert(0, current_dir+'Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test


def reset_cfg(cfg=None, segmentation_type='instance'):
    """
    Resets and returns a detectron2 configuration node based on the given segmentation type.

    Args:
        cfg (CfgNode, optional): A detectron2 configuration node. If None, a new configuration will be created.
        segmentation_type (str): The type of segmentation to use, either 'instance' or 'panoptic'.

    Returns:
        CfgNode: A configuration node initialized with the specified segmentation type.
    """
    global current_dir
    folder_path = current_dir+'Detic/'
    
    cfg_none_flag = False
    if cfg == None:
        cfg_none_flag = True
        cfg = get_cfg()

    if segmentation_type == 'panoptic':
        cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    else:
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(folder_path+"configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    
    if cfg_none_flag:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.

    return cfg

def reset_predictor(cfg=None, segmentation_type='instance', classifier=None, num_classes=None):
    """
    Resets and returns a predictor and configuration node based on the given segmentation type.

    Args:
        cfg (CfgNode, optional): A detectron2 configuration node. If None, a new configuration will be created.
        segmentation_type (str, optional): The type of segmentation to use, either 'instance' or 'panoptic'.
        classifier (optional): Classifier model for the segmentation task. Required if segmentation_type is not 'panoptic'.
        num_classes (int, optional): Number of classes for the segmentation task. Required if segmentation_type is not 'panoptic'.

    Returns:
        tuple: A tuple containing:
            - predictor (DefaultPredictor): The predictor object initialized with the specified configuration.
            - cfg (CfgNode): The configuration node used to initialize the predictor.
    """
    cfg = reset_cfg(segmentation_type=segmentation_type)
    predictor = DefaultPredictor(cfg)
    
    if segmentation_type != 'panoptic':
        assert classifier is not None and num_classes is not None
        reset_cls_test(predictor.model, classifier, num_classes)
    
    return predictor, cfg