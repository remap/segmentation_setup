from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from google.colab.patches import cv2_imshow
import cv2
from PIL import Image
import gif_utils # import save_frames_to_gif, show_gif
from setup import reset_cfg, reset_predictor

class Segmenter:
    def __init__(self, predictor, cfg, metadata=None, panoptic=False) -> None:
        self.predictor = predictor
        self.cfg = cfg
        if metadata is None:
            self.metadata = MetadataCatalog
        self.panoptic = panoptic

    def reset_segmentation_type(self, segmentation_type='instance'):
        if segmentation_type == 'panoptic':
            self.panoptic = True
        else:
            self.panoptic = False

        self.predictor, self.cfg = reset_predictor(self.cfg, segmentation_type=segmentation_type)


    # def visualize_segmentation(predictor, Visualizer, metadata, im, panoptic=False, show_image=True):
    def visualize_segmentation(self, im, show_image=True):

        """
        Visualize segmentation predictions.

        Args:
            predictor (detectron2.engine.defaults.DefaultPredictor): Predictor object.
            Visualizer (detectron2.utils.visualizer.Visualizer): Visualizer object.
            metadata (detectron2.data.catalog.Metadata): Metadata object.
            im (ndarray): Input image.
            panoptic (bool): If True, visualizes panoptic segmentation.
            show_image (bool): If True, shows image with segmentation.

        Returns:
            predictions (detectron2.structures.instances.Instances): Predictions object.
        """
        
        pred = self.predictor(im)

        if self.panoptic:
            preds = pred["panoptic_seg"]
            panoptic_seg, segments_info = preds
            v = Visualizer(im[:, :, ::-1], self.metadata.get(pred.cfg.DATASETS.TRAIN[0]), scale=1.2)
            viz_out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

        else:
            preds = pred['instances']
            v = Visualizer(im[:,:,::-1], self.metadata)
            viz_out = v.draw_instance_predictions(preds.to('cpu'))

        if show_image:
            cv2_imshow(viz_out.get_image()[:, :, ::-1])

        return preds, viz_out



    # def get_video_masks(predictor, Visualizer, metadata, frames=None, video=None, segmentation_type='instance'):
    def get_video_masks(self, frames=None, video_path=None):

        # assert that not both frames and video are None
        assert not (frames is None and video_path is None), "Either frames or video must be provided"
        if video_path is not None:
            frames = gif_utils.extract_video_frames(video_path)

        # Process video to get masks
        mask_data = []
        frames_preds = []
        masked_frames = []
        for k,frame in enumerate(frames):
            preds, masked_frame = self.visualize_segmentation(frame, show_image=False)

            masked_frames.append(Image.fromarray(masked_frame.get_image()))
            frames_preds.append((k, preds))
            if self.panoptic:
                mask_data.append((k, preds[0].pred_masks)) # extract segmentation masks from panoptic predictor
            else:    
                mask_data.append((k, preds.pred_masks))

            return masked_frames, frames_preds, mask_data
    

# def show_gif(gif_path):
#     gif_utils.show_gif(gif_path)

# def save_frames_to_gif(gif_path, frames, frame_duration=40):
#     gif_utils.save_frames_to_gif(gif_path, frames, frame_duration=frame_duration)