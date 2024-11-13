from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from google.colab.patches import cv2_imshow
import cv2
from PIL import Image
import gif_utils # import save_frames_to_gif, show_gif
import setup_utils
from copy import deepcopy
import importlib
importlib.reload(gif_utils)
from icecream import ic
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, wait
import torch

class Segmentor:
    def __init__(self, predictor, cfg, metadata=None, panoptic=False, classifier=None, vocabulary=None) -> None:
        self.predictor = predictor
        self.cfg = cfg
        if (metadata is None) or panoptic:
            self.metadata = MetadataCatalog
        else:
            self.metadata = metadata
            self.num_classes = len(metadata.thing_classes)

        self.instance_metadata = deepcopy(metadata)
        self.panoptic = panoptic
        self.classifier = classifier
        self.vocabulary = vocabulary

    def reset_segmentation_type(self, segmentation_type='instance',
                                classifier=None,
                                vocabulary=None,
                                metadata=None):
        if segmentation_type == 'panoptic':
            self.panoptic = True
            self.metadata = MetadataCatalog
        else:
            self.panoptic = False
            self.metadata = deepcopy(self.instance_metadata)

        if metadata is not None:
          self.metadata = metadata
          self.num_classes = len(metadata.thing_classes)

        if vocabulary is not None:
          self.vocabulary = vocabulary

        if classifier is not None:
          self.classifier = classifier
        #   TODO: else build a new classifier based on CLIP and classes in metadata.thing_classes

        self.predictor, self.cfg = setup_utils.reset_predictor(self.cfg, segmentation_type=segmentation_type,
                                                                classifier=classifier, num_classes=self.num_classes)


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
            v = Visualizer(im[:, :, ::-1], self.metadata.get(self.predictor.cfg.DATASETS.TRAIN[0]), scale=1.2)
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
                # extract segmentation masks from panoptic predictor
                # Create a mask tensor with the same spatial dimensions as preds[0]
                frame_pan_mask = (preds[0].unsqueeze(0) == torch.arange(len(preds[1]), device=preds[0].device).view(-1, 1, 1)).to(torch.bool)
                # the line above does the same thing as:
                # frame_pan_mask = preds[0].new_zeros((len(preds[1]), preds[0].shape[0], preds[0].shape[1]), dtype=torch.bool)
                # for i in range(len(preds[1])):
                #     # create mask from panoptic predictor
                #     frame_pan_mask[i] = preds[0] == i

                # Append the result to mask_data
                mask_data.append((k, frame_pan_mask))
            else:    
                mask_data.append((k, preds.pred_masks))

        return masked_frames, frames_preds, mask_data
    

    def video_thread(self, frame_tuple, masked_frames_queue, frames_preds_queue, mask_data_queue):
        k, frame = frame_tuple
        preds, masked_frame = self.visualize_segmentation(frame, show_image=False)

        masked_frames_queue.put((k, Image.fromarray(masked_frame.get_image())))
        frames_preds_queue.put((k, preds))
        if self.panoptic:
            mask_data_queue.put((k, preds[0].pred_masks)) # extract segmentation masks from panoptic predictor
        else:    
            mask_data_queue.put((k, preds.pred_masks))


    def get_video_masks_multithread(self, frames=None, video_path=None, max_workers=10):

        # assert that not both frames and video are None
        assert not (frames is None and video_path is None), "Either frames or video must be provided"
        if video_path is not None:
            frames = gif_utils.extract_video_frames(video_path)

        
        frames_dict = {k: (k, frame) for k, frame in enumerate(frames)}
        masked_frames_queue = Queue()
        frames_preds_queue = Queue()
        mask_data_queue = Queue()

        mask_data = []
        frames_preds = []
        masked_frames = []

        def thread_safe_video_thread(frame_tuple):
            """Wrapper to handle exceptions in threads and ensure queue population."""
            try:
                self.video_thread(frame_tuple, masked_frames_queue, frames_preds_queue, mask_data_queue)
            except Exception as e:
                print(f"Error processing frame {frame_tuple[0]}: {e}")


        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(thread_safe_video_thread, frame_tuple) for frame_tuple in frames_dict.values()]

            for future in as_completed(futures):
                try:
                    future.result()  # Raises exception if the thread failed
                except Exception as e:
                    print(f"Thread exception: {e}")

        # Ensure queues have the same size
        if not (masked_frames_queue.qsize() == frames_preds_queue.qsize() == mask_data_queue.qsize()):
            raise ValueError(
                f"Queue size mismatch: masked_frames={masked_frames_queue.qsize()}, "
                f"frames_preds={frames_preds_queue.qsize()}, mask_data={mask_data_queue.qsize()}"
            )
        
        # Collect results
        masked_frames = list(masked_frames_queue.queue)
        frames_preds = list(frames_preds_queue.queue)
        mask_data = list(mask_data_queue.queue)
        return masked_frames, frames_preds, mask_data


class MaskObject:
    def __init__(self, frames, mask_data, frames_preds):
        self.mask_data = mask_data
        self.frame_preds = frames_preds
        self.frames = frames

        self.mask_objects = self.create_mask_objects() # create masks images

        self.mask_data_dict = {}
        for k in range(len(frames_preds)):
            self.mask_data_dict[k] = {
                "masks": mask_data[k][1],
                "bboxes": frames_preds[k][1].pred_boxes if hasattr(frames_preds[k][1], 'pred_boxes') else None,
                "mask_objects": self.mask_objects[k], # masks images
            }
        
    def create_mask_objects(self):
        mask_objects = []
        for k, frame in enumerate(self.frames):
            mask_objects.append(self.get_mask_objects(frame, frame_num=k))
        return mask_objects

    def get_mask_objects(self, frame=None, masks=None, frame_num=None):
        """
        Apply multiple masks on the frame simultaneously, returning a 4D tensor of masked images.

        Args:
        - masks (torch.Tensor): 3D tensor of shape (num_masks, height, width), where each 2D slice is a binary mask.
        - frame (torch.Tensor): 3D tensor of shape (height, width, channels), representing the original image.

        Returns:
        - masked_images (torch.Tensor): 4D tensor of shape (num_masks, height, width, channels), where each slice along
                                        the first dimension is the masked image for the corresponding mask.
        """
        if frame is None:
            assert frame_num is not None, "Either frame or frame_num must be provided"
            frame = self.frames[frame_num]
        # Ensure frame is a tensor
        frame = torch.tensor(frame).to(torch.float32) if not isinstance(frame, torch.Tensor) else frame

        if masks is None:
            assert frame_num is not None, "Either masks or frame_num must be provided"
            masks = self.mask_data[frame_num][1]

        # Ensure masks are binary (0 or 1)
        binary_masks = torch.where(masks == 1, 1, 0).unsqueeze(-1).cpu()  # Shape: (num_masks, height, width, 1)

        # Expand frame to match the mask dimensions for broadcasting
        frame_expanded = frame.unsqueeze(0)  # Shape: (1, height, width, channels)

        # Apply masks to the frame
        masked_images = binary_masks * frame_expanded  # Shape: (num_masks, height, width, channels)

        return masked_images
    
    def get_mask_from_point_location(self, frame_number, x, y):

        # Retrieve all masks for the frame
        masks_info = zip(
                self.mask_data_dict[frame_number]['masks'],
                self.mask_data_dict[frame_number]['bboxes'],
                self.mask_data_dict[frame_number]['mask_objects']
            )

        for k, (mask, bbox, mask_obj) in enumerate(masks_info):
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox#.tensor[0]
                # Check if point is within bounding box
                if not(x_min <= x <= x_max and y_min <= y <= y_max):
                    continue
            if mask[int(y), int(x)] == 1:
                return mask, mask_obj, bbox, k

        return None  # No mask found containing the point