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

class Segmenter:
    def __init__(self, predictor, cfg, metadata=None, panoptic=False) -> None:
        self.predictor = predictor
        self.cfg = cfg
        if (metadata is None) or panoptic:
            self.metadata = MetadataCatalog
        else:
            self.metadata = metadata
        self.instance_metadata = deepcopy(metadata)
        self.panoptic = panoptic

    def reset_segmentation_type(self, segmentation_type='instance'):
        if segmentation_type == 'panoptic':
            self.panoptic = True
            self.metadata = MetadataCatalog
        else:
            self.panoptic = False
            self.metadata = deepcopy(self.instance_metadata)

        self.predictor, self.cfg = setup_utils.reset_predictor(self.cfg, segmentation_type=segmentation_type)


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

        # TODO: implement multi-threading
        for k,frame in enumerate(frames):
            preds, masked_frame = self.visualize_segmentation(frame, show_image=False)

            masked_frames.append(Image.fromarray(masked_frame.get_image()))
            frames_preds.append((k, preds))
            if self.panoptic:
                mask_data.append((k, preds[0].pred_masks)) # extract segmentation masks from panoptic predictor
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
