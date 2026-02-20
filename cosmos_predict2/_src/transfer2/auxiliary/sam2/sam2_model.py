# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import numpy as np
import pycocotools.mask as mask_util
import torch
from PIL import Image

from cosmos_predict2._src.imaginaire.flags import INTERNAL

if not INTERNAL:
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    from cosmos_predict2._src.transfer2.auxiliary.sam2.sam2_utils import (
        capture_fps,
        convert_masks_to_frames,
        generate_tensor_from_images,
        video_to_frames,
        write_video,
    )

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

SAM2_MODEL_CHECKPOINT = "facebook/sam2-hiera-large"
GROUNDING_DINO_MODEL_CHECKPOINT = "IDEA-Research/grounding-dino-base"


def rle_encode(mask: np.ndarray) -> dict:
    """
    Encode a boolean mask (of shape (T, H, W)) using the pycocotools RLE format,
    matching the format of eff_segmentation.RleMaskSAMv2 (from Yotta).

    The procedure is:
      1. Convert the mask to a numpy array in Fortran order.
      2. Reshape the array to (-1, 1) (i.e. flatten in Fortran order).
      3. Call pycocotools.mask.encode on the reshaped array.
      4. Return a dictionary with the encoded data and the original mask shape.
    """
    mask = np.array(mask, order="F")
    # Reshape the mask to (-1, 1) in Fortran order and encode it.
    encoded = mask_util.encode(np.array(mask.reshape(-1, 1), order="F"))
    return {"data": encoded, "mask_shape": mask.shape}


class VideoSegmentationModel:
    def __init__(self, **kwargs):
        """Initialize the model and load all required components."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize SAM2 predictor
        self.sam2_predictor = SAM2VideoPredictor.from_pretrained(SAM2_MODEL_CHECKPOINT).to(self.device)

        # Initialize GroundingDINO for text-based detection
        self.grounding_model_name = kwargs.get("grounding_model", GROUNDING_DINO_MODEL_CHECKPOINT)
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_name)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model_name).to(
            self.device
        )

    def get_boxes_from_text(self, image_path, text_prompt):
        """Get bounding boxes (and labels) from a text prompt using GroundingDINO."""
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        # Try with initial thresholds.
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.15,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )

        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0].get("labels", None)
        if len(boxes) == 0:
            print(f"No boxes detected for prompt: '{text_prompt}'. Trying with lower thresholds...")
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.1,
                text_threshold=0.1,
                target_sizes=[image.size[::-1]],
            )
            boxes = results[0]["boxes"].cpu().numpy()
            scores = results[0]["scores"].cpu().numpy()
            labels = results[0].get("labels", None)

        if len(boxes) > 0:
            print(f"Found {len(boxes)} boxes with scores: {scores}")
            # Sort boxes by confidence score in descending order
            sorted_indices = np.argsort(scores)[::-1]
            boxes = boxes[sorted_indices]
            scores = scores[sorted_indices]
            if labels is not None:
                labels = np.array(labels)[sorted_indices]
        else:
            print("Still no boxes detected. Consider adjusting the prompt or using box/points mode.")

        return {"boxes": boxes, "labels": labels, "scores": scores}

    def visualize_frame(self, frame_idx, obj_ids, masks, video_dir, frame_names, visualization_data, save_dir=None):
        """
        Process a single frame: load the image, apply the segmentation mask to black out the
        detected object(s), and save both the masked frame and the binary mask image.
        """
        # Load the frame.
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        img = Image.open(frame_path).convert("RGB")
        image_np = np.array(img)

        # Combine masks from the detection output.
        if isinstance(masks, torch.Tensor):
            mask_np = (masks[0] > 0.0).cpu().numpy().astype(bool)
            combined_mask = mask_np
        elif isinstance(masks, dict):
            first_mask = next(iter(masks.values()))
            combined_mask = np.zeros_like(first_mask, dtype=bool)
            for m in masks.values():
                combined_mask |= m
        else:
            combined_mask = None

        if combined_mask is not None:
            combined_mask = np.squeeze(combined_mask)

            # If the mask shape doesn't match the image, resize it.
            if combined_mask.shape != image_np.shape[:2]:
                mask_img = Image.fromarray((combined_mask.astype(np.uint8)) * 255)
                mask_img = mask_img.resize((image_np.shape[1], image_np.shape[0]), resample=Image.NEAREST)
                combined_mask = np.array(mask_img) > 127

            # Black out the detected region.
            image_np[combined_mask] = 0

            mask_image = (combined_mask.astype(np.uint8)) * 255
            mask_pil = Image.fromarray(mask_image)

        if save_dir:
            seg_frame_path = os.path.join(save_dir, f"frame_{frame_idx}_segmented.png")
            seg_pil = Image.fromarray(image_np)
            seg_pil.save(seg_frame_path)
            if combined_mask is not None:
                mask_save_path = os.path.join(save_dir, f"frame_{frame_idx}_mask.png")
                mask_pil.save(mask_save_path)

    def sample(self, **kwargs):
        """
        Main sampling function for video segmentation.
        Returns a list of detections in which each detection contains a phrase and
        an RLE-encoded segmentation mask (matching the output of the Grounded SAM model).
        """
        video_dir = kwargs.get("video_dir", "")
        mode = kwargs.get("mode", "points")
        input_data = kwargs.get("input_data", None)
        save_dir = kwargs.get("save_dir", None)
        visualize = kwargs.get("visualize", False)

        # Get frame names (expecting frames named as numbers with .jpg/.jpeg extension).
        frame_names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.sam2_predictor.init_state(video_path=video_dir)

            ann_frame_idx = 0
            ann_obj_id = 1
            boxes = None
            points = None
            labels = None
            box = None

            visualization_data = {"mode": mode, "points": None, "labels": None, "box": None, "boxes": None}

            if input_data is not None:
                if mode == "points":
                    points = input_data.get("points")
                    labels = input_data.get("labels")
                    frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
                    )
                    visualization_data["points"] = points
                    visualization_data["labels"] = labels
                elif mode == "box":
                    box = input_data.get("box")
                    frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=box
                    )
                    visualization_data["box"] = box
                elif mode == "prompt":
                    text = input_data.get("text")
                    first_frame_path = os.path.join(video_dir, frame_names[0])
                    gd_results = self.get_boxes_from_text(first_frame_path, text)
                    boxes = gd_results["boxes"]
                    labels_out = gd_results["labels"]
                    scores = gd_results["scores"]
                    print(f"scores: {scores}")
                    if len(boxes) > 0:
                        legacy_mask = kwargs.get("legacy_mask", False)
                        if legacy_mask:
                            # Use only the highest confidence box for legacy mask
                            print(f"using legacy_mask: {legacy_mask}")
                            frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                                inference_state=state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, box=boxes[0]
                            )
                            # Update boxes and labels after processing
                            boxes = boxes[:1]
                            if labels_out is not None:
                                labels_out = labels_out[:1]
                        else:
                            print(f"using new_mask: {legacy_mask}")
                            for object_id, (box, label) in enumerate(zip(boxes, labels_out)):
                                frame_idx, obj_ids, masks = self.sam2_predictor.add_new_points_or_box(
                                    inference_state=state, frame_idx=ann_frame_idx, obj_id=object_id, box=box
                                )
                        visualization_data["boxes"] = boxes
                        self.grounding_labels = [str(lbl) for lbl in labels_out] if labels_out is not None else [text]
                    else:
                        print("No boxes detected. Exiting.")
                        return []  # Return empty list if no detections

                if visualize:
                    self.visualize_frame(
                        frame_idx=ann_frame_idx,
                        obj_ids=obj_ids,
                        masks=masks,
                        video_dir=video_dir,
                        frame_names=frame_names,
                        visualization_data=visualization_data,
                        save_dir=save_dir,
                    )

            video_segments = {}  # keys: frame index, values: {obj_id: mask}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                }

                # For propagated frames, visualization_data is not used.
                if visualize:
                    propagate_visualization_data = {
                        "mode": mode,
                        "points": None,
                        "labels": None,
                        "box": None,
                        "boxes": None,
                    }
                    self.visualize_frame(
                        frame_idx=out_frame_idx,
                        obj_ids=out_obj_ids,
                        masks=video_segments[out_frame_idx],
                        video_dir=video_dir,
                        frame_names=frame_names,
                        visualization_data=propagate_visualization_data,
                        save_dir=save_dir,
                    )

        # --- Post-process video_segments to produce a list of detections ---
        if len(video_segments) == 0:
            return []

        first_frame_path = os.path.join(video_dir, frame_names[0])
        first_frame = np.array(Image.open(first_frame_path).convert("RGB"))
        original_shape = first_frame.shape[:2]  # (height, width)

        object_masks = {}  # key: obj_id, value: list of 2D boolean masks
        sorted_frame_indices = sorted(video_segments.keys())
        for frame_idx in sorted_frame_indices:
            segments = video_segments[frame_idx]
            for obj_id, mask in segments.items():
                mask = np.squeeze(mask)
                if mask.ndim != 2:
                    print(f"Warning: Unexpected mask shape {mask.shape} for object {obj_id} in frame {frame_idx}.")
                    continue

                if mask.shape != original_shape:
                    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                    mask_img = mask_img.resize((original_shape[1], original_shape[0]), resample=Image.NEAREST)
                    mask = np.array(mask_img) > 127

                if obj_id not in object_masks:
                    object_masks[obj_id] = []
                object_masks[obj_id].append(mask)

        detections = []
        for obj_id, mask_list in object_masks.items():
            mask_stack = np.stack(mask_list, axis=0)  # shape: (T, H, W)
            # Use our new rle_encode (which now follows the eff_segmentation.RleMaskSAMv2 format)
            rle = rle_encode(mask_stack)
            if mode == "prompt" and hasattr(self, "grounding_labels"):
                phrase = self.grounding_labels[0]
            else:
                phrase = input_data.get("text", "")
            detection = {"phrase": phrase, "segmentation_mask_rle": rle}
            detections.append(detection)

        return detections

    @staticmethod
    def parse_points(points_str):
        """Parse a string of points into a numpy array.
        Supports a single point ('200,300') or multiple points separated by ';' (e.g., '200,300;100,150').
        """
        points = []
        for point in points_str.split(";"):
            coords = point.split(",")
            if len(coords) != 2:
                continue
            points.append([float(coords[0]), float(coords[1])])
        return np.array(points, dtype=np.float32)

    @staticmethod
    def parse_labels(labels_str):
        """Parse a comma-separated string of labels into a numpy array."""
        return np.array([int(x) for x in labels_str.split(",")], dtype=np.int32)

    @staticmethod
    def parse_box(box_str):
        """Parse a comma-separated string of 4 box coordinates into a numpy array."""
        return np.array([float(x) for x in box_str.split(",")], dtype=np.float32)

    def __call__(
        self,
        input_video,
        output_video=None,
        output_tensor=None,
        prompt=None,
        box=None,
        points=None,
        labels=None,
        weight_scaler=None,
        binarize_video=False,
        legacy_mask=False,
    ):
        print(
            f"Processing video: {input_video} to generate segmentation video: {output_video} segmentation tensor: {output_tensor}"
        )
        assert os.path.exists(input_video)

        # Prepare input data based on the selected mode.
        if points is not None:
            mode = "points"
            input_data = {"points": self.parse_points(points), "labels": self.parse_labels(labels)}
        elif box is not None:
            mode = "box"
            input_data = {"box": self.parse_box(box)}
        elif prompt is not None:
            mode = "prompt"
            input_data = {"text": prompt}

        with tempfile.TemporaryDirectory() as temp_input_dir:
            fps = capture_fps(input_video)
            video_to_frames(input_video, temp_input_dir)
            with tempfile.TemporaryDirectory() as temp_output_dir:
                masks = self.sample(
                    video_dir=temp_input_dir,
                    mode=mode,
                    input_data=input_data,
                    save_dir=str(temp_output_dir),
                    visualize=True,
                    legacy_mask=legacy_mask,
                )
                if output_video:
                    os.makedirs(os.path.dirname(output_video), exist_ok=True)
                    frames = convert_masks_to_frames(masks)
                    if binarize_video:
                        frames = np.any(frames > 0, axis=-1).astype(np.uint8) * 255
                    write_video(frames, output_video, fps)
                if output_tensor:
                    generate_tensor_from_images(
                        temp_output_dir, output_tensor, fps, "mask", weight_scaler=weight_scaler
                    )
