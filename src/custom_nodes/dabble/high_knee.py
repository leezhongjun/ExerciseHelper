"""
Custom node to show keypoints and count the number of times the person's hand is waved
"""

from typing import Any, Dict, List, Tuple
import cv2
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

# setup global constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)       # opencv loads file in BGR format
YELLOW = (0, 255, 255)
THRESHOLD = 0.3               # ignore keypoints below this threshold
KP_RIGHT_SHOULDER = 6         # PoseNet's skeletal keypoints
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14

def map_bbox_to_image_coords(
   bbox: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """First helper function to convert relative bounding box coordinates to
   absolute image coordinates.
   Bounding box coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 4 floats x1, y1, x2, y2
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x1, y1, x2, y2 in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x1, y1, x2, y2 = bbox
   x1 *= width
   x2 *= width
   y1 *= height
   y2 *= height
   return int(x1), int(y1), int(x2), int(y2)


def map_keypoint_to_image_coords(
   keypoint: List[float], image_size: Tuple[int, int]
) -> List[int]:
   """Second helper function to convert relative keypoint coordinates to
   absolute image coordinates.
   Keypoint coords ranges from 0 to 1
   where (0, 0) = image top-left, (1, 1) = image bottom-right.

   Args:
      bbox (List[float]): List of 2 floats x, y (relative)
      image_size (Tuple[int, int]): Width, Height of image

   Returns:
      List[int]: x, y in integer image coords
   """
   width, height = image_size[0], image_size[1]
   x, y = keypoint
   x *= width
   y *= height
   return int(x), int(y)


def draw_text(img, x, y, text_str: str, color_code):
   """Helper function to call opencv's drawing function,
   to improve code readability in node's run() method.
   """
   cv2.putText(
      img=img,
      text=text_str,
      org=(x, y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.5,
      color=color_code,
      thickness=2,
   )


class Node(AbstractNode):
   """Custom node to display keypoints and count number of hand waves

   Args:
      config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
   """

   def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
      super().__init__(config, node_path=__name__, **kwargs)
      # setup object working variables
      self.right_wrist = None
      self.right_knee = None
      self.left_knee = None

      self.r_up = "down"
      self.l_up = "down"

      self.direction = None
      self.r_num_direction_changes = 0
      self.r_num_waves = 0
      self.l_num_direction_changes = 0
      self.l_num_waves = 0



   def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
      """This node draws keypoints and count hand waves.

      Args:
            inputs (dict): Dictionary with keys
               "img", "bboxes", "bbox_scores", "keypoints", "keypoint_scores".

      Returns:
            outputs (dict): Empty dictionary.
      """

      # get required inputs from pipeline
      img = inputs["img"]
      keypoints = inputs["keypoints"]
      keypoint_scores = inputs["keypoint_scores"]

      img_size = (img.shape[1], img.shape[0])  # image width, height

      
      


      if len(keypoints) == 0 or len(keypoint_scores) == 0:
         wave_str = f"high knees = {self.l_num_waves + self.r_num_waves}"
         draw_text(img, 20, 30, wave_str, YELLOW)
         return {}
      # hand wave detection using a simple heuristic of tracking the
      # right wrist movement
      the_keypoints = keypoints[0]              # image only has one person
      the_keypoint_scores = keypoint_scores[0]  # only one set of scores
      right_knee = None
      left_knee = None
      right_hip = None
      left_hip = None

      for i, keypoints in enumerate(the_keypoints):
         keypoint_score = the_keypoint_scores[i]

         if keypoint_score >= THRESHOLD:
            x, y = map_keypoint_to_image_coords(keypoints.tolist(), img_size)
            x_y_str = f"({x}, {y})"

            if i == KP_LEFT_HIP:
               left_hip = keypoints
               the_color = YELLOW
            elif i == KP_RIGHT_HIP:
               right_hip = keypoints
               the_color = YELLOW
            elif i == KP_RIGHT_KNEE:
               right_knee = keypoints
               the_color = YELLOW
            elif i == KP_LEFT_KNEE:
               left_knee = keypoints
               the_color = YELLOW
            else:                   # generic keypoint
               the_color = WHITE

            draw_text(img, x, y, x_y_str, the_color)

      if left_hip is not None and right_hip is not None\
         and left_knee is not None and right_knee is not None:
         
         if self.right_knee is None:
            self.right_knee = right_knee            # first knees data point

         if self.left_knee is None:
            self.left_knee = left_knee   

         #lower number - higher position
         if right_knee[1] < right_hip[1]:
            r_up = "up"
         else:
            r_up = "down"
         
         if r_up != self.r_up:
            self.r_num_direction_changes += 1
         if self.r_num_direction_changes>=2:
            self.r_num_waves += 1
            self.r_num_direction_changes = 0

         self.r_up = r_up

         if left_knee[1] < left_hip[1]:
            l_up = "up"
         else:
            l_up = "down"

         if l_up != self.l_up:
            self.l_num_direction_changes += 1
         if self.l_num_direction_changes>=2:
            self.l_num_waves += 1
            self.l_num_direction_changes = 0
            
         self.l_up = l_up
      wave_str = f"high knees = {self.l_num_waves + self.r_num_waves}"
      draw_text(img, 20, 30, wave_str, YELLOW)

      return {}