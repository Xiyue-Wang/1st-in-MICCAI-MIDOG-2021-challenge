import SimpleITK
from pathlib import Path

from pandas import DataFrame
import torch
import torchvision
from util.nms_WSI import nms

from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import evalutils
import json
from detection import UNetMitosisDetection
import os
import numpy as np
from numpy.linalg import norm



# TODO: We have this parameter to adapt the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
execute_in_docker = False

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


seed_torch(2021)

class Mitosisdetection(DetectionAlgorithm):
    def __init__(self, out_threshold):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),

                )
            ),
            input_path = Path("/input/") if execute_in_docker else Path("./test/"),
            output_file = Path("/output/mitotic-figures.json") if execute_in_docker else Path("./output/mitotic-figures.json")
        )
        # # TODO: This path should lead to your model weights
        if execute_in_docker:
           path_model = {
               'net1': '/opt/algorithm/checkpoints/fold1.pth'
           }
        else:
            path_model = {
                'net1': './fold1.pth'
            }

        print('Used model path', path_model)

        self.size = 512
        self.batchsize = 4
        self.out_threshold = out_threshold

        # TODO: You may adapt this to your model/algorithm here.
        self.md = UNetMitosisDetection(path_model, self.size, self.batchsize, self.out_threshold)



    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple points", points=scored_candidates, version={ "major": 1, "minor": 0 })

    # def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
    #     # Extract a numpy array with image data from the SimpleITK Image
    #     image_data = SimpleITK.GetArrayFromImage(input_image) # RGB
        
    #     # TODO: This is the part that you want to adapt to your submission.
    #     with torch.no_grad():
    #         boxes = self.md.process_image(image_data)

    #     candidates = list()
    #     for i, detection in enumerate(boxes):
    #         # our prediction returns x_1, y_1, x_2, y_2 -> transform to center coordinates
    #         x_1, y_1, x_2, y_2 = detection
    #         coord = tuple(((x_1 + x_2) / 2, (y_1 + y_2) / 2))

    #         # For the test set, we expect the coordinates in millimeters - this transformation ensures that the pixel
    #         # coordinates are transformed to mm - if resolution information is available in the .tiff image. If not,
    #         # pixel coordinates are returned.
    #         world_coords = input_image.TransformContinuousIndexToPhysicalPoint(
    #             [c for c in reversed(coord)]
    #         )
    #         candidates.append(tuple(reversed(world_coords)))

    #     # Note: We expect you to perform thresholding for your predictions. For evaluation, no additional thresholding
    #     # will be performed
    #     result = [{"point": [x, y, 0]} for x, y in candidates]
    #     return result

    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image) # RGB
        
        # TODO: This is the part that you want to adapt to your submission.
        with torch.no_grad():
            boxes = self.md.process_image(image_data)

        # add post-process to filter two cells of one pos case
        points = self.postprocess(boxes) 

        candidates = list()
        for i, detection in enumerate(points):
            # our prediction returns x_1, y_1, x_2, y_2 -> transform to center coordinates
            x_1, y_1 = detection
            coord = tuple((x_1, y_1))

            # For the test set, we expect the coordinates in millimeters - this transformation ensures that the pixel
            # coordinates are transformed to mm - if resolution information is available in the .tiff image. If not,
            # pixel coordinates are returned.
            world_coords = input_image.TransformContinuousIndexToPhysicalPoint(
                [c for c in reversed(coord)]
            )
            candidates.append(tuple(reversed(world_coords)))

        # Note: We expect you to perform thresholding for your predictions. For evaluation, no additional thresholding
        # will be performed
        result = [{"point": [x, y, 0]} for x, y in candidates]
        return result


    def postprocess(self, boxes, dist_threshold=30):
        '''
        Args:
            boxes[array]: (N, 4)
        Returns:
            all_points[array]: (N,2)
        '''
        if len(boxes) > 0:
            keep = np.ones(len(boxes)).astype(np.bool)
            points_x = (boxes[:,0] + boxes[:, 2]) / 2 # (N,)
            points_y = (boxes[:,1] + boxes[:, 3]) / 2 # (N,)
            points = np.concatenate([points_x[...,None], points_y[...,None]], 1) # (N,2)
            N = len(boxes)
            distance = np.zeros((N, N))
            for i in range(N):
                distance[i] = norm(points - points[i, :], axis=1) 
            for i in range(N):
                distance[i,i] = np.inf
        
            ids1, ids2 = [], []
            for i in range(N):
                for j in range(i+1, N):
                    if distance[i,j] < dist_threshold:
                        ids1.append(i)
                        ids2.append(j)

            new_centers = []
            for i in range(len(ids1)):
                keep[ids1[i]] = False
                keep[ids2[i]] = False
                point1 = points[ids1[i]]
                point2 = points[ids2[i]]
                new_center = (point1 + point2) / 2
                new_centers.append(new_center)
            
            new_centers = np.array(new_centers)
            keep_points = points[keep]
            if len(new_centers) > 0:
                all_points = np.concatenate([new_centers, keep_points], 0)
            else:
                all_points = keep_points
            return all_points
        else:
            return np.array([])


if __name__ == "__main__":
    # loads the image(s), applies DL detection model & saves the result
    Mitosisdetection(out_threshold=0.94).process()
