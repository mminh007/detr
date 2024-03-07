import torchvision
import os
from torchvision.datasets import CocoDetection

#url dataset


ANNOTATION_FILE_NAME = "_annotations.coco.json"

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path, image_processor, train = True):
        self.image_directory_path = image_directory_path
        annotation_path = os.path.join(self.image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(self.image_directory_path, annotation_path)
        self.image_processor = image_processor
    
    def __getitem__(self, id):
        images, annotations = super(CocoDetection, self).__getitem__(id)
        image_id = self.ids[id]
        annotations = {
            "image_id": image_id,
            "annotations": annotations,
        }
        encoding = self.image_processor(images = images, annotations = annotations, return_tensors = "pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        
        return pixel_values, target 


