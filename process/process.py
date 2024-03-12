import json
from pytorch_lightning.callbacks import Callback
import torch
import os
import cv2

def extract_label(json_path):
    """
    get label from file.json
    Retrieve label from file.json. Preview the storage file to modify the function accordingly 

    Args:
        json_path (_type_): _description_
    """
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    cats = {id: x for id,x in enumerate(data["categories"])}
    
    id2label = {id: v["name"] for id, v in enumerate(cats.values())}
    
    
    print(id2label)

class LogPredictionsCallback(Callback):
    
    def __init__(self, val_dataset, id2label, label2id, wandb_logger, image_processor, confidence_threshold, data_dir):
        self.val_dataset = val_dataset
        self.id2label = id2label
        self.label2id = label2id
        self.wandb_logger = wandb_logger
        self.image_processor = image_processor
        self.confidence_threshold = confidence_threshold
        self.data_dir = data_dir
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called when the validation batch ends.
        Access 'https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py' 
                'https://github.com/pytorch/vision/blob/main/torchvision/datasets/coco.py'        to learn more about the parameters of the CoCo dataset.
                
        wandb logger: More arguments can be passed for logging segmentation masks and bounding boxes, Refer  https://docs.wandb.ai/guides/track/log/media#image-overlays
                                                                                        https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
        """
    
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            # extract imgage from validation dataset
            labels = [item for item in batch["labels"]]
           
            # extract predictions
            sizes = torch.stack([target["orig_size"] for target in labels], dim = 0)
            results = self.image_processor.post_process_object_detection(
                                                                outputs=outputs, 
                                                                threshold=self.confidence_threshold, 
                                                                target_sizes=sizes
                                                                ) # -> list of dictionaries
            
            preds = {target["image_id"].item(): output for target, output in zip(labels, results)}
            preds = self._convert_to_coco_detection(preds)
            
            list_images = [] 
            list_caption = [f"sample_{idx}" for idx in range(len(list_images))]
            for idx, pred in enumerate(preds):
                image_id = pred["image_id"].item()
                path = self.val_dataset.coco.loadImgs(image_id)[0]["file_name"]
                image = cv2.imread(os.path.join(self.data_dir, path), cv2.COLOR_BGR2RGB)
                
                bbox = pred["bbox"].item()
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                score = pred["scores"].item()
                label = pred["category_id"].item()
                
                name = f"{label}: {score}"
                
                #draw bbox, label and score in image 
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=2)
                cv2.putText(image, name, (xmin, ymax), font, font_scale, color = (0,0,0))
                
                # Save image
                file_name = f"/content/output/pred_{path}"
                cv2.imwrite(file_name, image)
                list_images.append(file_name)
                
            # Option 1: log images with `WandbLogger.log_image`
            self.wandb_logger.log_image(key='sample_images', images=list_images, caption=list_caption)  
       
                
            
    def _convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim = 1)
    
    def _convert_to_coco_detection(self, predictions):
        results = []
        for idx, pred, in predictions.items():
            if len(pred) == 0:
                continue
            
            boxes = pred["boxes"]
            boxes = self._convert_to_xywh(boxes).tolist()
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()
            
            results.extend(
                {
                    "image_id": idx,
                    "category_id": labels[k],
                    "bbox": box,
                    "scores": scores[k],
                }
                for k, box in enumerate(boxes)
                )
        return results
            
    
                
        
        