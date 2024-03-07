import argparse
import os
import PIL
from PIL import Image
import json
from pathlib import Path
import torch
from data import CocoDetection
from model import Detr
from transformers import AutoImageProcessor, TrainingArguments, Trainer, DetrImageProcessor
from process import extract_label, LogPredictionsCallback
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import wandb
import accelerate
import cv2


def parse_args():
    parser = argparse.ArgumentParser("Set  transformer detection", add_help = False)
    parser.add_argument("--dataset_dir", default = None, type = str)
    parser.add_argument("--pretrained_model_name_or_path", default = "", type = str)
    parser.add_argument("--label_train_path", default = None, type = str,
                        help = "The file path containing the training labels")
    
    # Model 
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--lr_backbone", default = 1e-4, type = float)
    parser.add_argument("--batch_size", default = 2, type = int)
    parser.add_argument("--weight_decay", default = 1e-4, type = float)
    
    # Trainer
    parser.add_argument("--gradient_clip_val", default = 0.1, type = float)
    parser.add_argument("--max_epochs", default = 300, type = int)
    parser.add_argument("--accumulate_grad_batches", default = 8, type = int)
    parser.add_argument("--devices", default = 1, type = int)
    parser.add_argument("--accelerator", default = "cpu", type = str)
    parser.add_argument("--confidence_threshold", default = 0.5, type = float)
    parser.add_argument("--pub_to_hub", action = "store_true", type = bool)
    parser.add_argument("--hub_model_id", default = None, type = str)
    
    
    # Save model
    parser.add_argument("--output_dir", default = None, type = str)
    
    args = parser.parse_args()
    return args




def main():
    
    args = parse_args()
    
    TRAIN_PATH = os.path.join(args.dataset_dir, "train")
    VAL_PATH = os.path.join(args.dataset_dir, "valid")
    json_path = os.path.join(TRAIN_PATH, args.label_train_path)
    
    image_processor = DetrImageProcessor.from_pretrained(args.pretrained_model_name_or_path)
    
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors = "pt")
        labels = [item[1] for item in batch]
        
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels
        }
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    cats = {id: x for id,x in enumerate(data["categories"])}
    
    id2label = {id: v["name"] for id, v in enumerate(cats.values())}
    label2id = {v: k for k,v in id2label.items()}
    
    train_dataset  = CocoDetection( image_directory_path= TRAIN_PATH,
                                   image_processor = image_processor)
    val_dataset = CocoDetection(image_directory_path= VAL_PATH, 
                                image_processor = image_processor, train = False)
    
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                   collate_fn = collate_fn,
                                                   batch_size = args.batch_size,
                                                   shuffle = True)
    
   
    
    val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                   collate_fn = collate_fn,
                                                   batch_size = args.batch_size,
                                                   shuffle = True)
    
    # wandb logger
    confidence_threshold = args.confidence_threshold
    wandb_logger = WandbLogger(project = "DETR", log_model = "all")
    # log_predictions_callback = LogPredictionsCallback(val_dataset, id2label, label2id,
    #                                                 wandb_logger, image_processor, 
    #                                                 confidence_threshold,VAL_PATH)
    checkpoint_callback = ModelCheckpoint(monitor='validation_loss', mode='max')
    
    
    
    model = Detr(lr = args.lr, lr_backbone = args.lr_backbone,
                 weight_decay = args.weight_decay, 
                 checkpoint = args.pretrained_model_name_or_path,
                 id2label = id2label,
                 label2id = label2id)
    
    trainer = Trainer(devices = args.devices, accelerator= args.accelerator, 
                      max_epochs = args.max_epochs,
                      gradient_clip_val = args.gradient_clip_val,
                      accumulate_grad_batches = args.accumulate_grad_batches,
                      logger = wandb_logger,
                      callbacks=[checkpoint_callback],
                      default_root_dir = args.output_dir)
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
    if args.push_to_hub:
        model.model.push_to_hub("Soap007/detr-finetune-v1")
        image_processor.push_to_hub("Soap007/detr-finetune-v1")
    
     
if __name__ == "__main__":
    main()