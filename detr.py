import pytorch_lightning as pl
from transformers import AutoModelForObjectDetection, DetrForObjectDetection
import torch

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, checkpoint, id2label):
        super().__init__()
        self.checkpoint = checkpoint
        self.id2label = id2label
        self.model = DetrForObjectDetection.from_pretrained(pretrained_model_name_or_path = self.checkpoint,
                                                            id2label = self.id2label,
                                                            revision = "no_timm",
                                                            num_labels = len(self.id2label),
                                                            ignore_mismatched_sizes = True)
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
    
    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values = pixel_values, pixel_mask = pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k,v in t.items()} for t in batch["labels"]]
        
        outputs = self.model(pixel_values = pixel_values, pixel_mask = pixel_mask, labels = labels)
        
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        
        return loss, loss_dict, outputs
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)
        
        self.log("training_loss", loss, prog_bar = True)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item(), prog_bar = True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict, outputs = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, prog_bar = True)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())
        
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [ p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            }
        ]
        
        return torch.optim.AdamW(param_dicts, lr = self.lr, weight_decay = self.weight_decay)
    
    def train_dataloader(self):
        return train_dataloader
    
    def val_dataloader(self):
        return val_dataloader
    