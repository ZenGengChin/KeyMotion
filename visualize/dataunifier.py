from typing import Any, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import os
import numpy as np

class HML2SMPL(LightningModule):
    """
    Data Unifier to convert HumanML motion representation
    to SMPL representation.
    [B, L, 263] -> [B, L, 135]
    non-scale -> scale
    """
    def __init__(self,
                 dim_humanml:str= 263, 
                 dim_smpl:str= 135,
                 islinear:bool= False):
        super().__init__()
        self.dim_humanml = dim_humanml
        self.dim_smpl = dim_smpl
        self.islinear = islinear
        # self.net = nn.Sequential(
        #     nn.Linear(dim_humanml, 2000),
        #     nn.ReLU(),
        #     nn.Linear(2000, 1000),
        #     nn.ReLU(),
        #     nn.Linear(1000, dim_humanml),
        #     nn.ReLU(),
        #     nn.Linear(dim_humanml, dim_smpl),
        #     nn.ReLU(),
        #     nn.Linear(dim_smpl, dim_smpl),
        #     nn.ReLU(),
        #     nn.Linear(dim_smpl, dim_smpl)
        # )
        self.TRML = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        self.TRM = nn.TransformerEncoder(
            encoder_layer=self.TRML,
            num_layers=3
        )
        
        #self.RNN = nn.RNN(dim_humanml, 3000, 4, batch_first=False)
        
        self.inproj = nn.Linear(dim_humanml, 512)
        self.outproj = nn.Linear(512, dim_smpl)
        
        self.loss = nn.MSELoss(reduction='sum')
        
        
    def forward(self, motion_hml) -> Any:
        # assert motion_hml.shape[-1] == self.dim_humanml
        # motion_theta = self.linear1(motion_hml)
        # if not self.islinear:
        #     motion_theta =  self.active(motion_theta)
        # motion_theta = self.lienar2(motion_theta)
        motion_hml = self.inproj(motion_hml)
        motion_hml = self.TRM(motion_hml)
        motion_hml = self.outproj(motion_hml)
        return motion_hml
        
        # return self.outproj(self.RNN(motionhml))
    
    def training_step(self, batch) -> STEP_OUTPUT:
        torch.cuda.empty_cache()
        motion_hml = batch[0]
        motion_smpl = batch[1]
        motion_theta = self.forward(motion_hml=motion_hml)
        loss = self.loss(motion_theta, motion_smpl)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    # def validation_step(self, batch, batch_idx):
    #     torch.cuda.empty_cache()
    #     motion_hml = batch[0]
    #     motion_smpl = batch[1]
    #     motion_theta = self.forward(motion_hml=motion_hml)
    #     #print('--------------',motion_smpl.shape, motion_hml.shape)
    #     loss = self.loss(motion_theta, motion_smpl)
    #     self.log('validation_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     return loss
    
    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        motion_hml = batch[0]
        motion_smpl = batch[1]
        motion_theta = self.forward(motion_hml=motion_hml)
        loss = self.loss(motion_theta, motion_smpl)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(
            self.parameters(),
            lr=0.000002,
            weight_decay=0
        )
        
class HML2SMPLDataset(Dataset):
    def __init__(self, smpl_path, hml_path):
        super().__init__()
        self.smpl_path = smpl_path
        self.hml_path = hml_path
        self.smpl_list = os.listdir(smpl_path)
        self.hml_list = os.listdir(self.hml_path)
        self.id_list = [x for x in self.hml_list if x in self.smpl_list]
        
    def __getitem__(self, index) -> Any:
        motion_smpl = np.load(os.path.join(self.smpl_path,self.id_list[index]))
        motion_smpl = torch.Tensor(motion_smpl)
        motion_hml = np.load(os.path.join(self.hml_path,self.id_list[index]))
        motion_hml = torch.Tensor(motion_hml)
        has_nan = torch.isnan(motion_smpl).any().item()
        if has_nan:
            print('smpl na')
        has_nan = torch.isnan(motion_hml).any().item()
        if has_nan:
            print('hml na', motion_hml, self.id_list[index])
            import time
            time.sleep(10)
        return motion_hml, motion_smpl[1:,:], self.id_list[index]
    
    def __len__(self):
        return len(self.id_list)

class HML2SMPLDataModule(LightningDataModule):
    """
    each pair is of [B,L,263] -> [B,L,135]
    """
    def __init__(self, 
                 smpl_path:str='/media/zen/D/DATA/HumanML3D/process',
                 hml_path:str='/media/zen/Windows/Users/20209/Desktop/PhD/Experiments/CHMG/dataset/HumanML3D/new_joint_vecs',
                 bs:int = 1) -> None:
        super().__init__()
        self.smpl_path = smpl_path
        self.hml_path = hml_path
        self.bs = bs
        self.DATASET = HML2SMPLDataset(smpl_path=smpl_path, hml_path=hml_path)
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.DATASET,
            batch_size=self.bs,
            shuffle=False,
            num_workers=12,
            collate_fn=self.collate_fn,
            drop_last=True)
        
    def val_dataloader(self):
        return DataLoader(
            dataset=self.DATASET,
            batch_size=self.bs,
            shuffle=False,
            num_workers=12,
            collate_fn=self.collate_fn,
            drop_last=True)
        
    def test_dataloader(self):
        return DataLoader(
            dataset=self.DATASET,
            batch_size=self.bs,
            shuffle=False,
            num_workers=12,
            collate_fn=self.collate_fn,
            drop_last=True)

    def collate_fn(self,batch):
        motion_hml = [item[0] for item in batch]
        motion_hml = torch.stack(motion_hml)
        motion_smpl = [item[1] for item in batch]
        motion_smpl = torch.stack(motion_smpl)
        motion_id = [item[2] for item in batch]
        return motion_hml, motion_smpl, motion_id
    

def train():
    import pytorch_lightning
    model = HML2SMPL()
    if os.path.exists('./hml2smpl.chkpt'):
        model = model.load_from_checkpoint('./hml2smpl.chkpt')
    datamodule = HML2SMPLDataModule()
    trainer = pytorch_lightning.Trainer(
        accelerator='cuda',
        max_epochs=300
    )
    trainer.fit(model=model, train_dataloaders=datamodule)
    trainer.save_checkpoint('./hml2smpl.chkpt')


if __name__ == '__main__':
    train()