import os

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.utils.utils import mld_collate
from model.metrics import MMMetrics, TM2TMetrics
from model.sdm.architectures import t2m_motionenc, t2m_textenc


class T2MEvaluator:
    def __init__(self, cfg:DictConfig, 
                 eval_dataset: Dataset,
                 model:torch.nn.Module) -> None:
        """
        Args:
            cfg (DictConfig):
            eval_dataset (Dataset): eval dataest is mdm dataset of evalmode
            model (torch.nn.Module): model should have 
                forward(motions:, text:, lengths:)
        """
        self.cfg = cfg
        self.eval_dataset = eval_dataset
        self.model = model
        self.is_mm = False
        
        self.eval_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=32,  # can not be changed due to R-precision
            drop_last=True,
            shuffle=True,
            collate_fn=mld_collate
        )
        
        self.get_t2m_evaluator()
        self.configure_metrics()
    
    def evaluate(self):
        for batch in tqdm(self.eval_loader):
            rst = self.t2m_eval(batch=batch)
            self.TM2TMetrics.update(text_embeddings=rst['lat_t'],
                                    recmotion_embeddings=rst['lat_rm'],
                                    gtmotion_embeddings=rst['lat_m'],
                                    lengths=batch['length'])
        
        return self.TM2TMetrics.compute(sanity_flag=False)
        
    
    def get_t2m_evaluator(self):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        cfg = self.cfg
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATALOADER.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.DATALOADER.NAME
        dataname = "t2m" if dataname == "humanml" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path,
                         "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False
            
    def configure_metrics(self):
        self.TM2TMetrics = TM2TMetrics(
                diversity_times=30,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)
        if self.is_mm:
            self.MMMetrics = MMMetrics(
                mm_num_times=self.cfg.TEST.MM_NUM_TIMES,
                dist_sync_on_step=self.cfg.METRIC.DIST_SYNC_ON_STEP)

    def t2m_eval(self, batch):        
        texts = batch['text']
        motions = batch["motion"].detach().clone()
        keyframes = batch["keyframe"].detach().clone()
        keyframe_id = batch["keyframe_id"].detach().clone()
        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()
    
        # if self.trainer.datamodule.is_mm:
        #     texts = texts * self.cfg.TEST.MM_NUM_REPEATSs
        #     motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                         dim=0)
        #     lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
        #     word_embs = word_embs.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)
        #     pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                             dim=0)
        #     text_lengths = text_lengths.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)
        if 'NUMAE' in str(type(self.model)):
            # noise = torch.zeros_like((keyframes)).to(self.model.device)
            # noise[:,:20,:] = 0.02*torch.randn((32, 20, 263)).to(self.model.device)
            feats_rst = self.model.forward(keyframes.to(self.model.device).to(self.model.device),
                                           texts, lengths)
        elif 'KFSVAE' in str(type(self.model)):
            feats_rst = self.model.reconstruct(
                motions.to(self.model.device), 
                keyframe_id.to(self.model.device), 
                texts, lengths
            )
        elif 'KFSLDM' in str(type(self.model)):
            # do keyframe uniformize
            
            # kfid = (torch.ones_like(keyframe_id) * -1).long()
            # for i, ids in enumerate(keyframe_id):
            #     ids = ids[ids!=-1]
            #     ids_ = list(range(0, lengths[i], lengths[i]//(len(ids-1))))
            #     while len(ids_) > len(ids):
            #         ids_.pop()
            #     kfid[i,:len(ids)] = torch.Tensor(ids_).long()

            feats_rst = self.model.forward(
                motions.to(self.model.device), 
                keyframe_id.to(self.model.device), 
                texts, lengths
            )
        elif 'KFMAE' in str(type(self.model)):
            # motions_n = motions + 0.7 * torch.randn_like(motions).to(motions.device)
            feats_rst = self.model.forward(
                motions.to(self.model.device), 
                keyframe_id.to(self.model.device), 
                texts, lengths
            )
        elif 'sdm' in str(type(self.model)):
            feats_rst = self.model.forward(motions.to(self.model.device), texts, lengths)
        else:
            feats_rst = self.model.forward(motions.to(self.model.device), texts, lengths)
        
        # renorm for t2m evaluators
        renorm = self.eval_dataset.renorm4t2m
        feats_rst = torch.Tensor(renorm(feats_rst.cpu().detach().numpy()))
        motions = torch.Tensor(renorm(motions.cpu().detach().numpy()))

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                            self.cfg.DATALOADER.UNIT_LEN,
                            rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb
        }
        return rs_set
