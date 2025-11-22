
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.rtfs_blocks.ae import audio_encoder
from src.model.rtfs_blocks.VP import VPBlock
from src.model.rtfs_blocks.RTFS import APBlock, StackedRTFS
from src.model.rtfs_blocks.fusion import CAF
from src.model.rtfs_blocks.s3 import SpectralSourceSeparation
from src.model.rtfs_blocks.decoder import Decoder


class RTFSNet(nn.Module):
    def __init__(
            self,
            #fourier transform parameters
            n_fft=1024, 
            hop_length=128, 
            av_channels=256,
            win_length=256,

            #video processing block parameters
            v_channels=512, 
            vp_D=64,
            vp_q=4,
            vp_num_heads=8,
            vp_ffn_dim=128,
            vp_dropout=0.1,

            #audio processing (A.K.A. rtfs) blocks parameters
            ap_D=64,
            ap_q=2, # compression power of rtfs block -> trade-off between quality and memory
            ap_rnn_hidden=32,
            ap_rnn_layers=4,
            ap_attn_heads=4,
            ap_dropout=0.1,
            R=12, # number of repetetive processing of the same rtfs block after fusion


            #fusion block params
            caf_num_heads=8,

            ):
        
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.av_channels = av_channels

        self.v_channels = v_channels
        self.vp_D = vp_D
        self.vp_q = vp_q
        self.vp_num_heads = vp_num_heads
        self.vp_ffn_dim = vp_ffn_dim
        self.vp_dropout = vp_dropout

        self.ap_D = ap_D
        self.ap_q = ap_q
        self.ap_rnn_hidden = ap_rnn_hidden
        self.ap_rnn_layers = ap_rnn_layers
        self.ap_attn_heads = ap_attn_heads
        self.ap_dropout = ap_dropout

        self.caf_num_heads = caf_num_heads
        self.R = R



        self.ae = audio_encoder(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            av_channels=self.av_channels
            )

        self.vp_block = VPBlock(
            in_channels=self.v_channels, 
            D=self.vp_D, 
            q=self.vp_q, 
            num_heads=self.vp_num_heads, 
            ffn_dim=self.vp_ffn_dim, 
            dropout=self.vp_dropout
            )
        
        self.ap_block = APBlock(
            in_channels=self.av_channels, 
            D=self.ap_D, 
            q=self.ap_q, 
            rnn_hidden=self.ap_rnn_hidden, 
            rnn_layers=self.ap_rnn_layers, 
            attn_heads=self.ap_attn_heads, 
            dropout=self.ap_dropout
            )
        
        self.caf_block = CAF(
            audio_channels=self.av_channels, 
            video_channels=self.v_channels, 
            heads=self.caf_num_heads, 
            )
        
        self.sss = SpectralSourceSeparation(in_channels=self.av_channels)

        self.decoder = Decoder(
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            input_channels=self.av_channels,
              win_length=self.win_length
            )

    def forward(self, v0, mix_audio):  
        a0 = self.ae(mix_audio)  # [B, 256, 251, 513]


        ae = self.caf_block(self.ap_block(a0), self.vp_block(v0))  # [B, 256, 251, 513] 

        for _ in range(self.R):
            ae = self.ap_block(ae + a0)

        ae = self.sss(ae, a0)  # [B, 256, 251, 513]

        separated = self.decoder(ae, mix_audio.shape[-1])  # [B, 32000]

        return separated