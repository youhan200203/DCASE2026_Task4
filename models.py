import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("unilm/beats")
from BEATs import BEATs, BEATsConfig
import math

class SpatialFeatureExtractor(nn.Module):
    def __init__(self, n_fft=1024, hop_length=320):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=n_fft, hop_length=hop_length, n_mels=128)

    def stft(self, x):
        window = torch.hann_window(self.n_fft, device=x.device)
        return torch.stft(x, self.n_fft, self.hop_length, window=window, return_complex=True)

    def forward(self, mixture):
        B, C, T = mixture.shape
        assert C == 4, "FOA는 4채널이어야 합니다 (W,Y,Z,X)"

        W = self.stft(mixture[:, 0])
        Y = self.stft(mixture[:, 1])
        Z = self.stft(mixture[:, 2])
        X = self.stft(mixture[:, 3])

        eps = 1e-8

        #Intensity Vector
        energy = W.abs().pow(2) + eps
        iv_x = (W.conj() * X).real / energy  # (B, F, T)
        iv_y = (W.conj() * Y).real / energy
        iv_z = (W.conj() * Z).real / energy
        iv = torch.stack([iv_x, iv_y, iv_z], dim=1)  # (B, 3, F, T)

        #IPD (채널 간 위상차)
        def ipd(a, b):
            phase = torch.angle(a * b.conj())           # (B, F, T)
            return torch.stack([phase.cos(), phase.sin()], dim=1)  # (B, 2, F, T)    

        ipd_wy = ipd(W, Y)
        ipd_wz = ipd(W, Z)
        ipd_wx = ipd(W, X)
        ipd_feat = torch.cat([ipd_wy, ipd_wz, ipd_wx], dim=1)  # (B, 6, F, T)   

        #ILD (채널 간 크기 비율)
        def ild(a, b):
            return torch.log((a.abs() + eps) / (b.abs() + eps)).unsqueeze(1)  # (B,1,F,T)

        ild_feat = torch.cat([ild(W, Y), ild(W, Z), ild(W, X)], dim=1)  # (B, 3, F, T)   

        spatial_feat = torch.cat([iv, ipd_feat, ild_feat], dim=1)  # (B, 12, F, T)

        w_mel = self.mel_transform(mixture[:, 0])
        w_mel = (w_mel + eps).log().unsqueeze(1) # (B, 1, n_mels, T_mel)             

        return spatial_feat, w_mel, W
    
class BEATS(nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint = torch.load("/content/BEATs_iter3_plus_AS2M.pt")
        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats = BEATs(cfg)
        self.beats.load_state_dict(checkpoint['model'])
        for p in self.beats.parameters():
            p.requires_grad= False
        print('BEATs Frozen')

    def forward(self, w_mel):
        features, _ = self.beats.extract_features(w_mel.squeeze(1), padding_mask=None)
        return features

class BEATsPooledCrossAttn(nn.Module):
    def __init__(self, n_compressed=32, attn_dim=64):
        super().__init__()
        self.n_compressed = n_compressed
        self.attn_dim = attn_dim

        self.beats_down = nn.Linear(768, attn_dim)
        self.q_proj = nn.Linear(256, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, 4, batch_first=True, droupout=0.1)

        self.out_proj = nn.Linear(attn_dim, 256)
        self.norm = nn.LayerNorm(256)

    def _pool_tokens(self, beats_tokens):
        B, N, D = beats_tokens.shape
        seg_size = math.ceil(N/self.n_compressed)
        compressed = beats_tokens.view(B, self.n_compressed, seg_size, D).mean(dim=2)
        return compressed  # (B, 32, D) 

    def forward(self, spatial_feat, beats_tokens):
        B, D, F, T = spatial_feat.shape 
        compressed = self._pool_tokens(beats_tokens)
        kv = self.beats_down(compressed)

        spatial_seq = spatial_feat.flatten(2).permute(0, 2, 1) # (B, F*T, 256)
        q = self.q_proj(spatial_seq)
        out, _ = self.attn(q, kv, kv)
        out = self.out_proj(out)
        out = self.norm(spatial_seq + out)
        out = out.permute(0, 2, 1).view(B, D, F, T) # (B, D, F, T)
        return out

class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_proj = nn.Sequential(
        nn.Conv2d(12, 64, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.Conv2d(64, 256, kernel_size=1),
        nn.GELU(),
        )

        self.beats_fusion = BEATsPooledCrossAttn()
        
    def forward(self, spatial_feat, beats_tokens):
        spatial_proj = self.spatial_proj(spatial_feat)
        out = self.beats_fusion(spatial_proj, beats_tokens)
        return out
    
class DoAEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_queries = nn.Parameter(torch.randn(3, 256))
        self.key_proj = nn.Linear(256, 256)
        self.query_proj = nn.Linear(256, 256)
        self.doa_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        for _ in range(3)
        ])

    def forward(self, fused):
        B, D, F, T = fused.shape
        feat = fused.flatten(2).permute(0, 2, 1)
        keys = self.key_proj(feat)
        queries = self.query_proj(self.slot_queries)
        doas = []
        for k in range(3):
            q_k = queries[k].unsqueeze(0).unsqueeze(0).expand(B, 1, D)  # (B, 1, D)
        score = torch.bmm(q_k, keys.permute(0, 2, 1)) / (D ** 0.5)
        weight = F.softmax(score, dim=-1)
        pooled = torch.bmm(weight, feat).squeeze(1)
        doas.append(self.doa_heads[k](pooled))
        return torch.stack(doas, dim=1)

class SpatialConditionedSeparator(nn.Module):
    def __init__(self, n_classes=18, doa_threshold=0.1):
        super().__init__()
        self.doa_threshold = doa_threshold

        # FiLM condition: DoA(3) → scale/shift(D*2)
        self.cond_proj = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 512),
        )

        # Mask decoder (shared across K slots)
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def _film(self, feat, doa_k):
        gb = self.cond_proj(doa_k) # (B, D*2)
        gamma, beta = gb.chunk(2, dim=-1) # 각 (B, D)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta # (B, D, F, T)

    def forward(self, fused, doa_pred, class_logits, k_pred):
        B, D, F, T = fused.shape
        masks = []
        active = []

        for k in range(self.K_max):
            doa_k = doa_pred[:, k, :]             # (B, 3)
            is_active = (doa_k.norm(dim=-1) > self.doa_threshold)
            conditioned = self._film(fused, doa_k)  # (B, D, F, T)
            mask_k = self.mask_decoder(conditioned).squeeze(1)      # (B, F, T)
            
            mask_k = mask_k * is_active.float().view(B, 1, 1)
            active.append(is_active)
            masks.append(mask_k)

        return (
            torch.stack(masks, dim=1),  # (B, K_max, F, T)
            torch.stack(active, dim=1)
        )

class WaveformReconstructor(nn.Module):
    def __init__(self, n_fft=1024, hop_length=320):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, mix_spec_w, masks, original_length):
        B, K, F, T_frames = masks.shape
        window = torch.hann_window(self.n_fft, device=mix_spec_w.device)

        if masks.shape[-2:] != mix_spec_w.shape[-2:]:
            masks = F.interpolate(
                masks.view(B * K, 1, F, T_frames),
                size=mix_spec_w.shape[-2:],
                mode="bilinear", align_corners=False,
            ).view(B, K, *mix_spec_w.shape[-2:])

        waveforms = []
        for k in range(K):
            masked = mix_spec_w * masks[:, k]
            wav = torch.istft(
                masked, self.n_fft, self.hop_length,
                window=window, length=original_length,
            )
            waveforms.append(wav)

        return torch.stack(waveforms, dim=1)  # (B, K_max, T)

class BEATsClassifier(nn.Module):
    def __init__(self, beats):
        super().__init__()
        self.beats = beats
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128)

        # BEATs token(768) → class logits(18): 이것만 학습
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 18),
        )

    def forward(self, waveforms, active):
        B, K, T = waveforms.shape
        eps = 1e-8
        all_logits = []

        for k in range(K):
            wav_k = waveforms[:, k, :]               # (B, T)
            mel = self.mel_transform(wav_k)          # (B, n_mels, T_mel)
            mel = (mel + eps).log().unsqueeze(1)      # (B, 1, n_mels, T_mel)

            with torch.no_grad():
                tokens = self.beats(mel)             # (B, N, 768)

            token_mean = tokens.mean(dim=1)          # (B, 768)
            logits_k = self.head(token_mean)         # (B, n_classes)
            logits_k = logits_k * active[:, k].float().unsqueeze(-1)
            all_logits.append(logits_k)
        return torch.stack(all_logits, dim=1)        # (B, K_max, n_classes)

class SpatialSeparatorModel(nn.Module):
    def __init__(self, hidden_dim=256, n_compressed=32, doa_threshold=0.1):
        super().__init__() 
        self.spatial_extractor = SpatialFeatureExtractor()
        self.beats = BEATS()
        self.encoder = SpatialEncoder()
        self.doa_estimator = DoAEstimator()
        self.separator = SpatialConditionedSeparator()
        self.reconstructor = WaveformReconstructor()
        self.classifier = BEATsClassifier(self.beats)

    def forward(self, mixture):
        B, C, T = mixture.shape

        # 1. Feature extraction
        spatial_feat, w_mel, mix_spec_w = self.spatial_extractor(mixture)

        # 2. BEATs tokens (frozen) — encoder fusion용
        with torch.no_grad():
            beats_tokens = self.beats(w_mel) # (B, ~992, 768)

        # 3. Spatial encoding + BEATs fusion
        fused = self.encoder(spatial_feat, beats_tokens) # (B, D, F, T)

        # 4. DoA 추정
        doa_pred = self.doa_estimator(fused) # (B, K_max, 3)

        # 5. Separation (DoA conditioning만, class 없이)
        masks, active = self.separator(fused, doa_pred)
        # masks  : (B, K_max, F, T_frames)
        # active : (B, K_max) bool  → K = active.sum(dim=-1)

        # 6. Waveform 복원
        waveforms = self.reconstructor(mix_spec_w, masks, T)   # (B, K_max, T)

        # 7. Class 예측 (separated waveform에 BEATs 직접 적용)
        class_logits = self.classifier(waveforms, active)      # (B, K_max, n_classes)

        return {
            "waveforms": waveforms,
            "masks": masks,
            "doa_pred": doa_pred,
            "class_logits": class_logits,
            "active": active,
            "k_pred": active.sum(dim=-1)
        }