# services/face_parsing.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# BiSeNet (Face Parsing)
# Basado en la arquitectura típica usada en "face-parsing" (CelebAMask-HQ)
# -------------------------

def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

class _BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out

class _ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [_BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        feat8  = self.layer1(x)   # 1/4
        feat16 = self.layer2(feat8)   # 1/8
        feat32 = self.layer3(feat16)  # 1/16
        feat64 = self.layer4(feat32)  # 1/32
        return feat8, feat16, feat32, feat64

class _AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv(x)
        att = self.attn(feat)
        return feat * att

class _FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 4, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, sp, cx):
        feat = torch.cat([sp, cx], dim=1)
        feat = self.convblk(feat)
        att = self.attn(feat)
        return feat + feat * att

class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.backbone = _ResNet18()

        # Context Path
        self.arm16 = _AttentionRefinementModule(256, 128)
        self.arm32 = _AttentionRefinementModule(512, 128)

        self.conv_head16 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_head32 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Spatial Path (simple)
        self.sp = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ffm = _FeatureFusionModule(64 + 128, 256)

        self.cls = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        feat8, feat16, feat32, feat64 = self.backbone(x)

        sp = self.sp(x)  # 1/8 aprox

        # context
        cx32 = self.arm32(feat64)
        cx32 = F.interpolate(cx32, size=feat32.shape[2:], mode="bilinear", align_corners=False)
        cx32 = self.conv_head32(cx32)

        cx16 = self.arm16(feat32)
        cx16 = cx16 + cx32
        cx16 = F.interpolate(cx16, size=feat16.shape[2:], mode="bilinear", align_corners=False)
        cx16 = self.conv_head16(cx16)

        # subimos a tamaño de sp
        cx = F.interpolate(cx16, size=sp.shape[2:], mode="bilinear", align_corners=False)

        feat_fuse = self.ffm(sp, cx)
        out = self.cls(feat_fuse)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


# -------------------------
# Loader + API
# -------------------------

# Índices típicos CelebAMask-HQ (19 clases):
# 0 background
# 1 skin
# 6 ear_l
# 7 ear_r
# 17 hair
IDX_SKIN = 1
IDX_EAR_L = 6
IDX_EAR_R = 7
IDX_HAIR = 17

class FaceParsingService:
    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = device
        self.net = BiSeNet(n_classes=19).to(device)
        self.net.eval()

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No encuentro pesos de Face Parsing en: {weights_path}\n"
                f"Coloca el archivo .pth ahí (ej: 79999_iter.pth)."
            )

        state = torch.load(weights_path, map_location="cpu")
        # algunos pesos vienen como {"state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # limpia prefijos típicos
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        self.net.load_state_dict(new_state, strict=False)

    @torch.no_grad()
    def predict_masks(self, bgr_uint8: np.ndarray) -> dict:
        """
        Entrada: BGR uint8 (H,W,3)
        Salida: dict con máscaras binarias uint8 (0/255) para hair/skin/ear
        """
        h, w = bgr_uint8.shape[:2]
        rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)

        # input tamaño típico 512
        inp_size = 512
        img = cv2.resize(rgb, (inp_size, inp_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        # normalización simple (suficiente en práctica)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        logits = self.net(t)  # (1,19,512,512)
        parsing = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        # resize al tamaño original
        parsing = cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)

        hair = (parsing == IDX_HAIR).astype(np.uint8) * 255
        skin = (parsing == IDX_SKIN).astype(np.uint8) * 255
        ear  = ((parsing == IDX_EAR_L) | (parsing == IDX_EAR_R)).astype(np.uint8) * 255

        return {
            "hair": hair,
            "skin": skin,
            "ear": ear,
            "parsing": parsing,  # opcional (debug)
        }
