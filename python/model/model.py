import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def encode_board(chessboard):
    """
    Accepts a ChessBoard instance; grabs its internal C++ Board (. _b)
    and builds a (1,18,8,8) torch.Tensor on CUDA.
    """
    b = chessboard._b

    planes = []
    for bb in b.pieces():
        arr = np.zeros((8, 8), np.float32)
        for sq in range(64):
            if (bb >> sq) & 1:
                r, f = divmod(sq, 8)
                arr[r, f] = 1.0
        planes.append(arr)

    stm = 1.0 if b.white_to_move() else 0.0
    planes.append(np.full((8, 8), stm, np.float32))

    cr = b.castling_rights()
    for bit in (1, 2, 4, 8):
        val = 1.0 if (cr & bit) else 0.0
        planes.append(np.full((8, 8), val, np.float32))

    ep = b.ep_square()
    ep_plane = np.zeros((8, 8), np.float32)
    if ep >= 0:
        r, f = divmod(ep, 8)
        ep_plane[r, f] = 1.0
    planes.append(ep_plane)

    x = np.stack(planes)  # (18,8,8)
    t = torch.from_numpy(x).unsqueeze(0).cuda()  # (1,18,8,8)
    return t

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
