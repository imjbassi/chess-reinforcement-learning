```python
import torch
import torch.nn as nn
import numpy as np


def encode_board(chessboard):
    """
    Encodes a ChessBoard instance into a tensor representation for neural network input.
    
    Creates an 18-plane representation of the board state:
    - Planes 0-11: Bitboards for each piece type (6 white pieces + 6 black pieces)
    - Plane 12: Side to move (1.0 for white, 0.0 for black)
    - Planes 13-16: Castling rights (white kingside, white queenside, black kingside, black queenside)
    - Plane 17: En passant target square
    
    Args:
        chessboard: A ChessBoard instance with an internal C++ Board accessible via ._b
        
    Returns:
        torch.Tensor: Shape (1, 18, 8, 8) tensor on CUDA device if available, else CPU
    """
    b = chessboard._b
    planes = []
    
    # Encode piece positions (12 planes: 6 piece types × 2 colors)
    for bb in b.pieces():
        arr = np.zeros((8, 8), dtype=np.float32)
        for sq in range(64):
            if (bb >> sq) & 1:
                r, f = divmod(sq, 8)
                arr[r, f] = 1.0
        planes.append(arr)

    # Encode side to move (1 plane)
    stm = 1.0 if b.white_to_move() else 0.0
    planes.append(np.full((8, 8), stm, dtype=np.float32))

    # Encode castling rights (4 planes)
    cr = b.castling_rights()
    for bit in (1, 2, 4, 8):
        val = 1.0 if (cr & bit) else 0.0
        planes.append(np.full((8, 8), val, dtype=np.float32))

    # Encode en passant square (1 plane)
    ep = b.ep_square()
    ep_plane = np.zeros((8, 8), dtype=np.float32)
    if ep >= 0:
        r, f = divmod(ep, 8)
        ep_plane[r, f] = 1.0
    planes.append(ep_plane)

    # Stack planes and convert to tensor
    x = np.stack(planes, axis=0)  # Shape: (18, 8, 8)
    t = torch.from_numpy(x).unsqueeze(0)  # Shape: (1, 18, 8, 8)
    
    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = t.to(device)
    
    return t


class ResidualBlock(nn.Module):
    """
    Residual block for deeper network architectures.
    
    Implements a standard residual connection with two convolutional layers.
    Uses batch normalization and ReLU activation for improved training stability.
    """
    
    def __init__(self, num_filters):
        """
        Initialize the residual block.
        
        Args:
            num_filters: Number of filters in convolutional layers
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor of shape (batch_size, num_filters, 8, 8)
            
        Returns:
            torch.Tensor: Output tensor with residual connection applied
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out


class ChessNet(nn.Module):
    """
    Neural network for chess position evaluation and move prediction.
    
    Architecture:
    - Initial convolutional layer for feature extraction from board representation
    - Optional residual blocks for deeper networks
    - Shared convolutional layers for feature processing
    - Policy head: Outputs move probabilities (4096 possible moves)
    - Value head: Outputs position evaluation in range [-1, 1]
    
    The network uses a shared convolutional backbone followed by two separate heads:
    one for policy (move selection) and one for value (position evaluation).
    """
    
    def __init__(self, num_filters=64, num_res_blocks=0):
        """
        Initialize the ChessNet model.
        
        Args:
            num_filters: Number of filters in convolutional layers (default: 64)
            num_res_blocks: Number of residual blocks to add (default: 0 for backward compatibility)
        """
        super().__init__()
        
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        
        # Initial convolutional layer
        self.conv_input = nn.Sequential(
            nn.Conv2d(18, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks (if any)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Shared convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        # Policy head: predicts move probabilities
        # Outputs 4096 logits (64 from squares × 64 to squares)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4096)
        )

        # Value head: evaluates position quality
        # Outputs a single scalar in range [-1, 1] via tanh activation
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 18, 8, 8)
            
        Returns:
            tuple: (policy, value)
                - policy: Tensor of shape (batch_size, 4096) with move logits
                - value: Tensor of shape (batch_size, 1) with position evaluation in [-1, 1]
        """
        # Initial convolution
        x = self.conv_input(x)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Extract features using shared convolutional layers
        features = self.conv_layers(x)
        
        # Compute policy and value predictions
        policy = self.policy_head(features)
        value = self.value_head(features)
        
        return policy, value
```