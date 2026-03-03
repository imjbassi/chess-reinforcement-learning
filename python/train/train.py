import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import ChessNet


def train(batch_size=32, epochs=500, lr=1e-3, resume=True, kl_coeff=1e-4, data_path="selfplay.pt"):
    """
    Train the ChessNet model using self-play data.
    
    Args:
        batch_size: Number of samples per batch
        epochs: Number of training epochs
        lr: Learning rate for optimizer
        resume: Whether to resume from latest checkpoint if available
        kl_coeff: Coefficient for KL divergence regularization term
        data_path: Path to the self-play data file
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load self-play examples
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    examples = torch.load(data_path, map_location=device, weights_only=False)
    print(f"Loaded {len(examples)} training examples")
    
    # Prepare training data
    # Each example is (state_tensor, policy_distribution, value)
    states = torch.cat([ex[0] for ex in examples], dim=0)
    pis = torch.from_numpy(np.stack([ex[1] for ex in examples])).to(device)
    zs = torch.tensor([ex[2] for ex in examples], dtype=torch.float32).unsqueeze(1).to(device)

    ds = TensorDataset(states, pis, zs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model and optimizer
    net = ChessNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    start_epoch = 1
    best_loss = float("inf")
    
    # Resume from checkpoint if requested
    if resume and os.path.exists("model_latest.pt"):
        print("🔁 Resuming from model_latest.pt")
        checkpoint = torch.load("model_latest.pt", map_location=device, weights_only=True)
        net.load_state_dict(checkpoint)

    log = []
    
    for epoch in range(start_epoch, epochs + 1):
        net.train()
        epoch_p_loss = 0.0
        epoch_v_loss = 0.0
        epoch_kl_loss = 0.0

        for batch_s, batch_pi, batch_z in loader:
            batch_s = batch_s.to(device)
            batch_pi = batch_pi.to(device)
            batch_z = batch_z.to(device)

            # Forward pass
            logits, val = net(batch_s)

            # Policy loss: cross-entropy between predicted and target policy
            loss_p = -(batch_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            
            # Value loss: MSE between predicted and target value
            loss_v = F.mse_loss(val, batch_z)
            
            # KL divergence regularization to prevent overconfident predictions
            kl_loss = F.kl_div(F.log_softmax(logits, dim=1), batch_pi, reduction='batchmean')
            
            # Combined loss
            loss = loss_p + loss_v + kl_coeff * kl_loss

            # Backward pass and optimization
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Accumulate losses for logging
            epoch_p_loss += loss_p.item() * batch_s.size(0)
            epoch_v_loss += loss_v.item() * batch_s.size(0)
            epoch_kl_loss += kl_loss.item() * batch_s.size(0)

        # Update learning rate
        scheduler.step()

        # Calculate average losses
        N = len(ds)
        avg_p = epoch_p_loss / N
        avg_v = epoch_v_loss / N
        avg_kl = epoch_kl_loss / N
        total_loss = avg_p + avg_v + kl_coeff * avg_kl
        
        print(f"Epoch {epoch}/{epochs} — policy_loss: {avg_p:.4f}, value_loss: {avg_v:.4f}, kl_loss: {avg_kl:.4f}")

        # Log metrics
        log.append({
            "epoch": epoch,
            "policy_loss": avg_p,
            "value_loss": avg_v,
            "kl_loss": avg_kl,
            "total_loss": total_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        })

        # Save checkpoints
        torch.save(net.state_dict(), f"model_epoch{epoch}.pt")
        torch.save(net.state_dict(), "model_latest.pt")
        
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), "model_best.pt")
            print(f"✓ New best model saved (loss: {best_loss:.4f})")

    # Save training log
    df = pd.DataFrame(log)
    df.to_csv("training_log.csv", index=False)
    print(f"\n✓ Training complete. Log saved to training_log.csv")


if __name__ == "__main__":
    train()