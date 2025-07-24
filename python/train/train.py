import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import encode_board, ChessNet  # FIXED: Use correct import path

def train(batch_size=32, epochs=500, lr=1e-3, resume=True, kl_coeff=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    examples = torch.load("selfplay.pt")
    states = torch.cat([ex[0] for ex in examples], dim=0)
    pis    = torch.from_numpy(np.stack([ex[1] for ex in examples]))
    zs     = torch.tensor([ex[2] for ex in examples], dtype=torch.float32).unsqueeze(1)

    ds = TensorDataset(states, pis, zs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    net = ChessNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    start_epoch = 1
    best_loss = float("inf")
    if resume and os.path.exists("model_latest.pt"):
        print("🔁 Resuming from model_latest.pt")
        net.load_state_dict(torch.load("model_latest.pt", map_location=device))

    log = []
    for epoch in range(start_epoch, epochs + 1):
        net.train()
        epoch_p_loss = 0.0
        epoch_v_loss = 0.0

        for batch_s, batch_pi, batch_z in loader:
            batch_s = batch_s.to(device)
            batch_pi = batch_pi.to(device)
            batch_z = batch_z.to(device)

            logits, val = net(batch_s)

            loss_p = -(batch_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss_v = F.mse_loss(val, batch_z)
            kl_loss = F.kl_div(F.log_softmax(logits, dim=1), batch_pi, reduction='batchmean')
            loss = loss_p + loss_v + kl_coeff * kl_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_p_loss += loss_p.item() * batch_s.size(0)
            epoch_v_loss += loss_v.item() * batch_s.size(0)

        scheduler.step()

        N = len(ds)
        avg_p = epoch_p_loss / N
        avg_v = epoch_v_loss / N
        total_loss = avg_p + avg_v
        print(f"Epoch {epoch}/{epochs} — policy_loss: {avg_p:.4f}, value_loss: {avg_v:.4f}")

        log.append({
            "epoch": epoch,
            "policy_loss": avg_p,
            "value_loss": avg_v,
            "total_loss": total_loss,
        })

        torch.save(net.state_dict(), f"model_epoch{epoch}.pt")
        torch.save(net.state_dict(), "model_latest.pt")
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), "model_best.pt")

    df = pd.DataFrame(log)
    df.to_csv("training_log.csv", index=False)

if __name__ == "__main__":
    train()
