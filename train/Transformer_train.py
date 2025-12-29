import torch
from torch.nn.utils import clip_grad_norm_
import os
from torch.utils.data import DataLoader
from data.dataset import (
    TranslationDataset,
    collate_fn,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    vocab_en,
    vocab_de,
    Tokenize_en,
    Tokenize_de,
    train_data,
    test_data,
    StoT_de,
    StoT_en
)
from models.Transformer import Transformer,NoamLR

train_dataset = TranslationDataset(
    data=train_data,
    StoT_en=StoT_en,
    StoT_de=StoT_de,
    SOS_IDX=SOS_IDX,
    EOS_IDX=EOS_IDX
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn
)

EMBED_SIZE = 256
HEADS = 4
NUM_LAYERS = 2
EXPANSION = 4
DROPOUT = 0.2
MAX_LEN=50

model=Transformer(
    src_vocab_size=len(vocab_de),
    tgt_vocab_size=len(vocab_en),
    embed_size=EMBED_SIZE,
    num_layers=NUM_LAYERS,
    heads=HEADS,
    expansion=EXPANSION,
    dropout=DROPOUT,
    max_len=MAX_LEN,
    pad_idx=PAD_IDX
)

criterion=torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.1)

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=1,
    betas=(0.9,0.98),
    eps=1e-9
)

scheduler = NoamLR(
    optimizer=optimizer,
    model_dim=EMBED_SIZE,
    warmup_steps=1000,
    step_num=0
)

CHECKPOINT_PATH="weights/transformer.pt"
os.makedirs("weights", exist_ok=True)

def train_one_epoch(model,loader,optimizer,criterion):
    model.train()
    total_loss=0.0

    for src,tgt in loader:
        optimizer.zero_grad()

        outputs=model(src,tgt)
        out_dim=outputs.shape[-1]

        outputs=outputs[:,1:,:].reshape(-1,out_dim)
        tgt=tgt[:,1:].reshape(-1)

        loss=criterion(outputs,tgt)
        loss.backward()
        clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        lr=scheduler.step()  
        total_loss+=loss.item()
    
    return total_loss/len(loader)


start_epoch=0
best_train_loss=float("inf")
train_losses=[]

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.step_num=checkpoint["scheduler_step"]
    best_train_loss=checkpoint["best_train_loss"]
    train_losses=checkpoint["train_losses"]
    start_epoch=checkpoint["epoch"]

    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    print("no weights found")


num_epochs=10
for epoch in range(start_epoch, num_epochs):
    train_loss=train_one_epoch(
        model,train_loader,optimizer,criterion
    )

    train_losses.append(train_loss)
    if train_loss<best_train_loss:
        best_train_loss=train_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_step": scheduler.step_num,
            "best_train_loss": best_train_loss,
            "train_losses": train_losses,
        }, CHECKPOINT_PATH)

        print(f"Saved new best model at epoch {epoch+1}")

    print(
        f"Epoch {epoch+1}/{num_epochs}|"
        f"Train Loss: {train_loss:.4f}"
    )
