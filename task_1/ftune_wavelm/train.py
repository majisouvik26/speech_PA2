import torch
import torch.optim as optim
from tqdm import tqdm
from arcface_loss import ArcFaceLoss
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

def train_model(model, classifier, train_loader, num_epochs, learning_rate, device, label_map):
    """
    Fine-tune the model and classifier using the provided training DataLoader.
    
    label_map: Dictionary mapping identity (string) to numeric label.
    Mixed-precision training is used to help reduce GPU memory usage.
    """
    optimizer = optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate, weight_decay=1e-2)
    loss_fn = ArcFaceLoss(s=30.0, m=0.5)
    scaler = GradScaler()  

    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        epoch_loss = 0.0
        for audio, identities in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            audio = audio.to(device)
            labels = torch.tensor([label_map[id_] for id_ in identities], dtype=torch.long).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=device.type):
                embeddings = model(audio).last_hidden_state.mean(dim=1)
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                weight_norm = F.normalize(classifier.weight, p=2, dim=1)
                logits = torch.matmul(embeddings_norm, weight_norm.t())
                loss = loss_fn(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        torch.cuda.empty_cache() 
    print("Training complete.")
