import torch
import torch.optim as optim
from tqdm import tqdm
from arcface_loss import ArcFaceLoss

def train_model(model, classifier, train_loader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    loss_fn = ArcFaceLoss(s=30.0, m=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        classifier.train()
        epoch_loss = 0.0
        for audio, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            audio = audio.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model(audio).last_hidden_state.mean(dim=1)
            logits = classifier(embeddings)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

