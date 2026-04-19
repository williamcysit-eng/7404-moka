import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. Layer Definitions (MokA vs Standard LoRA)
# ==========================================

class StandardLoRALayer(nn.Module):
    """
    Standard LoRA implementation for Multimodal LLMs.
    Shared A and B matrices across all modalities.
    """
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        self.r = r
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.A.weight, a=torch.math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        
    def forward(self, x_concat):
        return self.B(self.A(x_concat))

class MokALayer(nn.Module):
    """
    Multimodal Low-Rank Adaptation (MokA) implementation.
    Modality-specific A matrices, cross-attention, and shared B matrix.
    """
    def __init__(self, in_features, out_features, r=4, lambda_a=1.0, lambda_v=1.0):
        super().__init__()
        self.r = r
        self.lambda_a = lambda_a
        self.lambda_v = lambda_v
        
        self.A_audio = nn.Linear(in_features, r, bias=False)
        self.A_visual = nn.Linear(in_features, r, bias=False)
        self.A_text = nn.Linear(in_features, r, bias=False)
        
        self.B = nn.Linear(r, out_features, bias=False)
        
        for A in [self.A_audio, self.A_visual, self.A_text]:
            nn.init.kaiming_uniform_(A.weight, a=torch.math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        
    def forward(self, x_audio, x_visual, x_text):
        # 1. Unimodal Compression
        h_a = self.A_audio(x_audio)
        h_v = self.A_visual(x_visual)
        h_t = self.A_text(x_text)
        
        # 2. Task-centric Cross-attention
        scores_a = torch.matmul(h_a, h_t.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.r, dtype=torch.float32))
        attn_a = F.softmax(scores_a, dim=-1)
        context_a = torch.matmul(attn_a, h_t)
        
        scores_v = torch.matmul(h_v, h_t.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.r, dtype=torch.float32))
        attn_v = F.softmax(scores_v, dim=-1)
        context_v = torch.matmul(attn_v, h_t)
        
        # Enhance non-text tokens
        h_a_enhanced = h_a + self.lambda_a * context_a
        h_v_enhanced = h_v + self.lambda_v * context_v
        
        # 3. Shared Multimodal Projection
        out_a = self.B(h_a_enhanced)
        out_v = self.B(h_v_enhanced)
        out_t = self.B(h_t)
        
        out_concat = torch.cat([out_a, out_v, out_t], dim=1)
        return out_concat

# ==========================================
# 2. Mock MLLM Model
# ==========================================

class MockMLLM(nn.Module):
    def __init__(self, in_features=16, out_features=16, r=4, use_moka=False):
        super().__init__()
        self.use_moka = use_moka
        
        # "Pretrained" frozen backbone
        self.frozen_base = nn.Linear(in_features, out_features, bias=False)
        self.frozen_base.weight.requires_grad = False
        # Small random initialization
        nn.init.normal_(self.frozen_base.weight, std=0.01)
        
        if use_moka:
            self.adapter = MokALayer(in_features, out_features, r=r)
        else:
            self.adapter = StandardLoRALayer(in_features, out_features, r=r)
            
        # Classification head
        self.classifier = nn.Linear(out_features, 2)
        
    def forward(self, x_audio, x_visual, x_text):
        x_concat = torch.cat([x_audio, x_visual, x_text], dim=1)
        
        # Forward pass through frozen base
        base_out = self.frozen_base(x_concat)
        
        # Forward pass through adapter
        if self.use_moka:
            adapt_out = self.adapter(x_audio, x_visual, x_text)
        else:
            adapt_out = self.adapter(x_concat)
            
        h = base_out + adapt_out
        
        # Global average pooling over the sequence dimension
        h_pooled = h.mean(dim=1)
        return self.classifier(h_pooled)

# ==========================================
# 3. Synthetic Dataset Generation
# ==========================================
# This synthetic task simulates a scenario where the text prompt 
# dictates whether the audio or visual modality holds the answer.

def generate_dataset(num_samples=2000, seq_a=5, seq_v=5, seq_t=2, dim=16):
    torch.manual_seed(42) # Reproducibility
    X_a = torch.randn(num_samples, seq_a, dim)
    X_v = torch.randn(num_samples, seq_v, dim)
    X_t = torch.randn(num_samples, seq_t, dim)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        task = i % 2
        if task == 0:
            # Task A: The answer is in the visual tokens.
            # Text signature: positive values
            X_t[i] = 1.5 
            if X_v[i].sum() > 0:
                y[i] = 1
            else:
                y[i] = 0
        else:
            # Task B: The answer is in the audio tokens.
            # Text signature: negative values
            X_t[i] = -1.5
            if X_a[i].sum() > 0:
                y[i] = 1
            else:
                y[i] = 0
                
    return X_a, X_v, X_t, y

# ==========================================
# 4. Training Loop
# ==========================================

def train_model(model, dataloader, model_name, epochs=15, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\\nTraining {model_name}...")
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        for xa, xv, xt, y in dataloader:
            optimizer.zero_grad()
            outputs = model(xa, xv, xt)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        acc = correct / total
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc*100:.2f}%")
            
    return acc

def main():
    print("Generating synthetic Multi-Modal task dataset...")
    # The task: Given a text prompt, predict the label based on either Audio or Visual data.
    # MokA's task-centric cross-attention allows it to explicitly route the text instruction
    # to the audio/visual features, whereas LoRA just concatenates everything.
    
    X_a, X_v, X_t, y = generate_dataset(num_samples=2000, dim=16)
    dataset = TensorDataset(X_a, X_v, X_t, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    r = 4 # Low rank
    epochs = 20
    lr = 0.005
    
    # Initialize models (using same random seed for fairness)
    torch.manual_seed(100)
    lora_model = MockMLLM(in_features=16, out_features=16, r=r, use_moka=False)
    
    torch.manual_seed(100)
    moka_model = MockMLLM(in_features=16, out_features=16, r=r, use_moka=True)
    
    print("\\n=======================================================")
    print("Baseline: Standard LoRA Training")
    print("=======================================================")
    lora_acc = train_model(lora_model, dataloader, "Standard LoRA", epochs=epochs, lr=lr)
    
    print("\\n=======================================================")
    print("Proposed: MokA Training")
    print("=======================================================")
    moka_acc = train_model(moka_model, dataloader, "MokA", epochs=epochs, lr=lr)
    
    print("\\n=======================================================")
    print("Final Results (Training Accuracy)")
    print("=======================================================")
    print(f"Standard LoRA Accuracy: {lora_acc*100:.2f}%")
    print(f"MokA Accuracy:          {moka_acc*100:.2f}%")
    
    if moka_acc > lora_acc:
        print("\\nSuccess! MokA achieved higher accuracy than standard LoRA, demonstrating")
        print("the effectiveness of unimodal adaptation and cross-modal attention.")
    else:
        print("\\nMokA did not outperform LoRA on this specific run.")

if __name__ == "__main__":
    main()
