import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardLoRALayer(nn.Module):
    """
    Standard LoRA implementation for Multimodal LLMs.
    In standard LoRA, the A and B matrices are shared across all modalities.
    """
    def __init__(self, in_features, out_features, r=4):
        super().__init__()
        self.r = r
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        
        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.A.weight, a=torch.math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        
    def forward(self, x_concat):
        """
        x_concat: [batch_size, total_seq_len, in_features]
        """
        # All modalities are processed by the same shared A and B matrices
        return self.B(self.A(x_concat))

class MokALayer(nn.Module):
    """
    Multimodal Low-Rank Adaptation (MokA) implementation from the paper.
    It incorporates unimodal A matrices, task-centric cross-attention, 
    and a shared B matrix.
    """
    def __init__(self, in_features, out_features, r=4, lambda_a=1.0, lambda_v=1.0):
        super().__init__()
        self.r = r
        self.lambda_a = lambda_a
        self.lambda_v = lambda_v
        
        # Unimodal A matrices
        self.A_audio = nn.Linear(in_features, r, bias=False)
        self.A_visual = nn.Linear(in_features, r, bias=False)
        self.A_text = nn.Linear(in_features, r, bias=False)
        
        # Shared Multimodal B matrix
        self.B = nn.Linear(r, out_features, bias=False)
        
        # Initialize A matrices with Kaiming uniform and B with zeros
        for A in [self.A_audio, self.A_visual, self.A_text]:
            nn.init.kaiming_uniform_(A.weight, a=torch.math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        
    def forward(self, x_audio, x_visual, x_text):
        """
        x_audio: [batch_size, seq_len_a, in_features]
        x_visual: [batch_size, seq_len_v, in_features]
        x_text: [batch_size, seq_len_t, in_features]
        """
        # 1. Unimodal Compression
        h_a = self.A_audio(x_audio)   # [B, seq_a, r]
        h_v = self.A_visual(x_visual) # [B, seq_v, r]
        h_t = self.A_text(x_text)     # [B, seq_t, r]
        
        # 2. Task-centric Cross-attention
        # Audio attends to Text
        # Query: h_a, Key: h_t, Value: h_t
        scores_a = torch.matmul(h_a, h_t.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.r, dtype=torch.float32))
        attn_a = F.softmax(scores_a, dim=-1) # [B, seq_a, seq_t]
        context_a = torch.matmul(attn_a, h_t) # [B, seq_a, r]
        
        # Visual attends to Text
        # Query: h_v, Key: h_t, Value: h_t
        scores_v = torch.matmul(h_v, h_t.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.r, dtype=torch.float32))
        attn_v = F.softmax(scores_v, dim=-1) # [B, seq_v, seq_t]
        context_v = torch.matmul(attn_v, h_t) # [B, seq_v, r]
        
        # Enhance non-text tokens
        h_a_enhanced = h_a + self.lambda_a * context_a
        h_v_enhanced = h_v + self.lambda_v * context_v
        
        # 3. Shared Multimodal Projection
        out_a = self.B(h_a_enhanced)
        out_v = self.B(h_v_enhanced)
        out_t = self.B(h_t)
        
        # Concatenate back to a single sequence
        out_concat = torch.cat([out_a, out_v, out_t], dim=1) # [B, seq_a + seq_v + seq_t, out_features]
        
        return out_concat, attn_a, attn_v


def run_demo():
    print("=" * 60)
    print("MokA: Multimodal Low-Rank Adaptation for MLLMs Demo")
    print("=" * 60 + "\n")
    
    # Hyperparameters
    batch_size = 1
    in_features = 4096  # e.g., typical LLM hidden size (like LLaMA 7B)
    out_features = 4096 # e.g., output dimension of a target dense layer
    r = 4
    
    # Token lengths
    seq_len_audio = 32
    seq_len_visual = 64
    seq_len_text = 10
    
    # Create mock inputs (random embeddings)
    x_audio = torch.randn(batch_size, seq_len_audio, in_features)
    x_visual = torch.randn(batch_size, seq_len_visual, in_features)
    x_text = torch.randn(batch_size, seq_len_text, in_features)
    
    # Concatenate for Standard LoRA
    x_concat = torch.cat([x_audio, x_visual, x_text], dim=1)
    
    print(f"[Inputs]")
    print(f"Audio Tokens:   {x_audio.shape}")
    print(f"Visual Tokens:  {x_visual.shape}")
    print(f"Text Tokens:    {x_text.shape}")
    print(f"Total Sequence Length: {x_concat.shape[1]}\n")
    
    # Initialize layers
    standard_lora = StandardLoRALayer(in_features, out_features, r=r)
    moka_layer = MokALayer(in_features, out_features, r=r)
    
    print("-" * 60)
    print("1. Standard LoRA Baseline")
    print("-" * 60)
    print("Standard LoRA processes all tokens indiscriminately using shared A and B matrices.")
    lora_out = standard_lora(x_concat)
    print(f"Output Shape: {lora_out.shape}\n")
    
    print("-" * 60)
    print("2. MokA (Multimodal-Aware) Strategy")
    print("-" * 60)
    print("MokA dynamically applies:")
    print("  - Modality-specific A matrices (Unimodal Adaptation)")
    print("  - Task-Centric Cross-Attention (Audio->Text, Visual->Text)")
    print("  - Shared B matrix for unified projection")
    
    moka_out, attn_a, attn_v = moka_layer(x_audio, x_visual, x_text)
    
    print(f"\n[Intermediate Shapes in MokA]")
    print(f"Audio-to-Text Attention Weights:  {attn_a.shape} (Queries: {seq_len_audio}, Keys: {seq_len_text})")
    print(f"Visual-to-Text Attention Weights: {attn_v.shape} (Queries: {seq_len_visual}, Keys: {seq_len_text})")
    
    print(f"\nMokA Final Output Shape: {moka_out.shape}\n")

if __name__ == "__main__":
    run_demo()
