# MokA Demo: Educational Reproduction

This repository contains an independent, minimal PyTorch demonstration of the techniques described in the NeurIPS 2025 paper:
**"MokA: Multimodal Low-Rank Adaptation for MLLMs"**.

*   **Authors**: Yake Wei, Yu Miao, Dongzhan Zhou, Di Hu
*   **Paper Link**: [arXiv:2506.05191](https://arxiv.org/abs/2506.05191)
*   **Official Project Page**: [https://gewu-lab.github.io/MokA](https://gewu-lab.github.io/MokA)

*Disclaimer: This repository is an unofficial, educational implementation designed to isolate and demonstrate the task-centric cross-attention mechanism of MokA against Standard LoRA on a synthetic toy problem. For the official implementation and pre-trained weights used in the paper's benchmarks (e.g., MUSIC-AVQA, POPE), please refer to the authors' official project page.*

## Citation
If you found the paper's ideas useful, please consider citing their work:
```bibtex
@inproceedings{wei2025moka,
  title={MokA: Multimodal Low-Rank Adaptation for MLLMs},
  author={Wei, Yake and Miao, Yu and Zhou, Dongzhan and Hu, Di},
  booktitle={39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025}
}
```

---

## 1. Approach to the Demo
The primary goal of the demo is to vividly illustrate the core architectural differences between a standard LoRA layer (used widely in LLMs) and the newly proposed MokA layer.

To achieve this, the best approach is to build a **minimal PyTorch implementation** that focuses on a single forward pass of mock Multimodal Large Language Model (MLLM) data. The demo script simulates an **Audio-Visual-Text** scenario, which is a key use case highlighted in the paper (e.g., the MUSIC-AVQA dataset).

The demo highlights the three critical components of the MokA architecture:
1. **Unimodal Matrix A**: Instead of a shared $A$ matrix, MokA uses modality-specific matrices ($A_{audio}$, $A_{visual}$, $A_{text}$) to prevent information from different modalities from interfering with each other during initial compression.
2. **Task-Centric Cross-Attention**: Non-text modalities (audio and visual) are often used as environmental context, while text provides the task description (the prompt). MokA uses cross-attention where the audio/visual tokens act as queries to attend to the text tokens (keys/values), thereby merging textual information into the non-text modalities.
3. **Shared Multimodal Matrix B**: After the unimodal compression and cross-modal enhancement, all tokens are projected into a unified space using a single shared $B$ matrix.

## 2. The Demo Implementation
The implementation is contained in [`demo_moka.py`](demo_moka.py).

### Use Case Demonstration
We simulate a scenario where a user asks a question about a video that includes audio.
*   **Audio Tokens**: Shape `[1, 32, 4096]` (e.g., 32 segments of an audio clip)
*   **Visual Tokens**: Shape `[1, 64, 4096]` (e.g., 64 frames of a video)
*   **Text Tokens**: Shape `[1, 10, 4096]` (e.g., "What instrument makes the sound?")

### Code Breakdown
1.  **Standard LoRA Baseline (`StandardLoRALayer`)**:
    *   Takes all 106 tokens concatenated together.
    *   Passes them through a single $A$ matrix and a single $B$ matrix.
    *   Outputs the modified tokens without any explicit mechanism distinguishing the modalities.
2.  **MokA Strategy (`MokALayer`)**:
    *   **Unimodal Compression**: Extracts $h_a$, $h_v$, and $h_t$ independently using `A_audio`, `A_visual`, and `A_text`.
    *   **Cross-Attention**:
        *   Calculates `attn_a` using $h_a$ to query $h_t$. This yields an attention weight matrix of shape `[1, 32, 10]`, showing how each of the 32 audio tokens pays attention to the 10 text tokens.
        *   Calculates `attn_v` using $h_v$ to query $h_t$. This yields an attention weight matrix of shape `[1, 64, 10]`.
        *   The context from text is added back to the audio and visual tokens ($h_{a\_enhanced} = h_a + \lambda_a \cdot context_a$).
    *   **Shared Projection**: Passes all enhanced tokens through a shared $B$ matrix and concatenates them back to the original sequence length of 106 tokens.

You can run the demo by executing:
```bash
python demo_moka.py
```

### End-to-End Training Demonstration
To demonstrate that MokA actually achieves **better training accuracy** over LoRA as stated in the paper, a second script [`train_demo.py`](train_demo.py) is provided. 

This script builds a synthetic "MockMLLM" task designed specifically to mimic the challenges of multimodal routing:
*   **The Task**: Predict a binary label depending on the **text query**. If the text indicates Task A, the answer is found in the **visual** tokens. If the text indicates Task B, the answer is found in the **audio** tokens.
*   **Standard LoRA**: By concatenating all tokens and using shared low-rank matrices, it struggles to dynamically route task-specific information. In 20 epochs, it typically caps around **54.5%** accuracy (barely better than guessing).
*   **MokA**: Due to its independent unimodal matrices and explicit cross-attention (where visual and audio tokens query the text task), it successfully learns to route context dynamically. In 20 epochs, it achieves over **76.5%** accuracy.

You can run the training simulation yourself:
```bash
python train_demo.py
```

## 3. Checking the Baseline (MokA vs. LoRA)
The paper provides extensive empirical evidence that the MokA technique actually beats a baseline of standard LoRA across multiple MLLMs and datasets.

By extracting individual unimodal features and explicitly routing task-relevant textual data into non-text features, the model better adapts to complex multimodal tasks compared to the "indiscriminate" parameter sharing of standard LoRA.

Below is a summary of the performance comparison directly extracted from the paper (using the **LLaMA2** backbone as the baseline reference):

### Audio-Visual-Text Scenario (Table 1)
*   **MUSIC-AVQA Dataset**: 
    *   LoRA: 73.41%
    *   **MokA: 75.71%**
*   **AVE Dataset**:
    *   LoRA: 69.84%
    *   **MokA: 74.68%**

### Visual-Text Scenario (Table 2)
*   **POPE Dataset** (Evaluating object hallucination):
    *   LoRA: 70.28%
    *   **MokA: 74.23%**
*   **MME_percep Dataset**:
    *   LoRA: 908.52
    *   **MokA: 1025.86**

### Speech-Text Scenario (Table 3)
*   **AIR-Bench_speech-en**:
    *   LoRA: 31.75%
    *   **MokA: 39.64%**

**Conclusion:** 
The experimental results from the paper conclusively demonstrate that **MokA consistently and significantly outperforms the standard LoRA baseline** across various datasets and multimodality combinations (Audio-Visual-Text, Visual-Text, and Speech-Text). The demo code successfully captures the mechanistic differences responsible for this performance gain.
