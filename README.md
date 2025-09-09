# Continuous Thought Machine (CTM) Implementation

A PyTorch implementation of the **Continuous Thought Machine** architecture, combining GPT-2 with iterative neural refinement for enhanced language modeling capabilities.

## 🎯 Repository Goals

This repository implements and extends the Continuous Thought Machine (CTM) architecture from Sakana AI's research paper ["The Continuous Thought Machine"](https://arxiv.org/abs/2505.05522). Our implementation focuses on:

- **Synapse Models**: CTM performs multiple "thinking" iterations to refine predictions
- **Neuron-Level Models**: Per-neuron MLPs process memory traces for fine-grained control
- **Synchronization as Representation**: Pairwise neuron synchronization serves as the core representation
- **GPT-2 Integration**: Leveraging pre-trained language models as feature extractors

## 📄 Paper Context

The CTM introduces three key innovations:

1. **Internal Recurrence**: A dimension over which "thought" occurs through iterative processing
2. **Neuron-Level Models**: Private MLPs applied per-neuron to activation histories
3. **Synchronization Representation**: Neural activity tracked over time to compute pairwise neuron synchronization

Our implementation extends the original work with:
- **Logits-level fusion** between GPT-2 and CTM predictions
- **Memory-efficient training** strategies
- **Flexible architecture** supporting different backbone models

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/alexoh/GPT.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Ingestion

```bash
python src/data/ingestion.py
```

### Training

```bash
cd src/models/model1
python train.py --iterations 20 --num_batches 40000 --batch_size 2  --block_size 256 --total_batch_size 2048 --device cuda
```

### Key Training Arguments

- `--iterations`: Number of CTM thinking iterations (default: 12)
- `--num_batches`: Total training batches (default: 4000)
- `--batch_size`: Batch size (default: 2)
- `--block_size`: Sequence length (default: 128)
- `--n_head`: Number of attention heads (default: 4)
- `--memory_length`: Memory window for NLMs (default: 16)
- `--hidden_dimensions`: Hidden dims for NLMs (default: 4)

## 🔧 Intentional Design Decisions

### 1. **Logits-Level Skip Connection**
```python
# Mix predictions at vocabulary level, not hidden space
ctm_logits = self.lm_head(ctm_prediction)
final_logits = alpha * ctm_logits + (1-alpha) * gpt_logits
```
**Rationale**: Enables semantic alignment and interpretable fusion of token predictions.

### 2. **Memory-Efficient Architecture**
- **Single iteration storage**: Only store final predictions, not all iterations
- **Gradient checkpointing**: Trade compute for memory in deep iteration loops
- **Selective tracking**: Optional activation tracking for analysis

**Rationale**: CTM's iterative nature creates significant memory overhead; these optimizations enable practical training.

### 3. **Flexible Synchronization**
```python
# Support multiple neuron selection strategies
self.n_sync_out = int(self.d_model // 2)  # Output synchronization
self.n_sync_act = int(self.d_model - self.n_sync_out)  # Action synchronization
```
**Rationale**: Allows experimentation with different synchronization bottlenecks and representations.

### 4. **GPT-2 Feature Integration**
```python
# Use GPT-2 as frozen feature extractor
gptlogits, _, gptfeatures = self.GPT(x, targets)
# Add positional encodings for attention
pos_enc = self.GPT.transformer.wpe(pos)
kv = (gptfeatures + pos_enc).flatten(2)
```
**Rationale**: Leverages pre-trained language knowledge while allowing CTM to learn refinements.

### 5. **Adaptive Skip Weighting**
```python
self.skip_weight = nn.Parameter(torch.tensor(0.2))  # Learnable mixing parameter
```
**Rationale**: Model learns optimal balance between GPT-2 baseline and CTM refinements.

## 📁 Project Structure

```
gpt/
├── src/models/model1/
│   ├── ctm.py              # Main CTM implementation
│   ├── gpt.py              # GPT-2 backbone
│   ├── train.py            # Training script
│   ├── logger.py           # Logging utilities
│   └── ctmtemplate.py      # Reference implementation
├── src/data/
│   ├── dataloader.py       # Efficient data loading
│   └── ingestion.py        # Data preprocessing
├── data/
│   ├── raw/train/          # Training .parquet files
│   └── raw/validation/     # Validation .parquet files
└── models/                 # Saved model checkpoints
```

## 🧪 Key Components

### CTM Architecture
- **SynapseUNet**: U-Net for synaptic processing
- **NLM (Neuron-Level Models)**: Per-neuron MLPs with GLU activations
- **Synchronization**: Pairwise neuron correlation with exponential decay
- **Skip Connection**: Learnable fusion of GPT-2 and CTM predictions

### Training Features
- **Mixed precision** training (bfloat16)
- **Gradient accumulation** for large effective batch sizes
- **Memory management** for MPS/CUDA devices
- **Validation tracking** with early stopping
- **Checkpoint saving** with best model selection

## 🔬 Future Research Directions

### 1. **Loss-Guided Iteration Control**
- **Adaptive stopping**: End iterations when loss converges
- **Loss-weighted fusion**: Use prediction quality to guide mixing
- **Multi-objective training**: Balance multiple loss components

### 2. **Architecture Enhancements**
- **Hierarchical synchronization**: Multi-scale neuron groupings
- **Attention mechanisms**: Replace scaled dot-product with custom attention
- **Memory architectures**: External memory banks for longer-term dependencies

### 3. **Training Optimizations**
- **Curriculum learning**: Progressive iteration count increase
- **Knowledge distillation**: Transfer from larger CTM models
- **Federated training**: Distributed CTM training strategies

### 4. **Evaluation and Analysis**
- **Interpretability tools**: Visualize synchronization patterns
- **Ablation studies**: Component contribution analysis
- **Benchmark evaluation**: Standard language modeling tasks

### 5. **Applications**
- **Reasoning tasks**: Mathematical and logical reasoning
- **Long-form generation**: Extended text generation
- **Multi-modal CTM**: Vision + language integration

## 📊 Performance Notes

Current implementation achieves:
- **Training stability** on small-scale datasets
- **Memory efficiency** through architectural optimizations
- **Flexible experimentation** via configurable hyperparameters

## 🤝 Contributing

Areas for contribution:
- **Memory optimizations** for larger models
- **Evaluation metrics** and benchmarking
- **Visualization tools** for synchronization analysis
- **Documentation** and tutorials

## 📚 References

- [The Continuous Thought Machine](https://arxiv.org/abs/2505.05522) - Sakana AI
- [GPT-2 Implementation](https://github.com/karpathy/nanoGPT) - Andrej Karpathy
- [Interactive CTM Demo](https://pub.sakana.ai/ctm/) - Sakana AI

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This is a research implementation focused on understanding and extending the CTM architecture. For production use, additional optimizations and testing are recommended.
