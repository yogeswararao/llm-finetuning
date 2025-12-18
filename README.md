# LLM Fine-tuning

A comprehensive collection of LLM fine-tuning methods with examples. Includes LoRA, QLoRA, AdaLoRA, Delta-LoRA, VeRA, Prompt Tuning, P-Tuning, and more. 

## Blog post

Check out the blog post for a high-level walk through: [10 Practical Ways to Fine-Tune an LLM]()

## Features

This project implements the following fine-tuning methods:

- **Full Fine-tuning**: Updates all model parameters
- **LoRA** (Low-Rank Adaptation): Parameter-efficient fine-tuning with low-rank matrices
- **LoRA-FA**: LoRA with frozen A matrices
- **LoRA+**: LoRA with different learning rates for A and B matrices
- **Delta-LoRA**: LoRA with trainable base weights
- **AdaLoRA**: Adaptive rank allocation for LoRA
- **VeRA**: Vector-based Random Matrix Adaptation
- **QLoRA**: LoRA with 4-bit quantization for memory efficiency
- **Prompt Tuning**: Learnable prompt tokens
- **P-Tuning**: Prompt tuning with encoder

## Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) - Python package manager

### Quick Setup

1. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
      
   ```bash
   source .venv/bin/activate
   ```

## Usage


Fine-tuning methods can be run as shown below:

```python
from src.finetuner import LoRAFineTuner

# Initialize and run LoRA fine-tuning
lora = LoRAFineTuner()
lora.run(save_model=True)
```

## Interactive Notebook

For an interactive exploration of all fine-tuning methods, check out [`llm-finetuning.ipynb`](llm-finetuning.ipynb)

## Results Comparison

Performance of different fine-tuning methods for the IMDb sentiment classification task:

| Method | Accuracy | Trainable Params | % of Total Params |
|--------|----------|-----------------|------------|
| Full Fine-tuning | 93.32% | ~68M | 100% |
| LoRA | 92.62% | ~887k | 1.3% |
| LoRA-FA | 91.20% | ~739k | ~1% |
| LoRA+ | 92.80% | ~887k | ~1.3% |
| Delta-LoRA (Approximation) | 92.31% | ~68M | 100% |
| AdaLoRA | 91.10% | ~1M | ~1.52% |
| QLoRA | 92.56% | ~296k | ~0.44% |
| VeRA | 91.02% | ~617k | ~0.91% |
| Prompt Tuning | 85.22% | ~607k | ~0.89% |
| P-Tuning | 89.96% | ~2.3M | ~3.43% |


## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
