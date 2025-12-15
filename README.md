# LLM Fine-tuning

A collection of LLM fine-tuning methods, including LoRA, QLoRA, AdaLoRA, and more.

## Medium Article

Check out the Medium article for a high level walk through: [Link to Medium Article](https://medium.com/@your-username/your-article-link)

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
- **P-Tuning v2**: Advanced prompt tuning with encoder

## Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

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

### Running Fine-tuning Methods

You can run any of the fine-tuning methods directly. For example:

```bash
python -m src.methods.lora
```

### Interactive Notebook

For an interactive exploration of all fine-tuning methods, check out the Jupyter notebook:

- [`llm-finetuning.ipynb`](llm-finetuning.ipynb) - Interactive notebook with examples and comparisons of all fine-tuning methods

## Results Comparison

The following table compares the performance of different fine-tuning methods on the IMDB sentiment classification task:

| Method | Accuracy | Trainable Params | % of Total Params |
|--------|----------|-----------------|------------|
| Full Fine-tuning | - | - | 100% |
| LoRA | - | - | ~0.1% |
| QLoRA | - | - | ~0.1% |
| AdaLoRA | - | - | ~0.1% |
| LoRA+ | - | - | ~0.1% |
| LoRA-FA | - | - | ~0.05% |
| Delta-LoRA | - | - | ~0.1% + base |
| VeRA | - | - | ~0.01% |
| Prompt Tuning | - | - | ~0.01% |
| P-Tuning v2 | - | - | ~0.01% |

*Note: Results are based on training with DistilBERT-base-uncased on the IMDB dataset. Actual values will be updated after running experiments.*


## License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
