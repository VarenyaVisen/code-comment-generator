import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

print("🎉 All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Test tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
test_text = "def hello_world(): return 'Hello, World!'"
tokens = tokenizer.encode(test_text)
print(f"Test tokenization successful: {len(tokens)} tokens")