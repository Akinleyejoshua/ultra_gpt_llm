import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from pathlib import Path
from config import GPTConfig
from model.gpt import GPT
from data.tokenizer import CharTokenizer
from inference.bot import ConversationBot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('checkpoints/latest_checkpoint.pt', map_location=device)
cfg = GPTConfig(**ckpt['config'])
print(f"Loaded config: block_size={cfg.block_size}, vocab_size={cfg.vocab_size}, n_layer={cfg.n_layer}, n_embd={cfg.n_embd}")
model = GPT(cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

tok = CharTokenizer.load('checkpoints/char_vocab.json')
print(f"Loaded tokenizer: vocab_size={tok.vocab_size}")

bot = ConversationBot(model=model, tokenizer=tok, device=device)
print(f"Bot block_size={bot.block_size}")

print("=" * 60)
print("TEST 1: 'Hello! How are you?'")
resp1 = bot.generate('Hello! How are you?', max_tokens=256)
print("RESPONSE 1:")
print(repr(resp1))
print("=" * 60)

print("TEST 2: 'What is deep learning?'")
resp2 = bot.generate('What is deep learning?', max_tokens=256)
print("RESPONSE 2:")
print(repr(resp2))
print("=" * 60)
