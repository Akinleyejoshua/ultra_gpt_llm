import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from config import GPTConfig
from model.gpt import GPT
from data.tokenizer import CharTokenizer
from inference.bot import ConversationBot

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('checkpoints/latest_checkpoint.pt', map_location=device)
cfg = GPTConfig(**ckpt['config'])
model = GPT(cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

tok = CharTokenizer.load('checkpoints/char_vocab.json')
bot = ConversationBot(model=model, tokenizer=tok, device=device)

print("=== 1. GREEDY DECODING (temperature=0.0): 100% Deterministic ===")
print("Run 1:")
r1 = bot.generate('Hello! How are you?', temperature=0.0, max_tokens=120)
print(r1)
bot.reset()
print("\nRun 2 (identical to Run 1):")
r2 = bot.generate('Hello! How are you?', temperature=0.0, max_tokens=120)
print(r2)

print("\n=== 2. PROBABILISTIC SAMPLING (temperature=0.7): Different each run ===")
bot.reset()
print("Sample A:")
print(bot.generate('Hello! How are you?', temperature=0.7, max_tokens=120))

bot.reset()
print("\nSample B:")
print(bot.generate('Hello! How are you?', temperature=0.7, max_tokens=120))

bot.reset()
print("\nSample C:")
print(bot.generate('Hello! How are you?', temperature=0.7, max_tokens=120))
