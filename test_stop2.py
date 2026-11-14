import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from config import GPTConfig
from model.gpt import GPT
from data.tokenizer import CharTokenizer
from inference.sampler import sample_next_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load('checkpoints/latest_checkpoint.pt', map_location=device)
cfg = GPTConfig(**ckpt['config'])
model = GPT(cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

tok = CharTokenizer.load('checkpoints/char_vocab.json')

def generate_with_early_stop(prompt, max_sentences=2, max_tokens=150, min_tokens=25):
    tokens = tok.encode(prompt)
    curr = list(tokens)
    out = ""
    sentence_count = 0
    
    for step in range(max_tokens):
        ctx = curr[-cfg.block_size:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_tok = sample_next_token(logits[0, -1, :], temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1)
        
        curr.append(next_tok)
        piece = tok.decode([next_tok])
        out += piece
        
        # Stop as soon as max_sentences are complete!
        if step >= min_tokens and piece in ('.', '!', '?'):
            sentence_count += 1
            if sentence_count >= max_sentences:
                break
                
    return out.strip()

print("Testing Early Natural Sentence Stopping:")
for i in range(3):
    ans = generate_with_early_stop("Hello! How are you?", max_sentences=1, max_tokens=200, min_tokens=25)
    print(f"Sample {i+1} (Length: {len(ans)} chars):")
    print(ans)
    print("-" * 50)
