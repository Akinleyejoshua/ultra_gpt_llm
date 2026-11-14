import json
from pathlib import Path

nb_path = Path("inference.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    # Clear old execution outputs so stale truncated text is removed
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

    new_source = []
    for line in cell.get("source", []):
        # 1. Update text generation function
        line = line.replace("max_new_tokens: int = 60", "max_new_tokens: int = 256")
        # 2. Update Console chat
        line = line.replace("max_tokens=80", "max_tokens=512")
        # 3. Update GUI slider
        line = line.replace(
            'tokens_slider = widgets.IntSlider(value=80, min=20, max=200, step=10, description="Max Toks:", layout=widgets.Layout(width="28%"))',
            'tokens_slider = widgets.IntSlider(value=512, min=64, max=1024, step=32, description="Max Toks:", layout=widgets.Layout(width="28%"))'
        )
        # 4. Update FastAPI test payload
        line = line.replace('"max_tokens": 40', '"max_tokens": 256')
        new_source.append(line)
    cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Successfully updated inference.ipynb with complete token generation settings!")
