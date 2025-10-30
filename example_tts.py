import torchaudio as ta
import torch
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxMultilingualTTS.from_checkpoint(
    "./checkpoints/v1/",
    device,
)

"""
## or you want to load a specific checkpoint
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
checkpoint_path="my_checkpoint"

#for safetensors
from safetensors.torch import load_file
t3_state = load_file(checkpoint_path, device="cpu")
model.t3.load_state_dict(t3_state)

#for torch file
t3_state  = torch.load(checkpoint_path)
model.t3.load_state_dict(t3_state)
"""

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text, language_id="")
ta.save("test-1.wav", wav, model.sr)
