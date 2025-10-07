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

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text, language_id="")
ta.save("test-1.wav", wav, model.sr)
