# Repositorio de Finetuning (WIP) para chatterbox tts
Utiliza los scripts de finetuning proporcionados para realizar el ajuste fino de los modelos chatterbox multilingual t3 y s3/flow.<br>
referencia: https://github.com/stlohrey/chatterbox-finetuning

# Uso

```
pip install chatterbox-tts
pip uninstall chatterbox-tts
cd src
```

```
#establecer idioma en finetune_t3.py en la línea 86
python finetune_t3.py \
--output_dir ./checkpoints/v1 \
--model_name_or_path ResembleAI/chatterbox \
--dataset_name MrDragonFox/DE_Emilia_Yodas_680h \
--train_split_name train \
--eval_split_size 0.0002 \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 2 \
--learning_rate 5e-5 \
--warmup_steps 100 \
--logging_steps 10 \
--eval_strategy steps \
--eval_steps 2000 \
--save_strategy steps \
--save_steps 4000 \
--save_total_limit 4 \
--fp16 True \
--report_to tensorboard \
--dataloader_num_workers 8 \
--do_train --do_eval \
--dataloader_pin_memory False \
--eval_on_start True \
--label_names labels_speech \
--text_column_name text_scribe
```

# Demo de Gradio

```
python gradio_tts_app.py
```


<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/XqS7RxUp)

_Hecho con ♥️ por <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

Estamos emocionados de presentar Chatterbox, el primer modelo de TTS de código abierto de grado de producción de [Resemble AI](https://resemble.ai). Bajo licencia MIT, Chatterbox ha sido evaluado frente a los principales sistemas de código cerrado como ElevenLabs, y es preferido consistentemente en evaluaciones comparativas directas.

Ya sea que estés trabajando en memes, videos, juegos o agentes de IA, Chatterbox da vida a tu contenido. También es el primer modelo de TTS de código abierto que soporta el **control de exageración emocional**, una característica poderosa que hace que tus voces destaquen. Pruébalo ahora en nuestra [aplicación de Gradio en Hugging Face.](https://huggingface.co/spaces/ResembleAI/Chatterbox)

Si te gusta el modelo pero necesitas escalarlo o ajustarlo para obtener una mayor precisión, consulta nuestro servicio de TTS con precios competitivos (<a href="https://resemble.ai">enlace</a>). Ofrece un rendimiento fiable con una latencia ultrabaja de menos de 200 ms, ideal para uso en producción en agentes, aplicaciones o medios interactivos.

# Detalles Clave
- TTS zeroshot SoTA
- Backbone Llama de 0.5B
- Control único de exageración/intensidad
- Ultra-estable con inferencia informada por alineación
- Entrenado con 0.5M horas de datos limpios
- Salidas con marca de agua
- Script sencillo de conversión de voz
- [Supera a ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Consejos
- **Uso General (TTS y Agentes de Voz):**
  - Los ajustes predeterminados (`exaggeration=0.5`, `cfg_weight=0.5`) funcionan bien para la mayoría de los prompts.
  - Si el hablante de referencia tiene un estilo de habla rápido, reducir el `cfg_weight` a aproximadamente `0.3` puede mejorar el ritmo.

- **Habla Expresiva o Dramática:**
  - Prueba valores de `cfg_weight` más bajos (ej. `~0.3`) e incrementa la `exaggeration` a `0.7` o más.
  - Una mayor `exaggeration` tiende a acelerar el habla; reducir el `cfg_weight` ayuda a compensar con un ritmo más lento y deliberado.


# Instalación
```
pip install chatterbox-tts
```


# Uso
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# Si deseas sintetizar con una voz diferente, especifica el audio prompt
AUDIO_PROMPT_PATH="YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```
Consulta `example_tts.py` para más ejemplos.

# Agradecimientos
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Marcado de Agua PerTh Integrado para una IA Responsable

Cada archivo de audio generado por Chatterbox incluye el [Marcador de Agua Perth (Perceptual Threshold) de Resemble AI](https://github.com/resemble-ai/perth): marcas de agua neuronales imperceptibles que sobreviven a la compresión MP3, la edición de audio y manipulaciones comunes, manteniendo una precisión de detección de casi el 100%.

# Discord Oficial

👋 ¡Únete a nosotros en [Discord](https://discord.gg/XqS7RxUp) y construyamos algo increíble juntos!

# Descargo de Responsabilidad
No utilices este modelo para hacer cosas malas. Los prompts provienen de datos disponibles gratuitamente en internet.
