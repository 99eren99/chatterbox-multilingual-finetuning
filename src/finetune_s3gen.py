import argparse
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig
)
from transformers import TrainingArguments as HfTrainingArguments
from datasets import load_dataset, DatasetDict, VerificationMode, Audio
import datasets

# Chatterbox imports
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3gen import S3Gen, S3GEN_SR
from chatterbox.models.s3gen.s3gen import S3Token2Mel, mel_spectrogram # S3GEN_SR for mels
from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR as S3_TOKENIZER_SR # S3Tokenizer operates at 16kHz
from chatterbox.models.s3gen.flow import CausalMaskedDiffWithXvec # The actual module we want to finetune
from chatterbox.models.s3gen.xvector import CAMPPlus # Speaker encoder used by S3Gen

logger = logging.getLogger(__name__)

# --- Training Arguments (can reuse CustomTrainingArguments) ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None, metadata={"help": "Enable early stopping."}
    )

# --- Model Arguments ---
@dataclass
class S3GenModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Path to base Chatterbox model"})
    local_model_dir: Optional[str] = field(default=None, metadata={"help": "Path to local Chatterbox model directory"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "HF cache directory"})
    # S3Gen specific finetuning args
    freeze_speaker_encoder: bool = field(default=True, metadata={"help": "Freeze S3Gen's internal speaker encoder (CAMPPlus)."})
    freeze_s3_tokenizer: bool = field(default=True, metadata={"help": "Freeze S3Gen's internal S3Tokenizer."})
    # The 'flow' part of S3Gen will be trained. HiFiGAN part will be frozen.

# --- Data Arguments ---
@dataclass
class S3GenDataArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "HF Dataset name"})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "HF Dataset config name"})
    train_split_name: str = field(default="train", metadata={"help": "Train split name"})
    eval_split_name: Optional[str] = field(default="validation", metadata={"help": "Eval split name"})
    audio_column_name: str = field(default="audio", metadata={"help": "Audio column in dataset"})
    # No text column needed for S3Gen training directly, but might be in dataset
    
    max_speech_token_len: int = field(default=750, metadata={"help": "Max S3 speech tokens for target sequence."})
    max_mel_len: int = field(default=1500, metadata={"help": "Max mel frames for target mel (max_speech_token_len * 2)"})
    # For CausalMaskedDiffWithXvec, we need prompt_tokens and prompt_feats for conditioning
    prompt_audio_duration_s: float = field(default=3.0, metadata={"help": "Duration of audio for prompt_token and prompt_feat."})
    
    eval_split_size: float = field(default=0.01, metadata={"help": "Eval split fraction"}) # Increased default
    ignore_verifications: bool = field(default=False, metadata={"help":"Ignore dataset verifications."})


# --- S3Gen Finetuning Dataset ---
class S3GenFineTuningDataset(Dataset):
    def __init__(self,
                 data_args: S3GenDataArguments,
                 s3gen_model: S3Gen, # Pass the S3Gen model directly
                 hf_dataset: datasets.Dataset):
        self.data_args = data_args
        self.s3gen_model = s3gen_model # Contains tokenizer, mel_extractor, speaker_encoder
        self.dataset_source = hf_dataset

        self.s3_tokenizer_sr = S3_TOKENIZER_SR # 16kHz for S3Tokenizer
        self.s3_gen_native_sr = S3GEN_SR     # 24kHz for mel spectrograms and CAMPPlus input
        
        self.prompt_audio_samples_16k = int(data_args.prompt_audio_duration_s * self.s3_tokenizer_sr)
        self.prompt_audio_samples_24k = int(data_args.prompt_audio_duration_s * self.s3_gen_native_sr)

        self._resamplers = {}

    def _get_resampler(self, orig_sr: int, target_sr: int) -> T.Resample:
        if (orig_sr, target_sr) not in self._resamplers:
            self._resamplers[(orig_sr, target_sr)] = T.Resample(orig_sr, target_sr)
        return self._resamplers[(orig_sr, target_sr)]

    def __len__(self):
        return len(self.dataset_source)

    def _load_and_preprocess_audio(self, audio_data_from_hf):
        waveform: Optional[torch.Tensor] = None
        original_sr: Optional[int] = None

        if isinstance(audio_data_from_hf, str):
            try: waveform, original_sr = torchaudio.load(audio_data_from_hf)
            except Exception: return None, None
        elif isinstance(audio_data_from_hf, dict) and "array" in audio_data_from_hf and "sampling_rate" in audio_data_from_hf:
            np_array = audio_data_from_hf["array"]
            if not isinstance(np_array, np.ndarray): return None, None
            waveform = torch.from_numpy(np_array).float()
            original_sr = audio_data_from_hf["sampling_rate"]
            if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        else: return None, None

        if waveform is None or original_sr is None or waveform.numel() == 0: return None, None
        if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Ensure float32
        waveform = waveform.float()

        # Prepare 16kHz version (for S3Tokenizer and potentially CAMPPlus if it expects 16k)
        if original_sr != self.s3_tokenizer_sr:
            resampler_16k = self._get_resampler(original_sr, self.s3_tokenizer_sr)
            wav_16k_tensor = resampler_16k(waveform)
        else:
            wav_16k_tensor = waveform.clone() # Clone to avoid modifying dataset source waveform
        
        # Prepare 24kHz version (for mel_extractor and CAMPPlus if it expects 24k)
        # Note: s3gen.speaker_encoder (CAMPPlus) in s3gen.py takes 16kHz input.
        # s3gen.mel_extractor takes 24kHz input.
        if original_sr != self.s3_gen_native_sr:
            resampler_24k = self._get_resampler(original_sr, self.s3_gen_native_sr)
            wav_24k_tensor = resampler_24k(waveform)
        else:
            wav_24k_tensor = waveform.clone()

        return wav_16k_tensor.squeeze(0), wav_24k_tensor.squeeze(0) # Return (L,) tensors

    def __getitem__(self, idx) -> Optional[Dict[str, torch.Tensor]]:
        item = self.dataset_source[idx]
        audio_data_hf = item[self.data_args.audio_column_name]

        wav_16k_tensor, wav_24k_tensor = self._load_and_preprocess_audio(audio_data_hf)
        if wav_16k_tensor is None or wav_24k_tensor is None or wav_16k_tensor.numel() == 0 or wav_24k_tensor.numel() == 0:
            return None

        # 1. Target Mel Spectrograms (speech_feat) from full 24kHz audio
        # mel_spectrogram returns (B, F, T_mel), we need (F, T_mel) then transpose to (T_mel, F) for flow model
        try:
            target_mel = self.s3gen_model.mel_extractor(wav_24k_tensor.unsqueeze(0)).squeeze(0).transpose(0, 1) # (T_mel, F)
            if target_mel.size(0) > self.data_args.max_mel_len:
                target_mel = target_mel[:self.data_args.max_mel_len, :]
            speech_feat_len = torch.tensor(target_mel.size(0), dtype=torch.long)
        except Exception as e:
            logger.error(f"Item {idx}: Error extracting target_mel: {e}", exc_info=True)
            return None

        # 2. Target S3 Speech Tokens (speech_token) from full 16kHz audio
        # S3Tokenizer expects list of numpy arrays
        try:
            speech_tokens_batch, speech_token_lengths_batch = self.s3gen_model.tokenizer.forward(
                [wav_16k_tensor.numpy()], max_len=self.data_args.max_speech_token_len
            )
            if speech_tokens_batch is None or speech_token_lengths_batch is None: return None
            target_s3_tokens = speech_tokens_batch.squeeze(0) # (T_tokens_s3)
            speech_token_len = torch.tensor(target_s3_tokens.size(0), dtype=torch.long, device="cpu")
            # Ensure token length matches mel length (T_mel = 2 * T_tokens_s3 usually)
            # This alignment is crucial. If S3Tokenizer's max_len truncates differently than mel's max_len, adjust.
            # For simplicity, assume S3Tokenizer's max_len aligns with max_mel_len / 2
            if target_mel.size(0) != 2 * target_s3_tokens.size(0) and target_mel.size(0) // 2 < target_s3_tokens.size(0):
                target_s3_tokens = target_s3_tokens[:target_mel.size(0)//2]
                speech_token_len = torch.tensor(target_s3_tokens.size(0), dtype=torch.long)
            elif target_mel.size(0) // 2 > target_s3_tokens.size(0) : # Pad tokens if mel is longer due to truncation
                 pad_size = target_mel.size(0)//2 - target_s3_tokens.size(0)
                 target_s3_tokens = F.pad(target_s3_tokens, (0, pad_size), value=0) # Pad with 0
                 # speech_token_len remains the original length before padding for this case

        except Exception as e:
            logger.error(f"Item {idx}: Error tokenizing target speech: {e}", exc_info=True)
            return None

        # 3. Speaker Embedding (embedding) from 16kHz audio (as per S3Gen's CAMPPlus usage)

        model_device = next(self.s3gen_model.speaker_encoder.parameters()).device
        try:
            # Temporarily set speaker encoder to eval mode for inference
            original_training_state = self.s3gen_model.speaker_encoder.training
            self.s3gen_model.speaker_encoder.eval() # <<< SET TO EVAL MODE

            speaker_embedding_batch = self.s3gen_model.speaker_encoder.inference(
                wav_16k_tensor.unsqueeze(0).to(model_device) # Input (1, L_wav)
            )
            speaker_embedding = speaker_embedding_batch.squeeze(0) # (D_spk)

            # Restore original training state if needed (though for dataset prep, it's fine to leave as eval)
            # self.s3gen_model.speaker_encoder.train(original_training_state) # Optional, see note below

        except Exception as e:
            logger.error(f"Item {idx}: Error getting speaker_embedding: {e}", exc_info=True)
            # Restore original training state in case of error too
            # if 'original_training_state' in locals():
            #    self.s3gen_model.speaker_encoder.train(original_training_state)
            return None
        finally:
            # Ensure the model is restored to its original state if you modified it
            # This is important if the main training loop expects it to be in a certain state.
            if 'original_training_state' in locals():
                self.s3gen_model.speaker_encoder.train(original_training_state)

        # 4. Prompt features for CausalMaskedDiffWithXvec conditioning
        # prompt_token, prompt_token_len, prompt_feat, prompt_feat_len
        prompt_wav_16k_segment = wav_16k_tensor[:self.prompt_audio_samples_16k]
        prompt_wav_24k_segment = wav_24k_tensor[:self.prompt_audio_samples_24k]

        if prompt_wav_16k_segment.numel() == 0 or prompt_wav_24k_segment.numel() == 0: # Handle very short audio
            # logger.warning(f"Item {idx}: Prompt audio segment too short. Using zero prompts.")
            # Max prompt token length for Causal model, assumed to be something like 150 (3s * 25Hz * 2 for mel ratio)
            # This needs to be consistent with how CausalMaskedDiffWithXvec expects prompt lengths.
            # For now, let's set a placeholder; this might need adjustment based on model internals.
            # The `embed_ref` in S3Gen handles this. Let's try to replicate part of it.
            # S3Gen's embed_ref uses self.t3.hp.speech_cond_prompt_len for T3, which is not right here.
            # S3Gen uses up to 10s for ref for S3Gen itself. Let's assume prompt for causal flow is also significant.
            # Max len for prompt tokens used by Causal flow (this is a guess, check model)
            max_flow_prompt_token_len = self.s3gen_model.flow.encoder.hp.get("prompt_token_max_len", 75) # Example, should come from flow model config
            
            prompt_s3_tokens = torch.zeros(max_flow_prompt_token_len, dtype=torch.long)
            prompt_s3_token_len = torch.tensor(0, dtype=torch.long)
            prompt_mel = torch.zeros(max_flow_prompt_token_len * 2, target_mel.size(1), dtype=torch.float) # (T_prompt_mel, F)
            prompt_mel_len = torch.tensor(0, dtype=torch.long)
        else:
            try:
                # Prompt S3 tokens (from 16kHz prompt audio)
                # Assuming CausalMaskedDiffWithXvec might have a max prompt length for tokens
                max_flow_prompt_token_len = getattr(self.s3gen_model.flow.encoder, 'prompt_token_max_len', 75) # Check actual attribute if exists
                prompt_s3_tokens_batch, prompt_s3_token_lengths_batch = self.s3gen_model.tokenizer.forward(
                    [prompt_wav_16k_segment.numpy()], max_len=max_flow_prompt_token_len
                )
                if prompt_s3_tokens_batch is None: return None
                prompt_s3_tokens = prompt_s3_tokens_batch.squeeze(0)
                prompt_s3_token_len = prompt_s3_token_lengths_batch.squeeze(0)

                # Prompt Mel (from 24kHz prompt audio)
                prompt_mel = self.s3gen_model.mel_extractor(prompt_wav_24k_segment.unsqueeze(0)).squeeze(0).transpose(0,1) # (T_mel_prompt, F)
                # Ensure prompt_mel length aligns with prompt_s3_tokens length (T_mel = 2 * T_tokens)
                if prompt_mel.size(0) > prompt_s3_tokens.size(0) * 2:
                    prompt_mel = prompt_mel[:prompt_s3_tokens.size(0) * 2, :]
                prompt_mel_len = torch.tensor(prompt_mel.size(0), dtype=torch.long)
                
                # If prompt_s3_tokens were truncated by max_len, adjust its length for consistency
                if prompt_mel.size(0) // 2 < prompt_s3_tokens.size(0):
                    prompt_s3_tokens = prompt_s3_tokens[:prompt_mel.size(0)//2]
                    prompt_s3_token_len = torch.tensor(prompt_s3_tokens.size(0), dtype=torch.long)


            except Exception as e:
                logger.error(f"Item {idx}: Error processing prompt features: {e}", exc_info=True)
                return None
        
        
        return {
            "speech_token": target_s3_tokens.long(),       # Target S3 tokens
            "speech_token_len": speech_token_len.long(), # Length of target S3 tokens
            "speech_feat": target_mel.float(),            # Target mel spectrogram (T_mel, F)
            "speech_feat_len": speech_feat_len.long(),    # Length of target mel (num frames)
            "embedding": speaker_embedding.float(),       # Speaker embedding (x-vector)
            "prompt_token_input": prompt_s3_tokens.long(),
            "prompt_token_len_input": prompt_s3_token_len.long(),
            "prompt_feat_input": prompt_mel.float(),            # (T_mel_prompt, F)
            # "prompt_feat_len" is implicitly prompt_mel.size(0), flow model might expect it explicitly
            # S3Gen's embed_ref sets prompt_feat_len=None, CausalFlow seems to use prompt_feat.shape[1]
        }

# --- S3Gen Data Collator ---
@dataclass
class S3GenDataCollator:
    # Padding values. Flow model might expect specific padding for mels (e.g., 0 or -log(1e-5))
    # S3 tokens are padded with 0.
    speech_token_pad_id: int = 0
    mel_pad_value: float = 0.0 # Or whatever the flow model expects for padded mel frames

    # Max lengths for prompts, if flow model expects fixed size padded prompts during training
    # This needs to match how CausalMaskedDiffWithXvec's encoder processes prompts
    # For now, assume dynamic padding for prompts in batch.
    # max_prompt_token_len_collator: Optional[int] = 75 # Example
    # max_prompt_mel_len_collator: Optional[int] = 150 # Example

    def __call__(self, features: List[Optional[Dict[str, torch.Tensor]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]
        if not valid_features: return {}
        
        batch_size = len(valid_features)
        # Device from first valid sample
        device = valid_features[0]["speech_token"].device if batch_size > 0 else "cpu"


        # Pad speech_token (target S3 tokens)
        speech_tokens = [f["speech_token"] for f in valid_features]
        max_len_st = max(s.size(0) for s in speech_tokens)
        padded_speech_tokens = torch.stack(
            [F.pad(s, (0, max_len_st - s.size(0)), value=self.speech_token_pad_id) for s in speech_tokens]
        )
        speech_token_lens = torch.stack([f["speech_token_len"] for f in valid_features])

        # Pad speech_feat (target mel spectrograms)
        # Mels are (T_mel, F). Pad along T_mel dimension (dim 0).
        speech_feats = [f["speech_feat"] for f in valid_features]
        max_len_sf = max(s.size(0) for s in speech_feats)
        mel_dim = speech_feats[0].size(1) # F dimension
        padded_speech_feats = torch.stack(
            [F.pad(s, (0, 0, 0, max_len_sf - s.size(0)), value=self.mel_pad_value) for s in speech_feats] # Pads last dim first (T_mel)
        ) # Result (B, T_mel_max, F)
        speech_feat_lens = torch.stack([f["speech_feat_len"] for f in valid_features])
        
        # Stack speaker embeddings
        embeddings = torch.stack([f["embedding"] for f in valid_features])

        # --- Pad prompt features ---
        prompt_tokens = [f["prompt_token_input"] for f in valid_features]
        # Use a fixed max prompt length if required by model, else pad to max in batch
        # target_prompt_token_len = self.max_prompt_token_len_collator or max(pt.size(0) for pt in prompt_tokens)
        target_prompt_token_len = max(pt.size(0) for pt in prompt_tokens) if prompt_tokens else 0
        target_prompt_token_len = max(1, target_prompt_token_len) # Ensure not 0

        padded_prompt_tokens = torch.stack(
             [F.pad(pt, (0, target_prompt_token_len - pt.size(0)), value=self.speech_token_pad_id) for pt in prompt_tokens]
        )
        prompt_token_lens = torch.stack([f["prompt_token_len_input"] for f in valid_features])

        prompt_feats = [f["prompt_feat_input"] for f in valid_features]
        # target_prompt_mel_len = self.max_prompt_mel_len_collator or max(pf.size(0) for pf in prompt_feats)
        target_prompt_mel_len = max(pf.size(0) for pf in prompt_feats) if prompt_feats else 0
        target_prompt_mel_len = max(1, target_prompt_mel_len)
        
        padded_prompt_feats = torch.stack(
            [F.pad(pf, (0, 0, 0, target_prompt_mel_len - pf.size(0)), value=self.mel_pad_value) for pf in prompt_feats]
        )
        # prompt_feat_lens = torch.stack([f["prompt_feat_len"] for f in valid_features]) # If dataset provides it

        # The CausalMaskedDiffWithXvec.forward in flow.py might not take all these directly
        # It was designed for inference. Its training `forward` (if like MaskedDiffWithXvec)
        # expects a `batch` dict.
        # The S3Gen model's `flow` attribute is CausalMaskedDiffWithXvec.
        # We need to construct the `batch` dict that ITS `forward` method expects for training.

        # Looking at CausalMaskedDiffWithXvec.inference, it takes:
        # token, token_len, prompt_token, prompt_token_len, prompt_feat, embedding
        # (prompt_feat_len is derived from prompt_feat.shape[1] for mel length)
        # For training, it likely expects `speech_token` as `token`, `speech_feat` as the target `feat`.
        
        # The MaskedDiffWithXvec.forward(self, batch, device) is the training entry point.
        # We need to provide a dictionary that matches this structure.
        return {
            'speech_token': padded_speech_tokens.to(device),
            'speech_token_len': speech_token_lens.to(device),
            'speech_feat': padded_speech_feats.to(device), # This is the target mel
            'speech_feat_len': speech_feat_lens.to(device),
            'embedding': embeddings.to(device),
            "prompt_token_input": padded_prompt_tokens.to(device),
            "prompt_token_len_input": prompt_token_lens.to(device),
            "prompt_feat_input": padded_prompt_feats.to(device),
            # "prompt_feat_len_input": prompt_feat_lens.to(device) # Derived from shape usually
        }


# --- Model Wrapper for S3Gen's Flow component ---
class S3GenFlowForFineTuning(torch.nn.Module):
    def __init__(self, s3gen_token_to_mel: S3Token2Mel): # Pass S3Token2Mel instance
        super().__init__()
        #self.s3_token_to_mel = s3gen_token_to_mel
        self.flow_model: CausalMaskedDiffWithXvec = s3gen_token_to_mel.flow # type: ignore
        
        # Create a dummy HF Compatible Config
        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_s3gen_flow_finetune"
            # Add any S3Gen/Flow specific hparams you want to save in config.json
            def __init__(self, **kwargs): super().__init__(**kwargs)
        self.config = HFCompatibleConfig()
        # Populate self.config with relevant params from s3_token_to_mel or flow_model if needed

    # In S3GenFlowForFineTuning.forward (Simplified Slicing)

    def forward(self,
                speech_token: torch.Tensor,
                speech_token_len: torch.Tensor,
                speech_feat: torch.Tensor,          # Target mel (B, T_target_mel_collator, F)
                speech_feat_len: torch.Tensor,
                embedding: torch.Tensor,
                prompt_token_input: torch.Tensor,
                prompt_token_len_input: torch.Tensor,
                prompt_feat_input: torch.Tensor,    # Prompt mel (B, T_prompt_mel_collator, F)
                labels = None
                ):

        # 1. Project speaker embedding
        projected_speaker_emb = self.flow_model.spk_embed_affine_layer(F.normalize(embedding, dim=1))

        # 2. Prepare 'mu' (linguistic conditioning from concatenated prompt and target tokens, then sliced for target)
        full_input_tokens = torch.cat([prompt_token_input, speech_token], dim=1)
        full_input_token_lens = prompt_token_len_input + speech_token_len
        vocab_size = getattr(self.flow_model, 'vocab_size', 6561)
        input_token_embed = self.flow_model.input_embedding(
            torch.clamp(full_input_tokens, min=0, max=vocab_size - 1)
        )
        h_encoded_full, _ = self.flow_model.encoder(input_token_embed, full_input_token_lens)
        h_projected_full = self.flow_model.encoder_proj(h_encoded_full)
        mu_full_for_cfm = h_projected_full.transpose(1, 2).contiguous()

        # 3. Prepare target mels (x1)
        target_mels_for_cfm_x = speech_feat.transpose(1, 2).contiguous() # (B, F, T_target_mel_collator)
        target_mel_mask = (~self.make_pad_mask(speech_feat_len, max_len=target_mels_for_cfm_x.size(2))).unsqueeze(1)

        # 4. Prepare 'cond' (acoustic prompt)
        # Original prompt mels: (B, F, T_prompt_mel_collator)
        original_prompt_mels = prompt_feat_input.transpose(1, 2).contiguous()
        
        # MODIFICATION: Pad 'cond' to match the time dimension of 'x1' (target_mels_for_cfm_x)
        # This assumes 'cond' provides information for the initial part, and is neutral for the rest.
        T_target_mel_collator = target_mels_for_cfm_x.size(2) # e.g., 1633
        T_prompt_mel_collator = original_prompt_mels.size(2)  # e.g., 150

        if T_prompt_mel_collator > T_target_mel_collator:
            # This case should ideally not happen if target is longer than prompt.
            # If it does, truncate the prompt to fit.
            padded_prompt_mels_for_cond = original_prompt_mels[:, :, :T_target_mel_collator]
            logger.warning(f"Prompt mel length ({T_prompt_mel_collator}) was longer than target mel length ({T_target_mel_collator}). Prompt was truncated.")
        else:
            padding_size = T_target_mel_collator - T_prompt_mel_collator
            # Pad original_prompt_mels on the time dimension (last dim)
            # Pad with zeros, assuming zeros mean "no conditioning" for the padded part.
            # The UNet should learn to interpret this.
            padded_prompt_mels_for_cond = F.pad(original_prompt_mels, (0, padding_size), mode='constant', value=0)
        # Now, padded_prompt_mels_for_cond has shape (B, F, T_target_mel_collator)

        # 5. Slice mu_full_for_cfm to get mu_target, aligned with target_mels_for_cfm_x
        num_prompt_mel_frames_collator = T_prompt_mel_collator # Using the original prompt length for slicing mu
        num_target_mel_frames_collator = T_target_mel_collator

        slice_start_idx = num_prompt_mel_frames_collator

        if slice_start_idx >= mu_full_for_cfm.size(2):
            logger.error(f"MU SLICING ERROR: mu_full_for_cfm (len {mu_full_for_cfm.size(2)}) too short for prompt part (len {slice_start_idx})")
            mu_conditioning_for_target_mels = torch.zeros_like(target_mels_for_cfm_x.float().expand(-1, mu_full_for_cfm.size(1), -1))
        else:
            mu_target_raw_slice = mu_full_for_cfm[:, :, slice_start_idx:]
            current_raw_target_slice_len = mu_target_raw_slice.size(2)
            if current_raw_target_slice_len < num_target_mel_frames_collator:
                padding_needed = num_target_mel_frames_collator - current_raw_target_slice_len
                mu_conditioning_for_target_mels = F.pad(mu_target_raw_slice, (0, padding_needed))
            elif current_raw_target_slice_len > num_target_mel_frames_collator:
                mu_conditioning_for_target_mels = mu_target_raw_slice[:, :, :num_target_mel_frames_collator]
            else:
                mu_conditioning_for_target_mels = mu_target_raw_slice

        # 6. Compute CFM loss
        cfm_loss_output, _ = self.flow_model.decoder.compute_loss(
            x1=target_mels_for_cfm_x,
            mask=target_mel_mask,
            mu=mu_conditioning_for_target_mels,
            spks=projected_speaker_emb,
            cond=padded_prompt_mels_for_cond # Use the padded cond
        )

        main_loss = cfm_loss_output
        return (main_loss, main_loss, torch.tensor(0.0, device=main_loss.device))


    def make_pad_mask(self, lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        # ... (same as before) ...
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.as_tensor(lengths, dtype=torch.long)
        if max_len is None:
            if lengths.numel() == 0: # Handle empty lengths tensor
                max_len = 0
            else:
                max_len = torch.max(lengths).item()

        bs = lengths.size(0)
        if bs == 0: # Handle batch size of 0
             return torch.empty(0, max_len, dtype=torch.bool, device=lengths.device)

        seq_range = torch.arange(0, max_len, device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand >= seq_length_expand

    


# Global trainer instance
trainer_instance: Optional[Trainer] = None

def main():
    global trainer_instance
    parser = HfArgumentParser((S3GenModelArguments, S3GenDataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    set_seed(training_args.seed)

    # --- Load Base Chatterbox Model to get S3Gen ---
    logger.info("Loading base ChatterboxMultilingualTTS model to extract S3Gen components...")
    # This part is similar to T3 finetuning: load the full model first.
    if model_args.local_model_dir:
        chatterbox_model = ChatterboxMultilingualTTS.from_local(ckpt_dir=model_args.local_model_dir, device="cpu")
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID # Fallback to default REPO_ID
        download_dir = Path(training_args.output_dir) / "pretrained_chatterbox_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download
        files_to_download = ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]
        for f in files_to_download:
            try: hf_hub_download(repo_id=repo_to_download, filename=f, local_dir=download_dir, local_dir_use_symlinks=False, cache_dir=model_args.cache_dir)
            except Exception as e: logger.warning(f"Could not download {f} from {repo_to_download}: {e}")
        chatterbox_model = ChatterboxMultilingualTTS.from_local(ckpt_dir=download_dir, device="cpu")

    s3gen_model_to_finetune: S3Gen = chatterbox_model.s3gen
    s3gen_token_to_mel_part: S3Token2Mel = s3gen_model_to_finetune # S3Gen inherits from S3Token2Wav which inherits from S3Token2Mel

    # --- Freeze parts of S3Gen ---
    # Freeze HiFiGAN part (mel2wav)
    for param in s3gen_token_to_mel_part.mel2wav.parameters():
        param.requires_grad = False
    logger.info("S3Gen HiFiGAN (mel2wav) part frozen.")

    if model_args.freeze_speaker_encoder:
        for param in s3gen_token_to_mel_part.speaker_encoder.parameters():
            param.requires_grad = False
        logger.info("S3Gen Speaker Encoder (CAMPPlus) frozen.")
    
    if model_args.freeze_s3_tokenizer:
        # S3Tokenizer in Chatterbox doesn't have trainable params in its dummy/provided version,
        # but a real one might.
        if hasattr(s3gen_token_to_mel_part.tokenizer, 'parameters'):
             for param in s3gen_token_to_mel_part.tokenizer.parameters(): # type: ignore
                param.requires_grad = False
        logger.info("S3Gen S3Tokenizer frozen (if it has parameters).")

    # Ensure the flow model part is trainable
    for param in s3gen_token_to_mel_part.flow.parameters():
        param.requires_grad = True
    logger.info("S3Gen Flow Model (CausalMaskedDiffWithXvec) set to trainable.")


    # --- Prepare Dataset ---
    logger.info("Loading and processing dataset for S3Gen finetuning...")
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    if data_args.dataset_name:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name,
                                    cache_dir=model_args.cache_dir, verification_mode=verification_mode)
        train_hf_dataset = raw_datasets[data_args.train_split_name]
        eval_hf_dataset = raw_datasets.get(data_args.eval_split_name) if data_args.eval_split_name else None
        if training_args.do_eval and not eval_hf_dataset and data_args.eval_split_size > 0 and len(train_hf_dataset) > 1:
            split_dataset = train_hf_dataset.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train_hf_dataset, eval_hf_dataset = split_dataset["train"], split_dataset["test"]
    else:
        raise ValueError("S3Gen finetuning currently requires a Hugging Face dataset_name.") # Simpler for now

    train_dataset = S3GenFineTuningDataset(data_args, s3gen_model_to_finetune, train_hf_dataset)
    eval_dataset = None
    if eval_hf_dataset and training_args.do_eval:
        eval_dataset = S3GenFineTuningDataset(data_args, s3gen_model_to_finetune, eval_hf_dataset)

    # --- Data Collator ---
    data_collator = S3GenDataCollator() # Uses defaults for padding

    # --- Model Wrapper for Trainer ---
    s3gen_flow_trainable_model = S3GenFlowForFineTuning(s3gen_token_to_mel_part)

    # --- Compute Metrics (can log individual loss components) ---
    def compute_metrics_s3gen(eval_preds):
        metrics = {}
        if isinstance(eval_preds.predictions, tuple) and len(eval_preds.predictions) >= 1:
            # eval_preds.predictions[0] is main_loss
            # eval_preds.predictions[1] is cfm_loss (if returned)
            # eval_preds.predictions[2] is reg_loss (if returned)
            # Trainer automatically logs eval_preds.predictions[0] as eval_loss
            if len(eval_preds.predictions) > 1 and eval_preds.predictions[1] is not None:
                metrics["eval_cfm_loss"] = float(np.mean(eval_preds.predictions[1]))
            if len(eval_preds.predictions) > 2 and eval_preds.predictions[2] is not None:
                metrics["eval_reg_loss"] = float(np.mean(eval_preds.predictions[2]))
        return metrics

    # --- Callbacks ---
    callbacks = []
    if training_args.early_stopping_patience is not None and training_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    # Add audio generation callback later if needed (more complex for S3Gen eval)

    # --- Trainer ---
    logger.info(f"Using dataloader_pin_memory: {training_args.dataloader_pin_memory}")
    trainer_instance = Trainer(
        model=s3gen_flow_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_s3gen if training_args.do_eval and eval_dataset else None,
        callbacks=callbacks if callbacks else None
    )
    if training_args.label_names is None: trainer_instance.label_names = [] # Model handles its own targets

    # --- Training ---
    if training_args.do_train:
        logger.info("*** Finetuning S3Gen Flow Model ***")
        train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer_instance.save_model() # Saves Trainer-wrapped S3GenFlowForFineTuning
        
        logger.info("Saving finetuned S3Gen (flow part) model weights for ChatterboxMultilingualTTS...")
        # The S3GenFlowForFineTuning model IS the s3_token_to_mel part (or its flow sub-module)
        # We need to save the state_dict of s3gen_token_to_mel_part
        # The trainer saves s3gen_flow_trainable_model.state_dict(), which contains s3_token_to_mel.xxx keys
        
        # To save in the original Chatterbox S3Gen format:
        # We need the state_dict of the s3gen_model_to_finetune, which now has updated flow params.
        finetuned_s3gen_state_dict = s3gen_model_to_finetune.state_dict()
        output_s3gen_safetensor_path = Path(training_args.output_dir) / "s3gen.safetensors"
        from safetensors.torch import save_file
        save_file(finetuned_s3gen_state_dict, output_s3gen_safetensor_path)
        logger.info(f"Finetuned S3Gen model weights saved to {output_s3gen_safetensor_path}")

        # Copy other necessary files for a complete local model dir from the *original* download
        # This assumes you want to package the finetuned s3gen with the original T3, VE etc.
        # If original_model_dir_for_copy is defined from the chatterbox_model loading:
        # if original_model_dir_for_copy: # (Define this based on initial load path)
        #     import shutil
        #     for f_name in ["ve.safetensors", "t3_cfg.safetensors", "tokenizer.json", "conds.pt"]:
        #         src_path = original_model_dir_for_copy / f_name
        #         if src_path.exists(): shutil.copy2(src_path, Path(training_args.output_dir) / f_name)
        #     logger.info(f"Other original model components copied to {training_args.output_dir}")


        metrics = train_result.metrics
        trainer_instance.log_metrics("train", metrics)
        trainer_instance.save_metrics("train", metrics)
        trainer_instance.save_state()

    # --- Evaluation ---
    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating S3Gen Flow Model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)

    logger.info("S3Gen finetuning script finished.")

if __name__ == "__main__":
    main()
