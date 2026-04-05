"""
Configuration for Compressed Context Memory (CCM) training and inference.

Based on the CCM paper (ICLR 2024): "Compressed Context Memory for Online Language Model Interaction"
Reference: https://github.com/snu-mllab/context-memory
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    compression_token: str = "<SUM>"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration matching CCM paper Table 14."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DataConfig:
    dataset_name: str = "knkarthick/dialogsum"
    max_input_length: int = 400
    max_target_length: int = 100


@dataclass
class TrainingConfig:
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_length: int = 512
    output_dir: str = "./compressed_context_model"
    log_interval: int = 50
    save_best: bool = True


@dataclass
class InferenceConfig:
    max_new_tokens: int = 100
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9


@dataclass
class CCMConfig:
    """Top-level configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
