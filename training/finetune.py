from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 8  # Rank
    lora_alpha: int = 16  # Scaling factor
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    use_nested_quant: bool = False


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning."""

    method: str = "lora"  # "lora", "qlora", "full"
    lora: LoRAConfig | None = None
    qlora: QLoRAConfig | None = None
    learning_rate: float = 2e-4
    batch_size: int = 4
    epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_steps: int | None = None
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    output_dir: str = "./finetuned_model"
    max_seq_length: int = 512
    instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n"


class LoRALayer(nn.Module):
    """LoRA layer for low-rank adaptation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        result = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return result


def apply_lora_to_linear(
    layer: nn.Linear,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """Apply LoRA to a linear layer."""
    in_features = layer.in_features
    out_features = layer.out_features
    lora_layer = LoRALayer(in_features, out_features, r, lora_alpha, lora_dropout)

    # Freeze original layer
    for param in layer.parameters():
        param.requires_grad = False

    # Create wrapper
    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, lora: LoRALayer) -> None:
            super().__init__()
            self.base = base
            self.lora = lora

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.lora(x)

    return LoRALinear(layer, lora_layer)


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Apply LoRA to model layers.

    Freezes all base model parameters and only keeps LoRA adapter
    parameters trainable.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_adapt = any(target in name for target in config.target_modules)
            if should_adapt:
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)

                lora_linear = apply_lora_to_linear(
                    module,
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                )
                setattr(parent, child_name, lora_linear)

    return model


class QLoRALayer(LoRALayer):
    """QLoRA layer with 4-bit quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__(in_features, out_features, r, lora_alpha, lora_dropout)


def quantize_model_4bit(model: nn.Module, compute_dtype: str = "float16") -> nn.Module:
    """Quantize model to 4-bit."""
    try:
        import importlib.util

        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError("Install bitsandbytes for QLoRA support")
        if importlib.util.find_spec("transformers") is None:
            raise ImportError("Install transformers for QLoRA support")

        from transformers import BitsAndBytesConfig

        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        resolved_dtype = compute_dtype_map.get(compute_dtype, torch.float16)

        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=resolved_dtype,
            bnb_4bit_use_double_quant=True,
        )

        for param in model.parameters():
            param.data = param.data.to(resolved_dtype)

        return model
    except ImportError:
        raise ImportError("Install bitsandbytes and transformers for QLoRA support")


def apply_qlora_to_model(model: nn.Module, config: QLoRAConfig) -> nn.Module:
    """Apply QLoRA to model."""
    # First quantize the model
    model = quantize_model_4bit(model, config.bnb_4bit_compute_dtype)

    # Then apply LoRA
    lora_config = LoRAConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=config.task_type,
    )

    model = apply_lora_to_model(model, lora_config)
    return model


class InstructionDataset(Dataset):
    """Dataset for instruction tuning."""

    def __init__(
        self,
        examples: list[dict[str, str]],
        tokenizer: Any,
        max_length: int = 512,
        instruction_template: str = "### Instruction:\n{instruction}\n\n### Response:\n",
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction_template = instruction_template

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format prompt
        instruction = example.get("instruction", "")
        response = example.get("response", example.get("output", ""))

        prompt = self.instruction_template.format(instruction=instruction)
        full_text = prompt + response

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()

        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SupervisedFineTuner:
    """Supervised fine-tuning trainer."""

    def __init__(
        self,
        model: nn.Module,
        config: FineTuningConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        tokenizer: Any = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

        # Apply fine-tuning method
        if config.method == "lora":
            lora_config = config.lora or LoRAConfig()
            self.model = apply_lora_to_model(model, lora_config)
        elif config.method == "qlora":
            qlora_config = config.qlora or QLoRAConfig()
            self.model = apply_qlora_to_model(model, qlora_config)

        # Setup data loaders
        use_cuda = torch.cuda.is_available()
        num_workers = min(4, 2) if use_cuda else 0
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_cuda,
            )
        else:
            self.val_loader: DataLoader | None = None

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train(self) -> None:
        """Train the model."""
        self.model.train()
        total_steps = len(self.train_loader) * self.config.epochs

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                # Forward pass
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                outputs = self.model(input_ids=input_ids)
                logits = (
                    outputs.logits
                    if hasattr(outputs, "logits")
                    else (outputs if isinstance(outputs, torch.Tensor) else outputs[0])
                )

                # Compute loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1
                epoch_loss += loss.item()

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    print(
                        f"Epoch {epoch+1}/{self.config.epochs} "
                        f"Step {self.global_step}/{total_steps} "
                        f"Loss: {loss.item():.4f}"
                    )

                # Validation
                if self.val_loader and self.global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate()
                    self.model.train()

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_model("best")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_model(f"checkpoint_{self.global_step}")

                # Max steps check
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    print(f"Reached max steps: {self.global_step}")
                    self.save_model("final")
                    return

            # End of epoch
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} complete - Average Loss: {avg_loss:.4f}")

            if self.val_loader:
                val_loss = self.evaluate()
                self.model.train()

        # Save final model
        self.save_model("final")

    def evaluate(self) -> float:
        """Evaluate the model."""
        if not self.val_loader:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                outputs = self.model(input_ids=input_ids)
                logits = (
                    outputs.logits
                    if hasattr(outputs, "logits")
                    else (outputs if isinstance(outputs, torch.Tensor) else outputs[0])
                )

                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else float("inf")
        print(f"Validation Loss: {avg_loss:.4f}")

        return avg_loss

    def save_model(self, name: str) -> None:
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / name
        torch.save(self.model.state_dict(), model_path)

        # Save config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)

        print(f"Model saved: {model_path}")

    def merge_and_save(self, output_path: str | None = None) -> None:
        """Merge LoRA weights with base model and save."""
        resolved_path = Path(output_path or self.config.output_dir)
        resolved_path.mkdir(parents=True, exist_ok=True)

        # Get merged state dict
        merged_state = {}

        for name, param in self.model.named_parameters():
            merged_state[name] = param.data

        # Save merged model
        torch.save(merged_state, resolved_path / "merged_model.pt")

        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(resolved_path)

        print(f"Merged model saved to: {resolved_path}")


class DatasetValidator:
    """Validate fine-tuning dataset."""

    def __init__(self, min_length: int = 10, max_length: int = 2048) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.issues: list[str] = []

    def validate_example(self, example: dict[str, str]) -> bool:
        """Validate a single example."""
        is_valid = True

        # Check required fields
        if "instruction" not in example and "input" not in example:
            self.issues.append("Missing 'instruction' or 'input' field")
            is_valid = False

        if "response" not in example and "output" not in example:
            self.issues.append("Missing 'response' or 'output' field")
            is_valid = False

        # Check lengths
        instruction = example.get("instruction", example.get("input", ""))
        response = example.get("response", example.get("output", ""))

        if len(instruction) < self.min_length:
            self.issues.append(f"Instruction too short: {len(instruction)} chars")
            is_valid = False

        if len(response) < self.min_length:
            self.issues.append(f"Response too short: {len(response)} chars")
            is_valid = False

        if len(instruction) > self.max_length:
            self.issues.append(f"Instruction too long: {len(instruction)} chars")
            is_valid = False

        if len(response) > self.max_length:
            self.issues.append(f"Response too long: {len(response)} chars")
            is_valid = False

        return is_valid

    def validate_dataset(
        self, examples: list[dict[str, str]]
    ) -> tuple[bool, list[str]]:
        """Validate entire dataset."""
        self.issues = []
        valid_count = 0

        for i, example in enumerate(examples):
            if self.validate_example(example):
                valid_count += 1

        total = len(examples)
        valid_ratio = valid_count / total if total > 0 else 0

        if valid_ratio < 0.9:
            self.issues.append(
                f"Too many invalid examples: {valid_count}/{total} valid ({valid_ratio:.1%})"
            )

        return len(self.issues) == 0, self.issues

    def get_report(self) -> str:
        """Get validation report."""
        if not self.issues:
            return "Dataset validation passed"
        return "\n".join([f"- {issue}" for issue in self.issues])


class Evaluator:
    """Evaluate fine-tuned model."""

    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate_perplexity(self, dataset: Dataset) -> float:
        """Calculate perplexity on dataset."""
        self.model.eval()
        total_loss = 0.0
        count = 0

        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    logits = (
                        outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                    )
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )

                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else float("inf")
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def evaluate_accuracy(self, dataset: Dataset) -> float:
        """Calculate accuracy on dataset."""
        self.model.eval()
        correct = 0
        total = 0

        loader = DataLoader(dataset, batch_size=8, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids)
                logits = (
                    outputs.logits
                    if hasattr(outputs, "logits")
                    else (outputs if isinstance(outputs, torch.Tensor) else outputs[0])
                )

                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def generate_response(
        self,
        instruction: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate response for instruction."""
        self.model.eval()

        # Format prompt
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore[operator]
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response part
        if "### Response:\n" in response:
            response = response.split("### Response:\n")[-1].strip()

        return response


def merge_lora_weights(
    base_model_path: str, lora_model_path: str, output_path: str
) -> None:
    """Merge LoRA weights with base model."""
    # Load base model
    base_state = torch.load(base_model_path, map_location="cpu")

    # Load LoRA weights
    lora_state = torch.load(lora_model_path, map_location="cpu")

    # Merge weights
    merged_state = base_state.copy()

    for key, lora_param in lora_state.items():
        if "lora_A" in key or "lora_B" in key:
            # This is a simplified merge
            # In practice, you'd compute: W' = W + BA * scaling
            base_key = key.replace("lora_A", "weight").replace("lora_B", "weight")
            if base_key in merged_state:
                merged_state[base_key] = merged_state[base_key] + lora_param

    # Save merged model
    torch.save(merged_state, output_path)
    print(f"Merged model saved to: {output_path}")
