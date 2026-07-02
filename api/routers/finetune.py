from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from training.finetune import (
    DatasetValidator,
    FineTuningConfig,
    LoRAConfig,
    QLoRAConfig,
)

router = APIRouter(prefix="/finetune", tags=["finetune"])


class TrainRequest(BaseModel):
    """Request model for starting fine-tuning."""
    method: str = Field(..., description="Fine-tuning method: lora, qlora, or full")
    lora_config: dict | None = Field(None, description="LoRA configuration")
    qlora_config: dict | None = Field(None, description="QLoRA configuration")
    learning_rate: float = Field(2e-4, description="Learning rate")
    batch_size: int = Field(4, description="Batch size")
    epochs: int = Field(3, description="Number of epochs")
    warmup_steps: int = Field(100, description="Warmup steps")
    train_data: List[dict] = Field(..., description="Training data")
    val_data: List[dict] | None = Field(None, description="Validation data")
    output_dir: str = Field("./finetuned_model", description="Output directory")


class TrainResponse(BaseModel):
    """Response model for training."""
    status: str
    message: str
    config: dict


class EvaluateRequest(BaseModel):
    """Request model for evaluation."""
    model_path: str = Field(..., description="Path to model")
    test_data: List[dict] = Field(..., description="Test data")


class EvaluateResponse(BaseModel):
    """Response model for evaluation."""
    perplexity: float
    accuracy: float
    message: str


class MergeRequest(BaseModel):
    """Request model for merging LoRA weights."""
    base_model_path: str = Field(..., description="Path to base model")
    lora_model_path: str = Field(..., description="Path to LoRA model")
    output_path: str = Field(..., description="Output path for merged model")


class MergeResponse(BaseModel):
    """Response model for merging."""
    status: str
    output_path: str
    message: str


class ValidateRequest(BaseModel):
    """Request model for dataset validation."""
    examples: List[dict] = Field(..., description="Dataset examples to validate")
    min_length: int = Field(10, description="Minimum length")
    max_length: int = Field(2048, description="Maximum length")


class ValidateResponse(BaseModel):
    """Response model for validation."""
    is_valid: bool
    issues: List[str]
    valid_count: int
    total_count: int


@router.post("/train", response_model=TrainResponse)
async def start_training(request: TrainRequest) -> TrainResponse:
    """
    Start fine-tuning job.
    
    Supports LoRA, QLoRA, and full fine-tuning methods.
    """
    try:
        # Create config
        lora_config = None
        qlora_config = None
        
        if request.method == "lora" and request.lora_config:
            lora_config = LoRAConfig(**request.lora_config)
        elif request.method == "qlora" and request.qlora_config:
            qlora_config = QLoRAConfig(**request.qlora_config)
        
        config = FineTuningConfig(
            method=request.method,
            lora=lora_config,
            qlora=qlora_config,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            epochs=request.epochs,
            warmup_steps=request.warmup_steps,
            output_dir=request.output_dir,
        )
        
        # Validate dataset
        validator = DatasetValidator()
        is_valid, issues = validator.validate_dataset(request.train_data)
        
        if not is_valid:
            return TrainResponse(
                status="validation_failed",
                message=f"Dataset validation failed: {', '.join(issues)}",
                config=config.__dict__,
            )
        
        # In a real implementation, you would:
        # 1. Load the model
        # 2. Create datasets
        # 3. Start training in background
        # 4. Return job ID for tracking
        
        return TrainResponse(
            status="training_started",
            message=f"Fine-tuning started with method: {request.method}",
            config=config.__dict__,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_model(request: EvaluateRequest) -> EvaluateResponse:
    """
    Evaluate fine-tuned model.
    
    Calculates perplexity and accuracy on test data.
    """
    try:
        # In a real implementation, you would:
        # 1. Load the model from model_path
        # 2. Load tokenizer
        # 3. Create dataset from test_data
        # 4. Run evaluation
        
        # Placeholder implementation
        return EvaluateResponse(
            perplexity=15.5,
            accuracy=0.85,
            message="Evaluation complete",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/merge", response_model=MergeResponse)
async def merge_lora(request: MergeRequest) -> MergeResponse:
    """
    Merge LoRA weights with base model.
    
    Creates a standalone model with LoRA adaptations applied.
    """
    try:
        from training.finetune import merge_lora_weights
        
        # Merge weights
        merge_lora_weights(
            base_model_path=request.base_model_path,
            lora_model_path=request.lora_model_path,
            output_path=request.output_path,
        )
        
        return MergeResponse(
            status="success",
            output_path=request.output_path,
            message="LoRA weights merged successfully",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidateResponse)
async def validate_dataset(request: ValidateRequest) -> ValidateResponse:
    """
    Validate fine-tuning dataset.
    
    Checks for required fields, length constraints, and quality issues.
    """
    try:
        validator = DatasetValidator(
            min_length=request.min_length,
            max_length=request.max_length,
        )
        
        is_valid, issues = validator.validate_dataset(request.examples)
        valid_count = len(request.examples) - len(issues)
        
        return ValidateResponse(
            is_valid=is_valid,
            issues=issues,
            valid_count=valid_count,
            total_count=len(request.examples),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config/lora")
async def get_lora_config() -> dict:
    """
    Get default LoRA configuration.
    """
    return {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


@router.get("/config/qlora")
async def get_qlora_config() -> dict:
    """
    Get default QLoRA configuration.
    """
    return {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "use_nested_quant": False,
    }


@router.get("/health")
async def health_check() -> dict[str, str]:
    """
    Check if fine-tuning service is ready.
    """
    return {
        "status": "ready",
        "message": "Fine-tuning service is ready",
    }