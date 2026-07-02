from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

import torch

from config import InferenceConfig
from generation.sampler import top_k_top_p_filter


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    stop_tokens: list[int] = field(default_factory=list)
    max_context_len: int | None = None


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user" or "assistant"
    content: str


class ConversationMemory:
    """Manages conversation history with context window management."""

    def __init__(self, max_turns: int = 10, tokenizer=None):
        self.max_turns = max_turns
        self.tokenizer = tokenizer
        self.history: list[ConversationTurn] = []

    def add_turn(self, role: str, content: str) -> None:
        """Add a new turn to the conversation."""
        self.history.append(ConversationTurn(role=role, content=content))
        # Trim history if needed
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def get_context(self, include_last_assistant: bool = False) -> str:
        """Get formatted conversation context."""
        turns = self.history
        if not include_last_assistant and turns and turns[-1].role == "assistant":
            turns = turns[:-1]
        
        context_parts = []
        for turn in turns:
            if turn.role == "user":
                context_parts.append(f"User: {turn.content}")
            else:
                context_parts.append(f"Assistant: {turn.content}")
        
        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []

    def __len__(self) -> int:
        return len(self.history)


class InferenceEngine:
    """Complete inference engine with multiple generation strategies."""

    def __init__(self, model, tokenizer, config: InferenceConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        self.device = next(model.parameters()).device
        self.model.eval()

    def _prepare_inputs(self, prompt: str, max_context_len: int | None = None) -> torch.Tensor:
        """Prepare input tokens from prompt."""
        tokens = self.tokenizer.encode(prompt)
        if not tokens:
            tokens = [self.tokenizer.vocab[self.tokenizer.eos_token]]
        
        # Apply max context window
        if max_context_len is not None and len(tokens) > max_context_len:
            tokens = tokens[-max_context_len:]
        
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if temperature == 0.0:
            # Greedy decoding
            return logits
        return logits / max(temperature, 1e-9)

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to discourage repeating tokens."""
        if penalty == 1.0:
            return logits
        
        # Get unique tokens from input
        unique_tokens = input_ids.unique()
        
        # Apply penalty
        for token in unique_tokens:
            if logits[0, token] > 0:
                logits[0, token] /= penalty
            else:
                logits[0, token] *= penalty
        
        return logits

    def _check_stop_tokens(self, token: int, stop_tokens: list[int]) -> bool:
        """Check if generation should stop."""
        return token in stop_tokens

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        stop_tokens: list[int] | None = None,
        max_context_len: int | None = None,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0 for greedy)
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of token IDs to stop generation
            max_context_len: Maximum context length
            do_sample: Whether to use sampling (False for greedy)

        Returns:
            Generated text
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        stop_tokens = stop_tokens if stop_tokens is not None else self.config.stop_tokens
        max_context_len = max_context_len if max_context_len is not None else self.config.max_context_len

        input_ids = self._prepare_inputs(prompt, max_context_len)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model output
                logits = self.model(input_ids, use_cache=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Get next token logits
                next_token_logits = logits[:, -1, :]

                # Apply repetition penalty
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, input_ids, repetition_penalty
                )

                # Apply temperature
                next_token_logits = self._apply_temperature(next_token_logits, temperature)

                # Apply top-k and top-p filtering
                if do_sample and (top_k is not None or top_p is not None):
                    next_token_logits = top_k_top_p_filter(
                        next_token_logits, top_k=top_k, top_p=top_p
                    )

                # Sample or greedy decode
                if do_sample and temperature > 0:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Check stop tokens
                if self._check_stop_tokens(next_token.item(), stop_tokens):
                    break

                # Append token
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Apply max context window
                if max_context_len is not None and input_ids.size(1) > max_context_len:
                    input_ids = input_ids[:, -max_context_len:]

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        return generated_text

    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_tokens: list[int] | None = None,
        max_context_len: int | None = None,
    ) -> Iterator[str]:
        """
        Stream generated text token by token.

        Yields:
            Generated text chunks
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        stop_tokens = stop_tokens if stop_tokens is not None else self.config.stop_tokens
        max_context_len = max_context_len if max_context_len is not None else self.config.max_context_len

        input_ids = self._prepare_inputs(prompt, max_context_len)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model output
                logits = self.model(input_ids, use_cache=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Get next token logits
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                next_token_logits = self._apply_temperature(next_token_logits, temperature)

                # Apply top-k and top-p filtering
                if top_k is not None or top_p is not None:
                    next_token_logits = top_k_top_p_filter(
                        next_token_logits, top_k=top_k, top_p=top_p
                    )

                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check stop tokens
                if self._check_stop_tokens(next_token.item(), stop_tokens):
                    break

                # Append token
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Apply max context window
                if max_context_len is not None and input_ids.size(1) > max_context_len:
                    input_ids = input_ids[:, -max_context_len:]

                # Yield decoded token
                token_text = self.tokenizer.decode([next_token.item()])
                yield token_text

    def beam_search(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        beam_width: int = 4,
        stop_tokens: list[int] | None = None,
        max_context_len: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Generate text using beam search.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            beam_width: Number of beams
            stop_tokens: List of token IDs to stop generation
            max_context_len: Maximum context length

        Returns:
            List of (score, text) tuples sorted by score
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        stop_tokens = stop_tokens if stop_tokens is not None else self.config.stop_tokens
        max_context_len = max_context_len or self.config.max_context_len

        input_ids = self._prepare_inputs(prompt, max_context_len)
        
        # Initialize beams: (sequence, score)
        beams = [(input_ids, 0.0)]
        completed_beams = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                new_beams = []
                
                for seq, score in beams:
                    # Get model output
                    logits = self.model(seq, use_cache=False)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    # Get next token logits
                    next_token_logits = logits[:, -1, :]
                    
                    # Get top beam_width candidates
                    probs = torch.softmax(next_token_logits, dim=-1)
                    top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)
                    
                    for i in range(beam_width):
                        token_id = top_indices[0, i].item()
                        token_prob = top_probs[0, i].item()
                        
                        # Update score (log probability)
                        new_score = score + torch.log(torch.tensor(token_prob)).item()
                        
                        # Create new sequence
                        new_seq = torch.cat([seq, top_indices[:, i : i + 1]], dim=1)
                        
                        # Check if completed
                        if token_id in stop_tokens:
                            completed_beams.append((new_score, new_seq))
                        else:
                            new_beams.append((new_seq, new_score))
                
                # Keep top beam_width beams
                new_beams.sort(key=lambda x: x[1], reverse=True)
                beams = new_beams[:beam_width]
                
                # Stop if all beams completed
                if not beams:
                    break
        
        # Add remaining beams to completed
        for seq, score in beams:
            completed_beams.append((score, seq))
        
        # Sort by score
        completed_beams.sort(key=lambda x: x[0], reverse=True)
        
        # Decode sequences
        results = []
        for score, seq in completed_beams[:beam_width]:
            generated_tokens = seq[0].tolist()[len(input_ids[0]) :]
            text = self.tokenizer.decode(generated_tokens)
            results.append((score, text))
        
        return results

    def batch_generate(
        self, prompts: list[str], **kwargs
    ) -> list[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters

        Returns:
            List of generated texts
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def batch_stream_generate(
        self, prompts: list[str], **kwargs
    ) -> list[Iterator[str]]:
        """
        Stream generation for multiple prompts.

        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters

        Returns:
            List of iterators yielding generated text
        """
        return [self.stream_generate(prompt, **kwargs) for prompt in prompts]


class StreamingInferenceEngine(InferenceEngine):
    """Inference engine with optimized streaming support."""

    def __init__(self, model, tokenizer, config: InferenceConfig | None = None):
        super().__init__(model, tokenizer, config)
        self.kv_cache = {}

    def _get_cache_key(self, prompt: str) -> str:
        """Get cache key for a prompt."""
        return str(hash(prompt))

    def stream_generate_with_cache(
        self,
        prompt: str,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream generation with KV cache for faster subsequent generations.
        
        Note: This is a simplified version. Full KV cache requires model support.
        """
        yield from self.stream_generate(prompt, **kwargs)


class ConversationEngine(InferenceEngine):
    """Inference engine optimized for conversational AI."""

    def __init__(self, model, tokenizer, config: InferenceConfig | None = None):
        super().__init__(model, tokenizer, config)
        self.memory = ConversationMemory(tokenizer=tokenizer)
        self.system_prompt = ""

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt for the conversation."""
        self.system_prompt = prompt

    def chat(self, user_message: str, **kwargs) -> str:
        """
        Generate response to user message in conversation context.

        Args:
            user_message: User's message
            **kwargs: Generation parameters

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.memory.add_turn("user", user_message)
        
        # Build prompt with context
        context = self.memory.get_context()
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{context}\nAssistant:"
        else:
            prompt = f"{context}\nAssistant:"
        
        # Generate response
        response = self.generate(prompt, **kwargs)
        
        # Add assistant response to history
        self.memory.add_turn("assistant", response)
        
        return response

    def stream_chat(self, user_message: str, **kwargs) -> Iterator[str]:
        """
        Stream response to user message.

        Yields:
            Response text chunks
        """
        # Add user message to history
        self.memory.add_turn("user", user_message)
        
        # Build prompt with context
        context = self.memory.get_context()
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{context}\nAssistant:"
        else:
            prompt = f"{context}\nAssistant:"
        
        # Stream response
        full_response = ""
        for chunk in self.stream_generate(prompt, **kwargs):
            full_response += chunk
            yield chunk
        
        # Add assistant response to history
        self.memory.add_turn("assistant", full_response)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.memory.clear()