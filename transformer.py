"""Inspired by https://github.com/karpathy/minGPT"""

from dataclasses import dataclass, field
from typing import Any, Optional
from collections import defaultdict

import logging
import math
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    nb_transformer_blocks: int
    nb_attn_heads: int
    nb_embeddings: int
    vocab_size: int
    context_length: int
    dropout_attn: float = field(default=0.1)
    dropout_embeddings: float = field(default=0.1)
    dropout_resid: float = field(default=0.1)
    lw_mean: float = field(default=0.0)
    lw_std: float = field(default=0.02)

    @classmethod
    def from_file(cls, filename: str) -> "TransformerConfig":
        with open(filename, "r") as file:
            config = json.load(file)

        return cls(**config)


@dataclass
class TrainerConfig:
    device: str = field(default="auto")
    nb_workers: int = field(default=0)
    max_iters: int = field(default=None)
    batch_size: int = field(default=64)
    learning_rate: int = field(default=3e-4)
    betas: tuple[float, float] = field(default=(0.9, 0.95))
    weight_decay: float = field(default=0.1)
    grad_norm_clip: float = field(default=1.0)

    @classmethod
    def from_file(cls, filename: str) -> "TrainerConfig":
        with open(filename, "r") as file:
            config = json.load(file)

        return cls(**config)


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.nb_embeddings % config.nb_attn_heads == 0, "nb_embeddings must be divisible by nb_attn_heads"
        self.config = config

        self.c_attn = nn.Linear(self.config.nb_embeddings, 3 * self.config.nb_embeddings)  # KQV projections
        self.c_proj = nn.Linear(self.config.nb_embeddings, self.config.nb_embeddings)  # Output projection
        self.dropout_attn = nn.Dropout(self.config.dropout_attn)
        self.dropout_resid = nn.Dropout(self.config.dropout_resid)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.config.context_length, self.config.context_length))
            .view(1, 1, self.config.context_length, self.config.context_length)
        )

    def forward(self, x):
        batch_size, seq_length, nb_embedding = x.size()

        q, k, v = self.c_attn(x).split(self.config.nb_embeddings, dim=2)
        qkv_view_size = (batch_size, seq_length, self.config.nb_attn_heads, nb_embedding // self.config.nb_attn_heads)
        k = k.view(qkv_view_size).transpose(1, 2)
        q = q.view(qkv_view_size).transpose(1, 2)
        v = v.view(qkv_view_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :seq_length, :seq_length] == 0, -torch.inf)
        att = F.softmax(att, dim=-1)
        att = self.dropout_attn(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, nb_embedding)
        y = self.dropout_resid(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.nb_embeddings)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.nb_embeddings)
        self._mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(config.nb_embeddings, 4 * config.nb_embeddings),
            c_proj=nn.Linear(4 * config.nb_embeddings, config.nb_embeddings),
            act=NewGELU(),
            dropout=nn.Dropout(config.dropout_embeddings),
        ))
        self.mlp = lambda x: self._mlp.dropout(self._mlp.c_proj(self._mlp.act(self._mlp.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(torch.nn.Module):

    def __init__(self, config: TransformerConfig, log_level: str):
        super().__init__()

        logger.setLevel(log_level)

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.config.vocab_size, self.config.nb_embeddings),
            wpe=nn.Embedding(self.config.context_length, self.config.nb_embeddings),
            dropout=nn.Dropout(self.config.dropout_resid),
            blocks=nn.ModuleList([TransformerBlock(self.config) for _ in range(self.config.nb_transformer_blocks)]),
            ln_f=nn.LayerNorm(self.config.nb_embeddings),
        ))
        self.output = nn.Linear(self.config.nb_embeddings, self.config.vocab_size, bias=False)

        # Initialise weights using special initialisation for residual projections (from GPT-2 paper)
        self.apply(lambda x: self._init_weights(x, mean=self.config.lw_mean, std=self.config.lw_std))
        for param_name, param in self.named_parameters():
            if param_name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2 * self.config.nb_transformer_blocks))

        n_params = sum(param.numel() for param in self.transformer.parameters())

        logger.info(f"Transformer : Number of parameters {(n_params/1e6):.2f}M")

    @staticmethod
    def _init_weights(module: torch.nn.Module, mean: float, std: float) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, token_sequence, targets: Optional[torch.Tensor] = None):
        logger.debug(f"Transformer : Token sequence size {token_sequence.size()} \t " +
                     f"Targets size {targets.size() if targets is not None else None}")

        device = token_sequence.device
        batch_size, nb_tokens = token_sequence.size()
        assert nb_tokens <= self.config.context_length, \
            f"Cannot forward sequence of length {nb_tokens}, block size is only {self.config.context_length}"
        pos = torch.arange(0, nb_tokens, dtype=torch.long, device=device).unsqueeze(0)

        token_embeddings = self.transformer.wte(token_sequence)
        position_embeddings = self.transformer.wpe(pos)

        logger.debug(f"Transformer : Token embeddings size {token_embeddings.size()} \t " +
                     f"Position embeddings size {position_embeddings.size()}")

        x = self.transformer.dropout(token_embeddings + position_embeddings)
        for block in self.transformer.blocks:
            x = block(x)
            logger.debug(f"Transformer : Block output size {x.size()}")

        x = self.transformer.ln_f(x)

        logger.debug(f"Transformer : Layer norm output size {x.size()}")

        logits = self.output(x)
        logger.debug(f"Transformer : Logits output size {logits.size()}")

        logger.debug(f"Transformer : Cross entropy logits size {logits.view(-1, logits.size(-1)).size()} \t " +
                     f"Targets size {targets.view(-1).size() if targets is not None else None}")

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        ) if targets is not None else None

        return logits, loss

    @torch.no_grad()
    def generate(
            self, token_sequence: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
            do_sample: bool = False, top_k: Optional[int] = None
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            token_sequence_sliced = token_sequence if token_sequence.size(1) <= self.config.context_length \
                else token_sequence[:, -self.config.context_length:]

            logger.debug(f"generate : Token sequence sliced size {token_sequence_sliced.size()}")

            logits, _ = self(token_sequence_sliced)
            logger.debug(f"generate : logits size {logits.size()}")

            logits = logits[:, -1, :] / temperature
            logger.debug(f"generate : new temperature-adjusted logits size {logits.size()}")

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -torch.inf
                logger.debug(f"generate : Top K size {v.size()}")

            probs = F.softmax(logits, dim=-1)

            logger.debug(f"generate : Probabilities size {probs.size()}")

            if do_sample:
                token_next = torch.multinomial(probs, num_samples=1)
            else:
                _, token_next = torch.topk(probs, k=1, dim=-1)

            token_sequence = torch.cat((token_sequence, token_next), dim=1)

            logger.debug(f"generate : Token sequence output size {token_sequence.size()}")

        return token_sequence


class Trainer:
    """Trainer class
    This class carries out conversion of dataset into a Torch DataLoader and training of GPT

    Args:
        config: TrainerConfig containing parameters for training
        model: GPT
        dataset: size (N, C)  N --> number of sequences, C --> context length, B --> batch size, V --> vocab size
                 this is converted into a dataset of size (B, C, V)

    """

    def __init__(self, config: TrainerConfig, model: Transformer, dataset: Any):
        self.config = config
        self.model = model
        self.optimizer = None
        self.dataset = dataset
        self.callbacks = defaultdict(list)

        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.model = self.model.to(self.device)

        logger.info(f"Transformer : Running on device {self.device}")

        self.batch_number = 0
        self.batch_start_time = 0.0
        self.batch_time = 0.0

    def configure_optimizers(self, trainer_config: TrainerConfig) -> None:
        decay = set()  # Parameters which will have decay
        no_decay = set()  # Parameters which will not have decay
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for module_name, module in self.model.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = f"{module_name}.{param_name}" if module_name else param_name

                if param_name.endswith("bias") or \
                        (param_name.endswith("weight") and isinstance(module, blacklist_weight_modules)):
                    no_decay.add(full_param_name)

                if param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_param_name)

        # Validate parameters split correctly between decay and no decay sets
        param_dict = {param_name: param for param_name, param in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, f"Parameters {inter_params} found in both sets"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters {str(param_dict.keys() - union_params)} were not separated into decay/no_decay sets"

        optim_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(decay))],
                "weight_decay": trainer_config.weight_decay
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]

        self.optimizer = torch.optim.AdamW(optim_groups, lr=trainer_config.learning_rate, betas=trainer_config.betas)

    def add_callback(self, event: str, callback):
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        self.callbacks[event] = [callback]

    def trigger_callbacks(self, event: str):
        for callback in self.callbacks.get(event, []):
            callback(self)

    def run(self) -> torch.nn.Module:
        logger.debug("run : Training started")

        self.configure_optimizers(self.config)

        logger.debug("run : Configured optimizer")

        train_loader = DataLoader(
            self.dataset,
            sampler=torch.utils.data.RandomSampler(self.dataset, replacement=True, num_samples=int(1e3)),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.nb_workers,
        )

        logger.debug("run : Created dataloader")

        self.model.train()
        self.batch_number = 0
        self.batch_start_time = time.time()
        data_iter = iter(train_loader)
        while True:
            logger.info(f"run : Batch number {self.batch_number}")

            try:
                batch = next(data_iter)
            except StopIteration:
                logger.debug("run : End of dataloader, recreating")
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(self.device) for t in batch]
            x, y = batch
            logger.debug(f"run : x size {x.size()} \t y size {y.size()}")

            logits, self.loss = self.model(x, y)
            logger.debug(f"run : logits size {logits.size()} \t loss size {self.loss.size()}")

            self.model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks("on_batch_end")
            self.batch_number += 1
            batch_end_time = time.time()
            self.batch_time = batch_end_time - self.batch_start_time
            self.batch_start_time = batch_end_time

            if self.config.max_iters and self.batch_number >= self.config.max_iters:
                break

        return self.model
