"""Test Transformer model"""


from transformer import TransformerConfig, Transformer, Trainer, TrainerConfig

import torch


transformer_config = TransformerConfig.from_file("gpt2.json")
transformer_config.context_length = 16
transformer_config.nb_embeddings = 96
print(transformer_config)

trainer_config = TrainerConfig()
trainer_config.nb_workers = 0
trainer_config.max_iters = 10
print(trainer_config)

# Instantiate GPT model
gpt = Transformer(transformer_config, log_level="INFO")

# Instantiate Trainer
trainer = Trainer(
    config=trainer_config,
    model=gpt,
    dataset=[(torch.randint(0, 2, (transformer_config.context_length,)),
              torch.randint(0, 2, (transformer_config.context_length,))) for _ in range(10)]
)

# Start training
trainer.run()

# Generate response
NB_TOKENS = 100
response = gpt.generate(torch.randint(0, 2, (24, transformer_config.context_length,)).to("cuda"), NB_TOKENS)
print(f"Response size when generating {NB_TOKENS} tokens for C={transformer_config.context_length}: {response.size()}")
print(response)

# See size of model output
print("Model output size: ", gpt(torch.randint(0, 2, (24, transformer_config.context_length,)).to("cuda"))[0].size())
