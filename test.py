from transformer import TransformerConfig, Transformer, Trainer, TrainerConfig

import torch

transformer_config = TransformerConfig.from_file("gpt2.json")
transformer_config.context_length = 16
transformer_config.nb_embeddings = 96
print(transformer_config)

gpt = Transformer(transformer_config)

trainer_config = TrainerConfig()
trainer_config.nb_workers = 0
trainer_config.max_iters = 10
print(trainer_config)

# We are training a GPT model using an input dataset of shape X --> (CONTEXT LENGTH,) and Y --> (CONTEXT LENGTH,)
# Before training this dataset becomes the actual train dataset of shape (BATCH_SIZE, CONTEXT_LENGTH, VOCAB_SIZE)
# The input for inference is of shape (BATCH SIZE, CONTEXT LENGTH) and the output from inference is of shape (BATCH_SIZE, CONTEXT_LENGTH, VOCAB SIZE)

trainer = Trainer(trainer_config, gpt, [(torch.randint(0, 2, (transformer_config.context_length,)), torch.randint(0, 2, (transformer_config.context_length,))) for _ in range(10)])
trainer.run()

gen = gpt.generate(torch.randint(0, 2, (24, transformer_config.context_length,)), 100)

print(gen.shape)
for n in gen.numpy():
    print(n)

print(gpt(torch.randint(0, 2, (24, transformer_config.context_length,)))[0].size())