import torch
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("huggyllama/llama-7b")
visualizer("Plants create energy through a process known as")
# quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
# model = AutoModelForCausalLM.from_pretrained(
#     "huggyllama/llama-7b",
#     dtype=torch.bfloat16,
#     quantization_config=quantization_config,
# )

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# input_ids = tokenizer(
#     "Plants create energy through a process known as", return_tensors="pt"
# ).to(model.device)

# output = model.generate(**input_ids, cache_implementation="static")
# print(tokenizer.decode(output[0], skip_special_tokens=True))
