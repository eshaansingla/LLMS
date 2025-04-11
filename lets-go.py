import pandas as pd
import matplotlib.pyplot as plt
import torch
with open("text.txt",'r',encoding="utf-8") as file:
	data=file.read()
chars = sorted(list(set(data)))
print(len(chars))
chars_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_chars = { i:ch for i, ch in enumerate(chars) }
encode= lambda x: ([chars_to_int[c] for c in x])
decode= lambda x:''.join([int_to_chars[c] for c in x])
print(encode('LLMS INTRODUCTION'))
print(decode(encode('LLMS INTRODUCTION')))

encoded_data = torch.tensor(encode(data), dtype=torch.long)
n = int(0.8 * len(encoded_data))
train_data = encoded_data[:n]
val_data = encoded_data[n:]
block_size = 8
for i in range(block_size):
    x = train_data[i:i+block_size]
    y = train_data[i+1:i+block_size+1]
    print(f'Context: {x.tolist()} -> Target: {y.tolist()}')
    print(f'Decoded Context: "{decode(x.tolist())}" -> Target: "{decode(y.tolist())}"')
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
