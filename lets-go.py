import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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



