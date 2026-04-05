"""
Compressed Context Memory (CCM) for Dialogue Summarization.

Implementation of the CCM paper (ICLR 2024):
"Compressed Context Memory for Online Language Model Interaction"
by Jang-Hyun Kim, Junyoung Yeom, Sangdoo Yun, Hyun Oh Song.

This system compresses dialogue context into a single <SUM> token's KV cache
using Conditional LoRA, enabling memory-efficient summarization inference.
"""
