import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()

        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.heads = heads

        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, 512)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, 512)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (batch_size, max_len, 512) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9)  # (batch_size, h, max_len, max_len)
        weights = F.softmax(scores, dim=-1)  # (batch_size, h, max_len, max_len)
        weights = self.dropout(weights)
        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)
        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, h * d_k)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        # (batch_size, max_len, h * d_k)
        interacted = self.concat(context)
        return interacted


torch.manual_seed(0)

# 定义参数
d_model = 2048
heads = 8
batch_size = 8
max_len = 64



# 在CPU上运行
device = torch.device("cpu")
model = MultiHeadAttention(heads, d_model).to(device)
query = torch.randn(batch_size, max_len, d_model).to(device)
key = torch.randn(batch_size, max_len, d_model).to(device)
value = torch.randn(batch_size, max_len, d_model).to(device)
mask = torch.ones(batch_size, 1, 1, max_len).to(device)
start = time.time()
for i in range(100):
    output = model(query, key, value, mask)
end = time.time()
print("CPU time: {:.3f} s".format((end - start)/100))

# 在GPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadAttention(heads, d_model).to(device)
query = torch.randn(batch_size, max_len, d_model).to(device)
key = torch.randn(batch_size, max_len, d_model).to(device)
value = torch.randn(batch_size, max_len, d_model).to(device)
mask = torch.ones(batch_size, 1, 1, max_len).to(device)
start = time.time()
for i in range(100):
    output = model(query, key, value, mask)
end = time.time()
print("GPU time: {:.5f} s".format((end - start)/100))