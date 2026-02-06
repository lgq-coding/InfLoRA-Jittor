import jittor as jt
import numpy as np

# 检查 Jittor 版本
print("Jittor version:", jt.__version__)

# 检查 CUDA 是否可用
print("CUDA available:", jt.has_cuda)


# 设置 GPU 设备
if jt.has_cuda:
    jt.flags.use_cuda = 1
    print("Using GPU:")
else:
    print("Using CPU")

# 简单的张量操作测试
a = jt.array([1, 2, 3])
b = jt.array([4, 5, 6])
c = a + b
print("Tensor addition:", c.data)

# 矩阵乘法测试
x = jt.randn(3, 4)
y = jt.randn(4, 3)
z = jt.matmul(x, y)
print("Matrix shape:", z.shape)

print("✅ Jittor installation successful!")