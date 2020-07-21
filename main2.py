
from collections import deque
import time
import random

a = list(range(10000))
b = deque(a)

start = time.time()

for _ in range(20000):
    random.sample(b, 100)

print(type(random.sample(b, 100)))
print(f'{time.time() - start}')