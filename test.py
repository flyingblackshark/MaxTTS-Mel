import jax
import jax.numpy as jnp

jax.distributed.initialize()
local_values = jax.process_index() * 111

# 使用 all_reduce 计算全局最大值
global_max = jax.lax.all_gather(local_values, jax.lax.max)

print(global_max)