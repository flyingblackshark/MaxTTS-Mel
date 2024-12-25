from flax import linen as nn
import common_types
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name

from layers import normalizations
from layers import attentions
from layers import initializers
from layers import embeddings
from layers import linears
from layers import quantizations
import jax
from typing import Optional

Embed = embeddings.Embed
RMSNorm = normalizations.RMSNorm
NdInitializer = initializers.NdInitializer
Attention = attentions.Attention
MlpBlock = linears.MlpBlock
Config = common_types.Config
AxisNames = common_types.AxisNames
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn
DType = common_types.DType
Array = common_types.Array
BATCH = common_types.BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
D_KV = common_types.D_KV


nd_dense_init = initializers.nd_dense_init
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant

class LatentSamplingModule(nn.Module):
    config: Config
    mesh: Mesh
    quant: Optional[Quant] = None

    @nn.compact
    def __call__(self, inputs, deterministic: bool = False):
        cfg = self.config
        mesh = self.mesh
        inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
        inputs = checkpoint_name(inputs, "decoder_layer_input")
        # 线性层
        mu = linears.DenseGeneral(
          cfg.mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("embed", "num_activations", "mlp"),
          name="mu_dense",
          quant=self.quant,
          matmul_precision=cfg.matmul_precision,
        )(inputs)
        #mu = nn.Dense(self.latent_dim)(inputs)
        log_sigma = linears.DenseGeneral(
          cfg.mlp_dim,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          kernel_axes=("embed", "num_activations", "mlp"),
          name="log_sigma_dense",
          quant=self.quant,
          matmul_precision=cfg.matmul_precision,
        )(inputs)
        #log_sigma = nn.Dense(self.latent_dim)(inputs) # 输出对数标准差，提高数值稳定性
        sigma = jnp.exp(log_sigma)

        if deterministic:  # 推理模式，直接使用均值
            z = mu
        else:  # 训练模式，进行采样
            key = self.make_rng("sample") #通常在实际应用中，key需要从外部传入，并根据需要进行分割
            epsilon = jax.random.normal(key, (inputs.shape[0],inputs.shape[1], cfg.mlp_dim)) # inputs.shape[0]是batch size
            z = mu + sigma * epsilon

        # 多层感知机 (示例：两层 MLP)
        y = MlpBlock(
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.dropout_rate,
            dtype=cfg.dtype,
            weight_dtype=cfg.weight_dtype,
            name="mlp_local",
            config=cfg,
            quant=self.quant,
        )(z, deterministic=deterministic)

        return y, mu, sigma