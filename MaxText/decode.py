# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI utility for running inference on a single stream"""

import jax

import max_utils
import maxengine

import os
import pyconfig
import sys
import tiktoken
import jax.numpy as jnp
CODEBOOK_PAD_TOKEN_ID = 0
def encode_tokens(
    tokenizer,
    string,
    prompt_tokens=None,
    #num_codebooks=128,
):
    #string = clean_text(string)
    string = f"<|im_start|>user\n{string}<|im_end|><|im_start|>assistant\n"

    new_tokens = tokenizer.encode(
      string,
      allowed_special={"<|im_start|>","<|im_end|>"}
      )
    tokens = jnp.asarray(new_tokens, dtype=jnp.int32)
    true_length = tokens.shape[0]
    # Codebooks
    # zeros = (
    #     jnp.ones((num_codebooks, tokens.shape[1]), dtype=jnp.int32)
    #     * CODEBOOK_PAD_TOKEN_ID
    # )
    # prompt = jnp.concatenate((tokens, zeros), axis=0)

    if prompt_tokens is None:
        return tokens,true_length

    # Get prompt tokens
    # if prompt_tokens.ndim == 3:
    #     assert (
    #         prompt_tokens.shape[0] == 1
    #     ), f"3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
    #     prompt_tokens = prompt_tokens[0]

    # assert prompt_tokens.ndim == 2
    data = prompt_tokens

    # if prompt_tokens.shape[0] > num_codebooks:
    #     logger.warning(
    #         f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
    #     )
    #     data = data[:num_codebooks]

    # Add pad token for each codebook
    data = jnp.concatenate(
        (data, jnp.zeros((data.shape[0], 1), dtype=jnp.int32)),
        axis=1,
    )

    # Since 1.0, we use <|semantic|>
    s0_token_id = tokenizer.encode_single_token("<|semantic|>")
    end_token_id = tokenizer.encode_single_token("<|im_end|>")
    main_token_ids = (
        jnp.ones((1, data.shape[1]), dtype=jnp.int32) * s0_token_id
    )
    main_token_ids = main_token_ids.at[0, -1].set(end_token_id)

    data = jnp.concatenate((main_token_ids, data), axis=0)
    prompt = jnp.concatenate((prompt, data), axis=1)
    true_length = prompt.shape[1]
    return prompt,true_length

def main(config):
  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  params = engine.load_params(rng_load_params)

  text = config.prompt
  cl100k_base = tiktoken.get_encoding("cl100k_base")
  
  enc = tiktoken.Encoding(
      name="cl100k_im",
      pat_str=cl100k_base._pat_str,
      mergeable_ranks=cl100k_base._mergeable_ranks,
      special_tokens={
          **cl100k_base._special_tokens,
          "<|im_start|>": 100264,
          "<|im_end|>": 100265,
          "<|semantic|>": 100266,
      }
  )

  tokens,true_length = encode_tokens(enc,text)
  # metadata = engine.get_tokenizer()
  # tokenizer_model = engine.build_tokenizer(metadata)
  # tokens, true_length = tokenizer_model.encode(text, is_bos=True, prefill_lengths=[config.max_prefill_predict_length])
  # assert true_length <= config.max_prefill_predict_length, "can't take too many tokens"
  # assert config.quantization != "fp8", "fp8 on NVIDIA GPUs is not supported in decode.py yet"

  # Split RNG before calling prefill
  rng, rng_prefill = jax.random.split(rng)
  prefill_result, first_token = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length, rng=rng_prefill)
  slot = 0

  rng, rng_init_decode = jax.random.split(rng)
  decode_state = engine.init_decode_state(rng_init_decode)
  decode_state = engine.insert(prefill_result, decode_state, slot=slot)

  steps = range(config.max_prefill_predict_length, config.max_target_length)
  sampled_tokens_list = []
  sampled_tokens_list.append(first_token)
  for _ in steps:
    rng, rng_generate = jax.random.split(rng)
    decode_state, sampled_tokens = engine.generate(params, decode_state, rng=rng_generate)
    sampled_tokens_list.append(sampled_tokens)

  results = [sampled_tokens.get_result_at_slot(slot).tokens.item() for sampled_tokens in sampled_tokens_list]
  results_mel = [sampled_tokens.mel_data[0] for sampled_tokens in sampled_tokens_list]
  #results_f0 = [sampled_tokens.f0_data[0] for sampled_tokens in sampled_tokens_list]
  mel_res = jnp.concatenate(results_mel)
  # 转换为分贝单位
  import matplotlib.pyplot as plt
  plt.figure(figsize=(10, 6))
  plt.imshow(mel_res.transpose(1,0), aspect='auto', origin='lower',)
            #extent=[time_bins[0], time_bins[-1], 0, mel_spectrogram.shape[0]])
  plt.colorbar(label='Power (dB)')
  plt.title("Mel-Spectrogram")
  plt.xlabel("Frames (86 frame = 1 s)")
  plt.ylabel("Mel Bands")
  plt.tight_layout()

  # 保存图片
  plt.savefig("mel_spectrogram.png")
  plt.show()
  print()
  import numpy as np
  end_index = np.where(np.asarray(results) == 100266)


  # output = tokenizer_model.decode(results)
  # print(f"Input `{text}` -> `{output}`")

  # if config.autoregressive_decode_assert != "":
  #   assert (
  #       output == config.autoregressive_decode_assert
  #   ), f"generated text mismatch {output=} {config.autoregressive_decode_assert=}"


def validate_config(config):
  assert config.load_full_state_path == "", (
      "Decode doesn't operate on full states! Convert to parameter checkpoint first." "Using generate_param_only_checkpoint."
  )


if __name__ == "__main__":
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  pyconfig.initialize(sys.argv)
  cfg = pyconfig.config
  validate_config(cfg)
  max_utils.print_system_information()
  main(cfg)
