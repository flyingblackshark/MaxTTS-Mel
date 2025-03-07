"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Operations used by Grain"""

import dataclasses
import warnings
from typing import Dict
from threading import current_thread
import datasets
from datasets.distributed import split_dataset_by_node
import grain.python as grain
import numpy as np
import tensorflow as tf
import max_logging
import tokenizer
import tiktoken
import librosa
import scipy.signal
from librosa.filters import mel as librosa_mel_fn

Features = Dict[str, tf.Tensor]
AUTOTUNE = tf.data.experimental.AUTOTUNE

########## Functions used by TFDS pipeline


def normalize_features(x, column_name):
  return {"inputs": x[column_name], "targets": x[column_name]}


def get_tokenizer(tokenizer_path, add_bos, add_eos):
  # Load tokenizer
  tokenizer_model = tokenizer.build_tokenizer(tokenizer_path, add_bos, add_eos)
  return tokenizer_model


def truncate_to_max_allowable_length(x, max_length):
  return {k: v[:max_length] for k, v in x.items()}


def shift_data_by_truncation(x):
  x["inputs"] = x["inputs"][:-1]
  x["targets"] = x["targets"][1:]
  return x


def add_segmentation_and_position(x, data_columns):
  for data_column in data_columns:
    x[f"{data_column}_segmentation"] = tf.cast(x[data_column] != 0, tf.int32)
    x[f"{data_column}_position"] = tf.broadcast_to(
        tf.range(x[data_column].shape[-1], dtype=np.int32)[None, :], x[data_column].shape
    )
  return x


########## Functions used by HF pipeline


def tokenization(example, hf_tokenizer, max_length, column_names):
  """Tokenize a HuggingFace dataset"""
  return {
      column_name: hf_tokenizer(example[column_name], truncation=True, max_length=max_length)["input_ids"]
      for column_name in column_names
  }

def get_mel(y, keyshift=0, speed=1, center=False):
    def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
      return np.log(np.clip(x,a_min=clip_val,a_max=None) * C)
    sampling_rate = 44100
    n_mels     = 128 #self.n_mels
    n_fft      = 2048 #self.n_fft
    win_size   = 2048 #self.win_size
    hop_length = 512 #self.hop_length
    fmin       = 40 #self.fmin
    fmax       = 16000 #self.fmax
    clip_val   = 1e-5 #self.clip_val
    
    factor = 2 ** (keyshift / 12)       
    n_fft_new = int(np.round(n_fft * factor))
    win_size_new = int(np.round(win_size * factor))
    hop_length_new = int(np.round(hop_length * speed))
    
    mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    
    pad_left = (win_size_new - hop_length_new) //2
    pad_right = max((win_size_new - hop_length_new + 1) //2, win_size_new - y.shape[-1] - pad_left)
    y = np.pad(y, ((0,0),(pad_left, pad_right)))
    _,_,spec = scipy.signal.stft(y,nfft=n_fft_new,noverlap=win_size_new-hop_length_new,nperseg=win_size_new)
    spectrum_win = np.sin(np.linspace(0, np.pi, win_size_new, endpoint=False)) ** 2
    spec *= spectrum_win.sum()
    spec = np.sqrt(spec.real**2 + spec.imag**2 + (1e-9))

    if keyshift != 0:
        size = n_fft // 2 + 1
        resize = spec.size(1)
        if resize < size:
            spec = np.pad(spec, ((0, 0),(0, size-resize)))
        spec = spec[:, :size, :] * win_size / win_size_new   
    #spec = spec.transpose(0,2,1)
    spec = np.matmul(mel_basis, spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec

@dataclasses.dataclass
class HFNormalizeFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""

  def __init__(self):
    cl100k_base = tiktoken.get_encoding("cl100k_base")

    enc = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            "<|text_start|>": 100264,
            "<|text_end|>": 100265,
            "<|speech_start|>": 100266,
            "<|speech_end|>": 100267,
            "<|semantic|>": 100268,
        }
    )
    self.tokenizezr = enc
    string_prefix = "<|text_start|>user\n"
    string_suffix = "<|text_end|><|speech_start|>assistant\n"

    self.encoded_prefix = enc.encode(
        string_prefix,
        allowed_special={"<|text_start|>","<|text_end|>","<|speech_start|>","<|speech_end|>"}
    )
    self.encoded_suffix = enc.encode(
        string_suffix,
        allowed_special={"<|text_start|>","<|text_end|>","<|speech_start|>","<|speech_end|>"}
    )
    self.semantic_token_id = enc.encode_single_token("<|semantic|>")
    #self.text_end_token_id  = enc.encode_single_token("<|text_end|>")
    self.speech_end_token_id  = enc.encode_single_token("<|speech_end|>")

  def map(self, features):
    text_tokens = self.tokenizezr.encode(text=features["text_normalized"])
    encoded = self.encoded_prefix + np.asarray(text_tokens).tolist() + self.encoded_suffix
    #audio_44k = librosa.resample(features["audio"]["array"], orig_sr=features["audio"]["sampling_rate"], target_sr=44100)
    mel = get_mel(features["audio"]["array"][...,np.newaxis])[0].transpose(1,0)
    mel_length = mel.shape[0]
    tokens = (
                encoded
                + [self.semantic_token_id] * mel_length
                + [self.speech_end_token_id]
            )
    tokens = np.asarray(tokens)
    prompt_length = len(encoded)
    mel = np.pad(mel,((prompt_length,1),(0,0)))
    inputs_mel = mel[:-1]
    targets_mel = mel[1:]
    return {
        "inputs": tokens[:-1],
        "targets": tokens[1:],
        "inputs_mel":inputs_mel,
        "targets_mel":targets_mel,
    }


class HFDataSource(grain.RandomAccessDataSource):
  """A class that makes HuggingFace IterableDataset a grain datasource without random access support"""

  def __init__(
      self,
      dataset: datasets.IterableDataset,
      dataloading_host_index: int,
      dataloading_host_count: int,
      num_threads: int,
      generate_padding_example: bool,
      max_target_length: int,
      data_column_names: list[str],
      source_sampling_rate = 16000,
  ):
    self.dataset = dataset
    self.num_threads = num_threads
    self.dataloading_host_count = dataloading_host_count
    self.dataloading_host_index = dataloading_host_index
    #self.generate_padding_example = generate_padding_example
    self.max_target_lenth = max_target_length
    self.data_column_names = data_column_names
    self.n_shards = dataset.n_shards
    self._check_shard_count()
    self.dataset_shards = [dataloading_host_index * self.num_threads + i for i in range(self.num_threads)]
    self.datasets = [split_dataset_by_node(dataset, world_size=self.n_shards, rank=x) for x in self.dataset_shards]
    self.data_iters = []
    self.out_of_data = False
    self.source_sampling_rate = source_sampling_rate

  def _check_shard_count(self):
    if self.n_shards < (self.dataloading_host_count * self.num_threads):
      warnings.warn(
          f"WARNING: Inefficient dataloading. Your train or eval dataset contains {self.n_shards} shards, "
          "smaller than number of host loading data. This is known to lead to inefficient dataloading. "
          "see https://github.com/google/maxtext/blob/main/getting_started/Data_Input_Pipeline.md#multihost-dataloading-best-practice"
      )
      self.n_shards = self.dataloading_host_count * self.num_threads

  def _update_shard(self, idx):
    new_shard = self.dataset_shards[idx] + self.dataloading_host_count * self.num_threads
    if new_shard < self.n_shards:
      max_logging.log(f"Updating host {self.dataloading_host_index} dataset {idx}, was on shard {self.dataset_shards[idx]}")
      max_logging.log(f"New shard is {new_shard}")
      self.dataset_shards[idx] = new_shard
      self.datasets[idx] = split_dataset_by_node(self.dataset, world_size=self.n_shards, rank=self.dataset_shards[idx])
      self.data_iters[idx] = iter(self.datasets[idx])
    else:
      max_logging.log(
          f"Run out of shards on host {self.dataloading_host_index}, shard {self.dataset_shards[idx]} is not available"
      )
      self.out_of_data = True
      # if self.generate_padding_example:
      #   max_logging.log(
      #       f"Host {self.dataloading_host_index} will start generating all-0 padding examples until step number is met."
      #   )

  def __len__(self):
    """Return length of the HF dataset. Since HuggingFace IterableDataset does not have length,
    a fake length bigger than the dataset is returned"""
    return 10_000_000_000

  def __getitem__(self, index):
    """Since HuggingFace IterableDataset does not support random access by index.
    The next item in the iterator is returned."""
    if not self.data_iters:
      self.data_iters = [iter(x) for x in self.datasets]
    idx = int(current_thread().name.split("_")[1])

    while True:
      try:
        if self.out_of_data:
              return {
                  "audio":{
                    "array": np.zeros(30 * self.source_sampling_rate, dtype=np.float32),
                  },
                  "text": np.zeros(self.max_target_lenth, dtype=np.int32),
                  "speaker": -1,
              }
          # if self.generate_padding_example:
          #   return {column_name: np.zeros(self.max_target_lenth, dtype=np.int32) for column_name in self.data_column_names}
          # else:
          #   return None
        data = next(self.data_iters[idx])
        return data
      except StopIteration:
        self._update_shard(idx)


########## Functions used by Grain pipeline


@dataclasses.dataclass
class ParseFeatures(grain.MapTransform):
  """Parse serialized example"""

  # def __init__(self, data_columns, tokenize):
  #   self.data_columns = data_columns
  #   if tokenize:
  #     self.dtype = tf.string
  #   else:
  #     self.dtype = tf.int64

  def map(self, features):
    parsed = tf.io.parse_example(features, {
      "tokens": tf.io.FixedLenFeature([], dtype=tf.string),
      "mel": tf.io.FixedLenFeature([], dtype=tf.string),
      "f0": tf.io.FixedLenFeature([], dtype=tf.string),
      #"prompt_length": tf.io.FixedLenFeature([], dtype=tf.int64)
      })
    tokens = tf.io.parse_tensor(parsed["tokens"],tf.int64).numpy()
    inputs = tokens[:-1]
    targets = tokens[1:]
    mel =  tf.io.parse_tensor(parsed["mel"],tf.float32).numpy().transpose(1,0)
    inputs_mel = mel[:-1]
    targets_mel = mel[:-1]
    f0 =  tf.io.parse_tensor(parsed["f0"],tf.int64).numpy()
    inputs_f0 = f0[:-1]
    targets_f0 = f0[:-1]
    #prompt_length = parsed["prompt_length"].numpy()

    return {
        "inputs": inputs,
        "targets": targets,
        "inputs_mel":inputs_mel,
        "targets_mel":targets_mel,
        "inputs_f0":inputs_f0,
        "targets_f0":targets_f0,
        #"prompt_length":prompt_length,
    }


@dataclasses.dataclass
class InputsTargetsFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_name):
    self.column_name = column_name

  def map(self, features):
    return {"inputs": features[self.column_name], "targets": features[self.column_name]}


@dataclasses.dataclass
class NormalizeFeatures(grain.MapTransform):
  """Normalize text feature keys."""

  def __init__(self, column_names, tokenize):
    self.column_names = column_names
    self.tokenize = tokenize

  def map(self, features):
    if self.tokenize:
      return {col: features[col].numpy()[0].decode() for col in self.column_names}
    else:
      return {col: features[col].numpy() for col in self.column_names}


@dataclasses.dataclass
class ReformatPacking(grain.MapTransform):
  """Reformat packing outputs."""

  def __init__(self, column_names):
    self.column_names = column_names

  def map(self, data):
    ret = {}
    for col in self.column_names:
      ret[f"{col}"] = data[0][col]
      ret[f"{col}_segmentation"] = data[1][col]
      ret[f"{col}_position"] = data[2][col]
    return ret


@dataclasses.dataclass
class PadToMaxLength(grain.MapTransform):
  """Pads each input to the specified length"""

  def __init__(self, max_length):
    self.max_length = max_length

  def map(self, data: dict[str, np.ndarray]):
    """map to each element"""

    def _pad(x, max_length):
      pad_amount = max(max_length - x.shape[0], 0)
      pad_amount = [(0, pad_amount)] + [(0, 0)] * (len(x.shape) - 1)
      return np.pad(x, pad_amount)
    data_columns = ("inputs","targets")
    #data_columns = list(data.keys())
    for data_column in data_columns:
      data[f"{data_column}_segmentation"] = (data[data_column] != 0).astype(np.int32)
      data[f"{data_column}_position"] = np.arange(data[data_column].shape[0], dtype=np.int32)
    for key, _ in data.items():
      data[key] = _pad(data[key], self.max_length)
    return data


def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [
      slice(None),
  ] * len(x.shape)
  slices[axis] = slice(0, -1)
  padded = np.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
  return padded[tuple(slices)]


def shift_and_refine(x, axis=1):
  """Shift inputs, set segmentation to 0 when target element is 0.
  Replace EOS by 0 for packed inputs."""
  x["inputs"] = shift_right(x["inputs"], axis=axis)
  targets_nonzero = x["targets"] != 0
  x["inputs_segmentation"] *= targets_nonzero
  x["targets_segmentation"] *= targets_nonzero
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  x["inputs"] *= x["inputs_segmentation"] == shift_right(x["inputs_segmentation"], axis=axis)

  return x


@dataclasses.dataclass
class ShiftData(grain.MapTransform):
  """Shift inputs and refine annotations."""

  def __init__(self, axis=1):
    self.axis = axis

  def map(self, data):
    return shift_and_refine(data, axis=self.axis)
