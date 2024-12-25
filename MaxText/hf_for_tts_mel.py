import datasets
import transformers
import grain.python as grain
from input_pipeline import _input_pipeline_utils
import multihost_dataloading
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from functools import partial
import librosa
import jax
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tiktoken
from librosa.filters import mel as librosa_mel_fn
import audax.core.stft
import jax_fcpe
from flax.core import FrozenDict, copy
from array_record.python.array_record_module import ArrayRecordWriter
DEVICE = "tpu"
MAX_LENGTH_AUDIO = 30 * 44100
MAX_LENGTH_TEXT = 10000
GLOBAL_BATCH_SIZE = 64
SOURCE_SAMPLERATE = 44100
def f0_to_coarse_numpy(f0, f0_bin=128, f0_mel_min=80.0, f0_mel_max=880.0):
    """
    将 f0 值转换为粗略表示的 NumPy 版本。

    Args:
        f0: 输入的 f0 值，NumPy 数组。
        f0_bin: f0 区间的数量。
        f0_mel_min: mel 刻度上的最小值。
        f0_mel_max: mel 刻度上的最大值。

    Returns:
        粗略的 f0 表示，NumPy 数组。
    """
    f0_mel = 1127 * np.log1p(f0 / 700)  # np.log1p(x) 等价于 np.log(1 + x)

    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.

    f0_mel = np.where(f0_mel > 0, f0_mel * a - b, f0_mel)

    f0_coarse = np.round(f0_mel).astype(np.int64)

    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))

    return f0_coarse
def dynamic_range_compression_jax(x, C=1, clip_val=1e-5):
    return jnp.log(jnp.clip(x,min=clip_val) * C)
def get_mel(y, keyshift=0, speed=1, center=False):
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
    hann_window= jnp.hanning(win_size_new)
    
    pad_left = (win_size_new - hop_length_new) //2
    pad_right = max((win_size_new - hop_length_new + 1) //2, win_size_new - y.shape[-1] - pad_left)
    # if pad_right < y.size(-1):
    #     mode = 'reflect'
    # else:
    #     mode = 'constant'
    y = jnp.pad(y, ((0,0),(pad_left, pad_right)))
    spec = audax.core.stft.stft(y,n_fft_new,hop_length_new,win_size_new,hann_window,onesided=True,center=False)
    spec = jnp.sqrt(spec.real**2 + spec.imag**2 + (1e-9))

    if keyshift != 0:
        size = n_fft // 2 + 1
        resize = spec.size(1)
        if resize < size:
            spec = jnp.pad(spec, ((0, 0),(0, size-resize)))
        spec = spec[:, :size, :] * win_size / win_size_new   
    spec = spec.transpose(0,2,1)
    spec = jnp.matmul(mel_basis, spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec

class HFParseAudioFeatures(grain.MapTransform):
  """Normalize feature keys for HuggingFace input"""
  def map(self, features):
    audio_44k = librosa.resample(features["audio"]["array"], orig_sr=SOURCE_SAMPLERATE, target_sr=44100)
    return {
        "audio": np.asarray(audio_44k, dtype=np.float32),
        "text": np.asarray(features["text"], dtype=np.int32),
    }   

class PadToMaxLength(grain.MapTransform):

  def map(self, data):
    audio_length = data["audio"].shape[0]
    padded_audio = np.pad(data["audio"],(0,MAX_LENGTH_AUDIO - data["audio"].shape[0]))
    text_length = data["text"].shape[0]
    padded_text = np.pad(data["text"],(0,MAX_LENGTH_TEXT - data["text"].shape[0]))
    return {
        "audio": padded_audio,
        "audio_length":audio_length,
        "text": padded_text,
        "text_length":text_length
    }
if __name__ == "__main__":
    if DEVICE == "tpu":
        jax.distributed.initialize()
        device_mesh = mesh_utils.create_device_mesh((4, 1))
    else:
        device_mesh = mesh_utils.create_device_mesh((1, 1))
    mesh = Mesh(device_mesh, axis_names=("data", "model")) 
    dataset = datasets.load_dataset(
        "MikhailT/hifi-tts",
        name="clean",
        split="train",
        streaming=True,
    )
    cl100k_base = tiktoken.get_encoding("cl100k_base")

    # In production, load the arguments directly instead of accessing private attributes
    # See openai_public.py for examples of arguments for specific encodings
    enc = tiktoken.Encoding(
        # If you're changing the set of special tokens, make sure to use a different name
        # It should be clear from the name what behaviour to expect.
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


    def process(example):
        # 使用 tiktoken 分词
        ids = enc.encode(text=example["text_normalized"])
        
        return {'input_ids': ids}
        #return ids
    dataset = dataset.map(process)
    
    def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
        """
        Get a NamedSharding for a given PartitionSpec, and the device mesh.
        A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
        """
        return NamedSharding(mesh, pspec)

    dataset = dataset.select_columns(["input_ids","audio"]).rename_column("input_ids", "text")
    #dataset = dataset.to_iterable_dataset()
    dataset = _input_pipeline_utils.HFDataSource(dataset,
                                                0,
                                                1,
                                                1,
                                                False,
                                                15000,
                                                "text")
    operations = []
    operations.append(HFParseAudioFeatures())
    operations.append(PadToMaxLength())
    operations.append(grain.Batch(batch_size=GLOBAL_BATCH_SIZE, drop_remainder=True))
    dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=0, shard_count=1, drop_remainder=True
      ),
      shuffle=False,
      seed=0,
    )

    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=operations,
        sampler=dummy_index_sampler,
        worker_count=1,  # only supports one worker for now, more workers results in duplicated data
        worker_buffer_size=1,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=128),
    )
    

    #multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)
    fcpe_model,fcpe_params = jax_fcpe.load_model()
    fcpe_params = FrozenDict(fcpe_params)
    
    MEL_PAD_TOKEN_ID = 0
    i = 0
    writer = None
    x_sharding = get_sharding_for_spec(PartitionSpec("data"))
    for item in dataloader:
        print(f"round {i}")
        if i%10240 == 0:
            num = i//10240
            if writer is not None:
                writer.close() 
            writer = ArrayRecordWriter(f"/dev/shm/dataset/hifi_tts_train_part_{num}.arrayrecord", 'group_size:1')
            
        mel_arr  = jax.jit(get_mel, in_shardings=x_sharding,out_shardings=x_sharding)(item["audio"])
        audio_16k = librosa.resample(item["audio"], orig_sr=SOURCE_SAMPLERATE, target_sr=16000)
        f0_arr = jax.jit(partial(jax_fcpe.get_f0,sr=16000,model=fcpe_model,params=fcpe_params), in_shardings=x_sharding,out_shardings=x_sharding)(audio_16k)
        f0_arr = jax.image.resize(f0_arr,shape=(f0_arr.shape[0],mel_arr.shape[-1],1),method="nearest")
        mel_arr = np.asarray(mel_arr)
        f0_arr = np.asarray(f0_arr)
        for k in range(GLOBAL_BATCH_SIZE):
            n_frames = item["audio_length"][k]//512
            text_length = item["text_length"][k]
            text_tokens = item["text"][k][:text_length]
            mel_slice = mel_arr[k,:,:n_frames]
            f0_slice = f0_arr[k,:n_frames].transpose(1,0)
            mel_slice = np.concatenate((mel_slice,f0_slice),axis=0)


            string_prefix = "<|im_start|>user\n"
            string_suffix = "<|im_end|><|im_start|>assistant\n"

            encoded_prefix = enc.encode(
                string_prefix,
                allowed_special={"<|im_start|>","<|im_end|>"}
            )

            encoded_suffix = enc.encode(
                string_suffix,
                allowed_special={"<|im_start|>","<|im_end|>"}
            )

            encoded = encoded_prefix + np.asarray(text_tokens).tolist() + encoded_suffix
            mel_dim = 129

            mel_token_id = enc.encode_single_token("<|semantic|>")
            mel_length = mel_slice.shape[1]
            tokens = (
                encoded
                + [mel_token_id] * mel_length
                + [enc.encode_single_token("<|im_end|>")]
            )
            prompt_length = len(encoded)

            codes = [[MEL_PAD_TOKEN_ID] * prompt_length for _ in range(mel_dim)]
            for book_idx, book in zip(range(mel_dim), mel_slice):
                for j in book:
                    codes[book_idx].append(j)
            for book in codes:
                book.extend([MEL_PAD_TOKEN_ID] * 1)

            #tokens = [tokens] + codes
            tokens = np.asarray(tokens)
            codes = np.asarray(codes)
            mel = codes[:-1]
            f0 = codes[-1]
            f0 = f0_to_coarse_numpy(f0)
            i+=1
            example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'tokens': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tokens).numpy()])),
                            'mel': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(mel).numpy()])),
                            'f0':tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(f0).numpy()])),
                            # 'prompt_length':tf.train.Feature(
                            #    int64_list=tf.train.Int64List(value=[prompt_length])
                            # )
                        }
                    )
                )
            writer.write(example.SerializeToString())
    writer.close() 