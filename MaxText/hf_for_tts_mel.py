from dataclasses import dataclass
import queue
import threading
from typing import List
import datasets
import grain.python as grain
import multihost_dataloading
from input_pipeline import _input_pipeline_utils
from jax.experimental import mesh_utils,multihost_utils
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
from collections import Counter
import os
from array_record.python.array_record_module import ArrayRecordWriter
from jax.experimental.compilation_cache import compilation_cache as cc
import subprocess
import shlex
import atexit
cc.set_cache_dir("/tmp/jax_cache")
DEVICE = "tpu"
MAX_LENGTH_AUDIO_44K = 30 * 44100
MAX_LENGTH_AUDIO_16K = 30 * 16000
MAX_LENGTH_TEXT = 10000
PER_DEVICE_BATCH_SIZE = 16
SOURCE_SAMPLERATE = 44100
@dataclass
class Output:
    tokens: np.ndarray
    mel: np.ndarray
    f0: np.ndarray
    length: int
    speaker_id: int

def mount_gcs_bucket(bucket_name, mount_point):
    """
    挂载 GCS 存储桶到指定挂载点。

    Args:
        bucket_name: 要挂载的 GCS 存储桶的名称（不包括 gs:// 前缀）。
        mount_point: 本地挂载点路径。
    """

    try:
        # 1. 创建挂载点目录（如果不存在）
        os.makedirs(mount_point, exist_ok=True)
        print(f"已创建挂载点目录：{mount_point}")

        # 2. 构建 gcsfuse 命令
        gcsfuse_cmd = f"gcsfuse {bucket_name} {mount_point}"

        # 添加 --implicit-dirs 选项以支持隐式目录
        #gcsfuse_cmd += " --implicit-dirs"

        # 建议添加 allow_other 选项，允许其他用户访问挂载点(如果需要)
        # gcsfuse_cmd += " --allow-other"

        # 使用 shlex.split 处理命令字符串，以正确处理空格和引号
        args = shlex.split(gcsfuse_cmd)

        # 3. 执行 gcsfuse 命令
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"gcsfuse 挂载失败：{stderr.decode()}")

        print(f"成功挂载存储桶 {bucket_name} 到 {mount_point}")

    except OSError as e:
        print(f"创建目录失败：{e}")
        return False
    except RuntimeError as e:
        print(e)
        return False
    except FileNotFoundError:
        print("gcsfuse 命令未找到。请确保已安装 gcsfuse。")
        return False

    return True

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
    audio_16k = librosa.resample(features["audio"]["array"], orig_sr=SOURCE_SAMPLERATE, target_sr=16000)
    return {
        "audio_44k": np.asarray(audio_44k, dtype=np.float32),
        "audio_16k": np.asarray(audio_16k, dtype=np.float32),
        "text": np.asarray(features["text"], dtype=np.int32),
        "speaker_id": np.asarray(features["speaker"], dtype=np.int32),
    }   

class PadToMaxLength(grain.MapTransform):

  def map(self, data):
    audio_length = data["audio_44k"].shape[0]
    padded_audio_44k = np.pad(data["audio_44k"],(0,MAX_LENGTH_AUDIO_44K - data["audio_44k"].shape[0]))
    padded_audio_16k = np.pad(data["audio_16k"],(0,MAX_LENGTH_AUDIO_16K - data["audio_16k"].shape[0]))
    text_length = data["text"].shape[0]
    padded_text = np.pad(data["text"],(0,MAX_LENGTH_TEXT - data["text"].shape[0]))
    return {
        "audio_44k": padded_audio_44k,
        "audio_16k": padded_audio_16k,
        "audio_length":audio_length,
        "text": padded_text,
        "text_length":text_length,
        "speaker_id": data["speaker_id"],
    }

if __name__ == "__main__":
    bucket_name = "fbs-us2"  # 替换为你的存储桶名称
    home_dir = os.path.expanduser("~")
    mount_point = os.path.join(home_dir, "bucket")

    if mount_gcs_bucket(bucket_name, mount_point):
        print("挂载完成。")
    else:
        print("挂载过程中发生错误。")
    #if DEVICE == "tpu":
    jax.distributed.initialize()
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=("data")) 
    ds1 = datasets.load_dataset(
        "MikhailT/hifi-tts",
        name="all",
        split="train.clean",
        streaming=True,
    )
    ds2 = datasets.load_dataset(
        "MikhailT/hifi-tts",
        name="all",
        split="train.other",
        streaming=True,
    )
    dataset = datasets.concatenate_datasets([ds1,ds2])
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


    def process(example):
        ids = enc.encode(text=example["text_normalized"])
        
        return {'input_ids': ids}
    dataset = dataset.map(process)
    
    def get_sharding_for_spec(pspec: PartitionSpec) -> NamedSharding:
        """
        Get a NamedSharding for a given PartitionSpec, and the device mesh.
        A NamedSharding is simply a combination of a PartitionSpec and a Mesh instance.
        """
        return NamedSharding(mesh, pspec)

    dataset = dataset.select_columns(["input_ids","audio","speaker"]).rename_column("input_ids", "text")
    dataset = _input_pipeline_utils.HFDataSource(dataset,
                                                jax.process_index(),
                                                jax.process_count(),
                                                1,
                                                False,
                                                15000,
                                                "text")
    operations = []
    operations.append(HFParseAudioFeatures())
    operations.append(PadToMaxLength())
    operations.append(grain.Batch(batch_size=PER_DEVICE_BATCH_SIZE * jax.device_count() // jax.process_count(), drop_remainder=True))
    dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=jax.process_index(), shard_count=jax.process_count(), drop_remainder=True
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
    

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, mesh)
    fcpe_model,fcpe_params = jax_fcpe.load_model()
    fcpe_params = FrozenDict(fcpe_params)
    
    MEL_PAD_TOKEN_ID = 0
    iter_count = 0
    writer = ArrayRecordWriter(os.path.join(mount_point,f"dataset2/hifi_tts_train-shared-{jax.process_index()}.arrayrecord"), 'group_size:1')
    #writer = None
    def close_writer():
        global writer
        global q
        if writer:
            q.put(None)
            q.join()
            writer.close()

    def writer_thread(q, writer):
        while True:
            try:
                data = q.get(timeout=1)  # 设置超时，避免无限阻塞
                if data is None:  # 哨兵值，用于结束线程
                    break
                writer.write(data)
                q.task_done()  # 标记任务完成
            except queue.Empty:
                continue

    q = queue.Queue()

    # 创建并启动写入线程
    t = threading.Thread(target=writer_thread, args=(q, writer))
    t.daemon = True  # 设置为守护线程，主线程退出时自动退出
    t.start()
    atexit.register(close_writer)

    mel_x_sharding = get_sharding_for_spec(PartitionSpec("data"))
    x_sharding = get_sharding_for_spec(PartitionSpec("data"))
    out_sharding = get_sharding_for_spec(PartitionSpec(None))
    
    os.makedirs(os.path.join(mount_point,"dataset2"),exist_ok=True)
    batch_count = 0 
    for item in multihost_gen:
        batch_count += 1
        print(f"batch {batch_count} round {iter_count}",flush=True)
        # if iter_count%10240 == 0:
        #     print(f"round {iter_count}",flush=True)
        #     num = iter_count//10240
        #     if writer is not None:
        #         writer.close() 
        #     writer = ArrayRecordWriter(os.path.join(mount_point,f"dataset2/hifi_tts_train_part_{num}-shared-{jax.process_index()}.arrayrecord"), 'group_size:1')
            
        mel_arr  = jax.jit(get_mel, in_shardings=mel_x_sharding,out_shardings=out_sharding)(item["audio_44k"])
        @partial(jax.jit,in_shardings=(get_sharding_for_spec(PartitionSpec(None)),x_sharding),out_shardings=out_sharding)
        def f0_process_wrap(params,audio):
            f0_arr = jax_fcpe.get_f0(audio,sr=16000,model=fcpe_model,params=params)
            f0_arr = jax.image.resize(f0_arr,shape=(f0_arr.shape[0],mel_arr.shape[-1],1),method="nearest")
            return f0_arr
        # f0_arr = jax.jit(partial(jax_fcpe.get_f0,sr=16000,model=fcpe_model,params=fcpe_params), in_shardings=x_sharding,out_shardings=out_sharding)(item["audio_16k"])
        # f0_arr = jax.image.resize(f0_arr,shape=(f0_arr.shape[0],mel_arr.shape[-1],1),method="nearest")
        f0_arr = f0_process_wrap(fcpe_params,item["audio_16k"])
        text_arr = jax.device_put(item["text"],out_sharding)

        slice_size = PER_DEVICE_BATCH_SIZE * jax.device_count() // jax.process_count()

        mel_arr = np.asarray(mel_arr)
        text_arr = np.asarray(text_arr)
        f0_arr = np.asarray(f0_arr)
        for k in range(slice_size*jax.process_index(),slice_size*(jax.process_index()+1)):
            n_frames = item["audio_length"][k]//512
            text_length = int(item["text_length"][k])
            text_tokens = text_arr[k][:text_length]
            speaker_id = item["speaker_id"][k]
            
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
            codes = np.pad(mel_slice,((0,0),(prompt_length,1)))
            # codes = [[MEL_PAD_TOKEN_ID] * prompt_length for _ in range(mel_dim)]
            # for book_idx, book in zip(range(mel_dim), mel_slice):
            #     for j in book:
            #         codes[book_idx].append(j)
            # for book in codes:
            #     book.extend([MEL_PAD_TOKEN_ID] * 1)
            tokens = np.asarray(tokens)
            codes = np.asarray(codes)
            mel = codes[:-1]
            f0 = codes[-1]
            f0 = f0_to_coarse_numpy(f0)
            iter_count+=1

            example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'tokens': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tokens).numpy()])),
                            'mel': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(mel).numpy()])),
                            'f0':tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(f0).numpy()])),
                            'speaker_id':tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(speaker_id).numpy()])),
                        }
                    )
                )
            q.put(example.SerializeToString())
            #writer.write(example.SerializeToString())
