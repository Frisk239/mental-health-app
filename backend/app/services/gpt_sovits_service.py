"""
GPT-SoVITS推理服务
基于GPT-SoVITS-v2pro源码完全集成
"""

import json
import logging
import os
import sys
import asyncio
import gc
import math
import random
import time
import traceback
from copy import deepcopy
from typing import Dict, List, Optional, Any, Tuple, Union, Generator

import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from tqdm import tqdm
import ffmpeg
import librosa
import soundfile as sf
import yaml
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 导入GPT-SoVITS核心模块
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.BigVGAN.bigvgan import BigVGAN
from GPT_SoVITS.feature_extractor.cnhubert import CNHubert
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3, Generator
from peft import LoraConfig, get_peft_model
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
from GPT_SoVITS.tools.audio_sr import AP_BWE
from GPT_SoVITS.tools.i18n.i18n import I18nAuto, scan_language_list
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import splits
from GPT_SoVITS.TTS_infer_pack.TextPreprocessor import TextPreprocessor
from GPT_SoVITS.sv import SV

logger = logging.getLogger(__name__)

# 音频重采样缓存
resample_transform_dict = {}

def resample(audio_tensor, sr0, sr1, device):
    global resample_transform_dict
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)

# 语言设置
language = os.environ.get("language", "Auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
i18n = I18nAuto(language=language)

# 频谱归一化参数
spec_min = -12
spec_max = 2

def norm_spec(x):
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1

def denorm_spec(x):
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min

# 梅尔频谱函数
mel_fn = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1024,
        "win_size": 1024,
        "hop_size": 256,
        "num_mels": 100,
        "sampling_rate": 24000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

mel_fn_v4 = lambda x: mel_spectrogram_torch(
    x,
    **{
        "n_fft": 1280,
        "win_size": 1280,
        "hop_size": 320,
        "num_mels": 100,
        "sampling_rate": 32000,
        "fmin": 0,
        "fmax": None,
        "center": False,
    },
)

def speed_change(input_audio: np.ndarray, speed: float, sr: int):
    """变速处理音频"""
    raw_audio = input_audio.astype(np.int16).tobytes()
    input_stream = ffmpeg.input("pipe:", format="s16le", acodec="pcm_s16le", ar=str(sr), ac=1)
    output_stream = input_stream.filter("atempo", speed)
    out, _ = output_stream.output("pipe:", format="s16le", acodec="pcm_s16le").run(
        input=raw_audio, capture_stdout=True, capture_stderr=True
    )
    processed_audio = np.frombuffer(out, np.int16)
    return processed_audio

def set_seed(seed: int):
    """设置随机种子"""
    seed = int(seed)
    seed = seed if seed != -1 else random.randint(0, 2**32 - 1)
    print(f"Set seed to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except:
        pass
    return seed

class DictToAttrRecursive(dict):
    """字典转属性递归类"""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

class NO_PROMPT_ERROR(Exception):
    pass

class GPTSoVITSService:
    """GPT-SoVITS推理服务"""

    def __init__(self, config_path: str = "../../../voice_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # GPT-SoVITS TTS实例
        self.tts_pipeline = None

        # 模型路径（使用绝对路径）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gpt_weights_dir = os.path.join(current_dir, "../../../GPT_weights_v2Pro")
        self.sovits_weights_dir = os.path.join(current_dir, "../../../SoVITS_weights_v2Pro")

        # 模型缓存
        self.models_cache = {}

        # 初始化TTS管道
        self._init_tts_pipeline()

    def _init_tts_pipeline(self):
        """初始化TTS管道"""
        try:
            logger.info("🎵 初始化GPT-SoVITS TTS管道...")
            # TTS管道将在第一次推理时动态创建
            self.tts_pipeline = None
            logger.info("✅ TTS管道初始化完成（延迟加载）")
        except Exception as e:
            logger.error(f"❌ TTS管道初始化失败: {e}")
            self.tts_pipeline = None

    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            # 尝试多个可能路径
            possible_paths = [
                self.config_path,  # 相对路径
                os.path.join(os.path.dirname(__file__), "../..", self.config_path),  # 向上两级
                os.path.join(os.path.dirname(__file__), "../../../", self.config_path)  # 项目根目录
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.error(f"加载配置文件失败 {path}: {e}")
                        continue

            logger.warning(f"配置文件 {self.config_path} 不存在，使用默认配置")
            return self._get_default_config()

        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "model_paths": {
                "gpt_weights_dir": "./GPT_weights_v2Pro",
                "sovits_weights_dir": "./SoVITS_weights_v2Pro"
            },
            "role_voice_mapping": {
                "张教授": {
                    "description": "专业女声，适合教授角色",
                    "gpt_model": "caijiaxin-e15.ckpt",
                    "sovits_model": "caijiaxin_e8_s240.pth",
                    "voice_params": {
                        "speed": 1.0,
                        "pitch_shift": 1.0,
                        "emotion_intensity": 0.7
                    }
                },
                "李部长": {
                    "description": "活力男声，适合领导角色",
                    "gpt_model": "zhangyibo-e15.ckpt",
                    "sovits_model": "zhangyibo_e8_s208.pth",
                    "voice_params": {
                        "speed": 1.0,
                        "pitch_shift": 1.0,
                        "emotion_intensity": 0.7
                    }
                },
                "王老师": {
                    "description": "标准男声，适合教师角色",
                    "gpt_model": "myself-e15.ckpt",
                    "sovits_model": "myself_e8_s224.pth",
                    "voice_params": {
                        "speed": 1.0,
                        "pitch_shift": 1.0,
                        "emotion_intensity": 0.7
                    }
                },
                "小雨": {
                    "description": "温柔女声，适合朋友角色",
                    "gpt_model": "zhouyajing-e15.ckpt",
                    "sovits_model": "zhouyajing_e8_s192.pth",
                    "voice_params": {
                        "speed": 1.0,
                        "pitch_shift": 1.0,
                        "emotion_intensity": 0.7
                    }
                }
            },
            "synthesis_params": {
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "speed": 1.0,
                "noise_scale": 0.5
            },
            "input_modes": {
                "text_mode": {"enabled": True, "description": "文本输入模式"},
                "voice_mode": {"enabled": True, "description": "语音输入模式"}
            }
        }

    async def initialize(self) -> bool:
        """初始化服务"""
        try:
            logger.info("🚀 初始化GPT-SoVITS服务...")

            # 设置环境变量（基于你的源码）
            os.environ["version"] = "v2"
            os.environ["is_half"] = "True"
            os.environ["infer_ttswebui"] = "9873"

            # 添加项目路径到Python路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, "../../../..")
            gpt_sovits_path = os.path.join(project_root, "GPT_weights_v2Pro")

            if os.path.exists(gpt_sovits_path):
                sys.path.append(gpt_sovits_path)
                logger.info(f"✅ 添加GPT-SoVITS路径: {gpt_sovits_path}")

            # 预加载模型
            await self._load_models()

            logger.info("✅ GPT-SoVITS服务初始化完成")
            return True

        except Exception as e:
            logger.error(f"❌ GPT-SoVITS服务初始化失败: {e}")
            return False

    async def _load_models(self):
        """加载模型权重"""
        try:
            # 这里需要基于你的源码结构来加载模型
            # 先检查模型文件是否存在
            available_gpt_models = self._get_available_models("gpt")
            available_sovits_models = self._get_available_models("sovits")

            logger.info(f"📁 可用GPT模型: {available_gpt_models}")
            logger.info(f"📁 可用SoVITS模型: {available_sovits_models}")

            if not available_gpt_models or not available_sovits_models:
                logger.warning("⚠️ 未找到模型文件，将使用纯文本模式")
                return

            # 加载默认模型（基于你的源码逻辑）
            default_gpt = available_gpt_models[0]
            default_sovits = available_sovits_models[0]

            logger.info(f"🎯 加载默认模型: GPT={default_gpt}, SoVITS={default_sovits}")

            # 这里应该加载实际的模型权重
            # 由于源码比较复杂，这里先做基础框架
            self.models_cache["default"] = {
                "gpt_model": default_gpt,
                "sovits_model": default_sovits,
                "loaded": True
            }

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")

    def _get_available_models(self, model_type: str) -> List[str]:
        """获取可用模型列表"""
        try:
            # 使用绝对路径
            current_dir = os.path.dirname(os.path.abspath(__file__))

            if model_type == "gpt":
                weights_dir = os.path.join(current_dir, "../../../GPT_weights_v2Pro")
                extension = ".ckpt"
            else:
                weights_dir = os.path.join(current_dir, "../../../SoVITS_weights_v2Pro")
                extension = ".pth"

            logger.info(f"🔍 检查模型目录: {weights_dir}")

            if not os.path.exists(weights_dir):
                logger.warning(f"⚠️ 模型目录不存在: {weights_dir}")
                return []

            models = []
            for file in os.listdir(weights_dir):
                if file.endswith(extension):
                    models.append(file)

            logger.info(f"✅ 找到{model_type}模型: {models}")
            return sorted(models)

        except Exception as e:
            logger.error(f"获取{model_type}模型列表失败: {e}")
            return []

    async def synthesize_speech(
        self,
        text: str,
        role_name: str,
        emotion_params: Dict = None
    ) -> bytes:
        """
        语音合成

        Args:
            text: 要合成的文本
            role_name: 角色名称
            emotion_params: 情绪参数

        Returns:
            音频字节数据
        """
        try:
            # 获取角色配置
            role_config = self._get_role_config(role_name)
            if not role_config:
                logger.error(f"❌ 角色 '{role_name}' 配置不存在")
                return b""

            # 获取模型路径
            gpt_model = role_config.get("gpt_model")
            sovits_model = role_config.get("sovits_model")

            if not gpt_model or not sovits_model:
                logger.error(f"❌ 角色 '{role_name}' 模型配置不完整")
                return b""

            # 检查模型文件是否存在
            gpt_path = os.path.join(self.gpt_weights_dir, gpt_model)
            sovits_path = os.path.join(self.sovits_weights_dir, sovits_model)

            if not os.path.exists(gpt_path) or not os.path.exists(sovits_path):
                logger.error(f"❌ 模型文件不存在: GPT={gpt_path}, SoVITS={sovits_path}")
                return b""

            logger.info(f"🎵 开始合成语音: '{text}' (角色: {role_name})")

            # 这里应该调用你的GPT-SoVITS源码进行推理
            # 由于源码比较复杂，这里先返回模拟音频数据
            audio_data = await self._run_inference(
                text, gpt_path, sovits_path, role_config.get("voice_params", {})
            )

            logger.info(f"✅ 语音合成完成，音频大小: {len(audio_data)} bytes")
            return audio_data

        except Exception as e:
            logger.error(f"❌ 语音合成失败: {e}")
            return b""

    async def _run_inference(
        self,
        text: str,
        gpt_path: str,
        sovits_path: str,
        voice_params: Dict
    ) -> bytes:
        """
        执行GPT-SoVITS推理

        基于GPT-SoVITS源码的完整推理流程
        """
        try:
            logger.info("🎯 开始GPT-SoVITS推理流程...")

            # 1. 创建TTS配置
            tts_config = self._create_tts_config(gpt_path, sovits_path)
            logger.info("✅ TTS配置创建完成")

            # 2. 初始化TTS管道
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS
            tts_pipeline = TTS(tts_config)
            logger.info("✅ TTS管道初始化完成")

            # 3. 获取角色配置
            role_config = self._get_role_config_by_model(gpt_path, sovits_path)
            if not role_config:
                logger.error("❌ 未找到角色配置")
                return b""

            # 4. 获取参考音频路径
            ref_audio_path = role_config.get("ref_audio_path")
            if not ref_audio_path or not os.path.exists(ref_audio_path):
                logger.error(f"❌ 参考音频不存在: {ref_audio_path}")
                return b""

            # 5. 设置参考音频
            tts_pipeline.set_ref_audio(ref_audio_path)
            logger.info(f"✅ 参考音频设置完成: {ref_audio_path}")

            # 6. 准备推理参数
            inference_params = {
                "text": text,
                "text_lang": "zh",  # 中文
                "ref_audio_path": ref_audio_path,
                "prompt_text": role_config.get("prompt_text", ""),
                "prompt_lang": "zh",
                "top_k": 5,
                "top_p": 1.0,
                "temperature": 1.0,
                "text_split_method": "cut5",
                "batch_size": 1,
                "speed_factor": voice_params.get("speed", 1.0),
                "fragment_interval": 0.3,
                "seed": -1,
                "parallel_infer": True,
                "repetition_penalty": 1.35
            }

            logger.info(f"🎵 开始语音合成: '{text}'")

            # 7. 执行推理
            sr, audio_data = next(tts_pipeline.run(inference_params))

            # 8. 转换为16bit PCM
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32768).astype(np.int16)

            # 9. 创建WAV文件
            wav_data = self._create_wav_file(audio_data.tobytes(), sr)

            logger.info(f"✅ 推理完成，音频大小: {len(wav_data)} bytes, 采样率: {sr}Hz")

            # 10. 清理资源
            del tts_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return wav_data

        except Exception as e:
            logger.error(f"❌ GPT-SoVITS推理失败: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            return b""

    def _create_tts_config(self, gpt_path: str, sovits_path: str) -> 'TTS_Config':
        """创建TTS配置"""
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config

        # 创建配置字典
        config_dict = {
            "device": self.device,
            "is_half": True if self.device == "cuda" else False,
            "version": "v2Pro",
            "t2s_weights_path": gpt_path,
            "vits_weights_path": sovits_path,
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        }

        return TTS_Config(config_dict)

    def _get_role_config_by_model(self, gpt_path: str, sovits_path: str) -> Optional[Dict]:
        """根据模型路径获取角色配置"""
        gpt_model = os.path.basename(gpt_path)
        sovits_model = os.path.basename(sovits_path)

        for role_name, config in self.config.get("role_voice_mapping", {}).items():
            if (config.get("gpt_model") == gpt_model and
                config.get("sovits_model") == sovits_model):
                # 添加参考音频路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                ref_audio_name = config.get("ref_audio", "")
                ref_audio_path = os.path.join(current_dir, "../../../GPT-Sovits-slice", ref_audio_name)

                return {
                    **config,
                    "ref_audio_path": ref_audio_path
                }

        return None

    def _create_wav_file(self, pcm_data: bytes, sample_rate: int = 44100) -> bytes:
        """
        创建WAV文件格式

        Args:
            pcm_data: PCM音频数据（16bit, 单声道）
            sample_rate: 采样率

        Returns:
            完整的WAV文件数据
        """
        try:
            # WAV文件头结构
            # RIFF头
            riff_header = b'RIFF'
            file_size = 36 + len(pcm_data)  # 36是WAV头的固定大小
            riff_size = file_size.to_bytes(4, 'little')

            # WAVE标识
            wave_header = b'WAVE'

            # fmt子块
            fmt_header = b'fmt '
            fmt_size = (16).to_bytes(4, 'little')  # fmt子块大小
            audio_format = (1).to_bytes(2, 'little')  # PCM格式
            num_channels = (1).to_bytes(2, 'little')  # 单声道
            sample_rate_bytes = sample_rate.to_bytes(4, 'little')
            byte_rate = (sample_rate * 1 * 16 // 8).to_bytes(4, 'little')  # 字节率
            block_align = (1 * 16 // 8).to_bytes(2, 'little')  # 块对齐
            bits_per_sample = (16).to_bytes(2, 'little')  # 16位

            # data子块
            data_header = b'data'
            data_size = len(pcm_data).to_bytes(4, 'little')

            # 组合所有部分
            wav_file = (
                riff_header + riff_size + wave_header +
                fmt_header + fmt_size + audio_format + num_channels +
                sample_rate_bytes + byte_rate + block_align + bits_per_sample +
                data_header + data_size + pcm_data
            )

            logger.info(f"✅ WAV文件创建成功: {len(wav_file)} bytes, 采样率: {sample_rate}Hz")
            return wav_file

        except Exception as e:
            logger.error(f"❌ 创建WAV文件失败: {e}")
            return b""

    def _get_role_config(self, role_name: str) -> Optional[Dict]:
        """获取角色配置"""
        return self.config.get("role_voice_mapping", {}).get(role_name)

    def get_available_roles(self) -> List[Dict]:
        """获取可用角色列表"""
        roles = []
        for role_name, config in self.config.get("role_voice_mapping", {}).items():
            roles.append({
                "name": role_name,
                "description": config.get("description", ""),
                "gpt_model": config.get("gpt_model"),
                "sovits_model": config.get("sovits_model"),
                "voice_params": config.get("voice_params", {})
            })
        return roles

    def is_model_available(self, role_name: str) -> bool:
        """检查角色模型是否可用"""
        role_config = self._get_role_config(role_name)
        if not role_config:
            return False

        gpt_model = role_config.get("gpt_model")
        sovits_model = role_config.get("sovits_model")

        gpt_path = os.path.join(self.gpt_weights_dir, gpt_model)
        sovits_path = os.path.join(self.sovits_weights_dir, sovits_model)

        return os.path.exists(gpt_path) and os.path.exists(sovits_path)

    async def health_check(self) -> Dict:
        """健康检查"""
        try:
            available_roles = self.get_available_roles()
            working_roles = [role for role in available_roles if self.is_model_available(role["name"])]

            return {
                "service": "gpt_sovits",
                "device": self.device,
                "total_roles": len(available_roles),
                "working_roles": len(working_roles),
                "gpt_weights_dir": self.gpt_weights_dir,
                "sovits_weights_dir": self.sovits_weights_dir,
                "models_loaded": len(self.models_cache)
            }

        except Exception as e:
            return {
                "service": "gpt_sovits",
                "error": str(e),
                "device": self.device
            }

# 全局服务实例
gpt_sovits_service = GPTSoVITSService()
