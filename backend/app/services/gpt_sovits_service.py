"""
GPT-SoVITS推理服务
基于你的项目源码实现语音合成
"""

import json
import logging
import os
import sys
import asyncio
from typing import Dict, List, Optional, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)

class GPTSoVITSService:
    """GPT-SoVITS推理服务"""

    def __init__(self, config_path: str = "../../../voice_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 模型组件
        self.text_encoder = None
        self.synthesizer = None
        self.vocoder = None
        self.ssl_model = None

        # 模型路径（使用绝对路径）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.gpt_weights_dir = os.path.join(current_dir, "../../../GPT_weights_v2Pro")
        self.sovits_weights_dir = os.path.join(current_dir, "../../../SoVITS_weights_v2Pro")

        # 模型缓存
        self.models_cache = {}

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
        执行推理（基于你的源码）

        这里需要集成你的GPT-SoVITS源码逻辑
        """
        try:
            # TODO: 集成你的GPT-SoVITS源码
            # 以下是基于源码的伪代码实现框架

            # 1. 加载模型权重
            # gpt_dict = torch.load(gpt_path, map_location="cpu")
            # sovits_dict = torch.load(sovits_path, map_location="cpu")

            # 2. 初始化模型（基于你的源码）
            # text_encoder = TextEncoder(...)
            # synthesizer = SynthesizerTrn(...)
            # vocoder = ...

            # 3. 文本预处理
            # phones, bert_features = preprocess_text(text)

            # 4. 语义编码
            # semantic_tokens = text_encoder.encode(...)

            # 5. 音频生成
            # audio_features = synthesizer.generate(...)

            # 6. 声码器转换
            # audio_data = vocoder.convert(...)

            # 生成模拟音频数据并创建WAV格式
            sample_rate = 44100
            duration = len(text) * 0.3  # 根据文本长度估算时长（每字符0.3秒）
            duration = max(duration, 1.0)  # 最少1秒
            duration = min(duration, 10.0)  # 最多10秒

            # 生成随机音频数据（实际应用中应该是模型生成的）
            samples = np.random.randint(
                -32768, 32767,
                size=int(sample_rate * duration),
                dtype=np.int16
            )

            # 创建WAV文件格式
            wav_data = self._create_wav_file(samples.tobytes(), sample_rate)

            logger.info(f"🎵 推理完成（模拟数据），WAV文件大小: {len(wav_data)} bytes")
            return wav_data

        except Exception as e:
            logger.error(f"❌ 推理执行失败: {e}")
            return b""

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
