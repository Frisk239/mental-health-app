"""
Whisper中文语音识别服务
基于OpenAI Whisper Large v3模型的高质量中文语音识别
"""

import os
import logging
import asyncio
import tempfile
from typing import Optional, Tuple
import numpy as np
import librosa

logger = logging.getLogger(__name__)

class WhisperSTTService:
    """基于OpenAI Whisper的中文语音识别服务"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.initialized = False

    async def initialize(self) -> bool:
        """初始化Whisper模型"""
        try:
            logger.info("🚀 开始初始化Whisper Large v3模型...")

            # 检查模型文件是否存在
            model_files = [
                os.path.join(self.model_path, "config.json"),
                os.path.join(self.model_path, "model.safetensors"),
                os.path.join(self.model_path, "preprocessor_config.json")
            ]

            if not all(os.path.exists(f) for f in model_files):
                logger.error(f"❌ Whisper模型文件不完整: {self.model_path}")
                logger.error(f"需要文件: {model_files}")
                return False

            # 动态导入whisper库
            try:
                import whisper
                logger.info("✅ Whisper库导入成功")
            except ImportError:
                logger.error("❌ Whisper库未安装，请运行: pip install openai-whisper")
                return False

            # 加载本地Whisper模型
            try:
                logger.info("📥 加载Whisper Large v3模型...")
                self.model = whisper.load_model(
                    "large-v3",
                    download_root=self.model_path,
                    device="cpu"  # 使用CPU
                )
                logger.info("✅ Whisper模型加载成功")
                self.initialized = True
                return True

            except Exception as e:
                logger.error(f"❌ Whisper模型加载失败: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Whisper服务初始化失败: {e}")
            return False

    async def speech_to_text(self, audio_data: bytes, language: str = "zh") -> str:
        """
        将语音数据转换为文字

        Args:
            audio_data: 音频字节数据（WebM格式）
            language: 语言代码（默认中文）

        Returns:
            识别出的文字
        """
        try:
            if not self.initialized or not self.model:
                logger.error("❌ Whisper模型未初始化")
                return "语音识别服务暂时不可用，请用文本回复。"

            logger.info("🎤 开始Whisper语音识别...")

            # 音频预处理：WebM转WAV
            audio_array = self._preprocess_audio(audio_data)
            if audio_array is None:
                return "音频格式不支持，请用文本回复。"

            logger.info(f"📊 音频预处理完成，长度: {len(audio_array)} 采样点")

            # 使用Whisper进行语音识别
            try:
                logger.info("🔍 开始Whisper推理...")

                # Whisper期望numpy数组或音频文件路径
                # 创建临时WAV文件供Whisper使用
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                    wav_path = temp_wav.name

                try:
                    # 保存为WAV文件
                    import soundfile as sf
                    sf.write(wav_path, audio_array, 16000, format='WAV')

                    # 使用Whisper进行识别
                    result = self.model.transcribe(
                        wav_path,
                        language=language,  # 指定中文
                        task="transcribe",  # 转录任务
                        verbose=False,     # 不显示详细信息
                        fp16=False,        # CPU模式不使用FP16
                        temperature=0,     # 确定性解码
                        no_speech_threshold=0.6,  # 静音检测阈值
                        logprob_threshold=-1.0,   # 日志概率阈值
                        compression_ratio_threshold=2.4,  # 压缩比阈值
                        condition_on_previous_text=True  # 条件文本
                    )

                    recognized_text = result["text"].strip()
                    logger.info(f"✅ Whisper识别成功: {recognized_text}")

                    # 清理和格式化文本
                    text = self._postprocess_text(recognized_text)

                    if text.strip():
                        logger.info(f"🎯 最终识别结果: {text}")
                        return text
                    else:
                        logger.warning("⚠️ Whisper识别结果为空")
                        return "抱歉，我没有听清楚，请再说一遍或用文本回复。"

                finally:
                    # 清理临时文件
                    try:
                        if os.path.exists(wav_path):
                            os.unlink(wav_path)
                    except Exception as e:
                        logger.warning(f"⚠️ 清理临时文件失败: {e}")

            except Exception as e:
                logger.error(f"❌ Whisper推理失败: {e}")
                return "语音识别服务异常，请用文本回复。"

        except Exception as e:
            logger.error(f"❌ 语音识别处理异常: {e}")
            return "语音识别服务暂时不可用，请用文本回复。"

    def _preprocess_audio(self, audio_data: bytes) -> Optional[np.ndarray]:
        """音频预处理：WebM格式转换"""
        try:
            import tempfile
            import os

            # 保存WebM数据到临时文件
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
                temp_webm.write(audio_data)
                webm_path = temp_webm.name

            try:
                # 使用ffmpeg转换WebM到WAV
                wav_path = webm_path.replace('.webm', '.wav')

                try:
                    import ffmpeg
                    # 使用ffmpeg-python转换音频
                    ffmpeg.input(webm_path).output(
                        wav_path,
                        acodec='pcm_s16le',  # 16位PCM
                        ac=1,                # 单声道
                        ar='16000',         # 16kHz采样率
                        f='wav'             # WAV格式
                    ).run(quiet=True, overwrite_output=True)

                    logger.info("✅ WebM转WAV成功")

                except ImportError:
                    logger.warning("⚠️ ffmpeg-python不可用，尝试使用subprocess调用ffmpeg")
                    import subprocess

                    # 使用subprocess调用系统ffmpeg
                    result = subprocess.run([
                        'ffmpeg', '-i', webm_path, '-acodec', 'pcm_s16le',
                        '-ac', '1', '-ar', '16000', '-f', 'wav', '-y', wav_path
                    ], capture_output=True, text=True)

                    if result.returncode != 0:
                        logger.error(f"❌ ffmpeg转换失败: {result.stderr}")
                        return None

                    logger.info("✅ ffmpeg转换成功")

                # 使用librosa加载WAV文件，保持原始采样率
                audio_array, sample_rate = librosa.load(wav_path, sr=None, mono=True)
                logger.info(f"📊 音频加载成功，长度: {len(audio_array)} 采样点，采样率: {sample_rate}")

                # 确保音频长度合适
                if len(audio_array) < 1600:  # 少于0.1秒
                    logger.warning("⚠️ 音频太短，跳过识别")
                    return None

                if len(audio_array) > 16000 * 60:  # 多于60秒
                    logger.warning("⚠️ 音频太长，截取前60秒")
                    audio_array = audio_array[:16000 * 60]

                # 音频归一化
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))

                logger.info(f"✅ 音频预处理完成，最终长度: {len(audio_array)}")
                return audio_array.astype(np.float32)

            finally:
                # 清理临时文件
                try:
                    if os.path.exists(webm_path):
                        os.unlink(webm_path)
                    if os.path.exists(wav_path):
                        os.unlink(wav_path)
                except Exception as e:
                    logger.warning(f"⚠️ 清理临时文件失败: {e}")

        except Exception as e:
            logger.error(f"❌ 音频预处理失败: {e}")
            return None

    def _postprocess_text(self, text: str) -> str:
        """文本后处理"""
        if not text:
            return ""

        # 移除多余的空格和换行
        text = " ".join(text.split())

        # Whisper有时会在文本开头添加语言标识，移除它
        if text.startswith(("中文", "Chinese", "ZH", "zh")):
            # 移除开头的语言标识
            words = text.split()
            if len(words) > 1:
                text = " ".join(words[1:])

        # 移除可能的标点符号问题
        text = text.strip(".,，。")

        # 确保以中文标点结束（如果没有标点）
        if text and not any(p in text[-1] for p in "。！？，；："):
            # 检查是否可能是句子结尾
            if len(text) > 10:  # 较长的文本
                text += "。"

        return text.strip()

    def is_configured(self) -> bool:
        """检查服务是否已配置"""
        return self.initialized and self.model is not None

    async def health_check(self) -> dict:
        """健康检查"""
        try:
            return {
                "service": "whisper_stt",
                "configured": self.is_configured(),
                "model_path": self.model_path,
                "model_loaded": self.model is not None,
                "initialized": self.initialized
            }
        except Exception as e:
            return {
                "service": "whisper_stt",
                "configured": False,
                "error": str(e)
            }

# 全局服务实例
whisper_stt_service = WhisperSTTService(
    model_path=os.path.join(os.path.dirname(__file__), "../../../openai-whisper-large-v3")
)
