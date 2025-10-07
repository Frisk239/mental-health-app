"""
Paraformer中文语音识别服务
基于ONNX Runtime的本地中文语音识别
"""

import os
import json
import logging
import asyncio
import pickle
from typing import Optional, Dict, Any, Tuple
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO

logger = logging.getLogger(__name__)

class ParaformerSTTService:
    """基于Paraformer的中文语音识别服务"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.session = None
        self.token_list = None
        self.am_mvn = None
        self.initialized = False

    async def initialize(self) -> bool:
        """初始化Paraformer模型"""
        try:
            logger.info("🚀 开始初始化Paraformer中文语音识别模型...")

            # 检查模型文件是否存在
            model_path = os.path.join(self.model_dir, "model.onnx")
            token_path = os.path.join(self.model_dir, "token_list.pkl")
            am_mvn_path = os.path.join(self.model_dir, "am.mvn")

            if not all(os.path.exists(f) for f in [model_path, token_path, am_mvn_path]):
                logger.error(f"❌ 模型文件不完整: {self.model_dir}")
                logger.error(f"需要文件: {model_path}, {token_path}, {am_mvn_path}")
                return False

            # 直接使用ONNX Runtime加载模型
            try:
                import onnxruntime as ort
                logger.info("✅ ONNX Runtime库导入成功")

                # 创建推理会话
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']  # 使用CPU
                )

                logger.info("✅ ONNX模型加载成功")

                # 加载词汇表
                with open(token_path, 'rb') as f:
                    self.token_list = pickle.load(f)
                logger.info(f"✅ 词汇表加载成功，包含 {len(self.token_list)} 个token")

                # 加载音频归一化参数
                self.am_mvn = self._load_am_mvn(am_mvn_path)
                logger.info("✅ 音频归一化参数加载成功")

                self.initialized = True
                logger.info("✅ Paraformer模型初始化完成")
                return True

            except ImportError:
                logger.error("❌ ONNX Runtime库未安装，请运行: pip install onnxruntime")
                return False
            except Exception as e:
                logger.error(f"❌ ONNX模型加载失败: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Paraformer服务初始化失败: {e}")
            return False

    async def speech_to_text(self, audio_data: bytes, language: str = "zh") -> str:
        """
        将语音数据转换为文字

        Args:
            audio_data: 音频字节数据
            language: 语言代码（默认中文）

        Returns:
            识别出的文字
        """
        try:
            if not self.initialized or not self.session:
                logger.error("❌ Paraformer模型未初始化")
                return "语音识别服务暂时不可用，请用文本回复。"

            logger.info("🎤 开始Paraformer语音识别...")

            # 音频预处理
            audio_array, sample_rate = self._preprocess_audio(audio_data)
            if audio_array is None:
                return "音频格式不支持，请用文本回复。"

            logger.info(f"📊 音频预处理完成，长度: {len(audio_array)} 采样点，采样率: {sample_rate}")

            # 特征提取
            features = self._extract_features(audio_array, sample_rate)
            if features is None:
                return "音频特征提取失败，请用文本回复。"

            logger.info(f"🔍 特征提取完成，形状: {features.shape}")

            # ONNX模型推理
            try:
                result = self._run_inference(features)
                logger.info(f"🔍 推理结果: {result}")

                # 解码结果
                text = self._decode_result(result)

                # 清理和格式化文本
                text = self._postprocess_text(text)

                if text.strip():
                    logger.info(f"✅ 语音识别成功: {text}")
                    return text
                else:
                    logger.warning("⚠️ 语音识别结果为空")
                    return "抱歉，我没有听清楚，请再说一遍或用文本回复。"

            except Exception as e:
                logger.error(f"❌ Paraformer推理失败: {e}")
                return "语音识别服务异常，请用文本回复。"

        except Exception as e:
            logger.error(f"❌ 语音识别处理异常: {e}")
            return "语音识别服务暂时不可用，请用文本回复。"

    def _preprocess_audio(self, audio_data: bytes) -> Optional[Tuple[np.ndarray, int]]:
        """音频预处理：支持WebM格式转换"""
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
                return audio_array.astype(np.float32), sample_rate

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

    def _load_am_mvn(self, mvn_path: str) -> Dict[str, np.ndarray]:
        """加载音频归一化参数"""
        try:
            mvn_stats = {}
            with open(mvn_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析Kaldi格式的mvn文件
            # 提取数值部分（跳过XML标签）
            import re

            # 查找所有方括号内的数值
            bracket_matches = re.findall(r'\[([^\]]+)\]', content)
            if bracket_matches:
                for i, match in enumerate(bracket_matches):
                    try:
                        # 解析数值
                        values = []
                        for val in match.split():
                            val = val.strip()
                            if val and val != ']':
                                try:
                                    values.append(float(val))
                                except ValueError:
                                    continue

                        if values:
                            if i == 0:
                                mvn_stats['mean'] = np.array(values)
                            elif i == 1:
                                mvn_stats['std'] = np.array(values)
                            else:
                                mvn_stats[f'param_{i}'] = np.array(values)

                    except Exception as e:
                        logger.warning(f"⚠️ 解析第{i}个参数块失败: {e}")
                        continue

            logger.info(f"✅ 加载了 {len(mvn_stats)} 个归一化参数")
            return mvn_stats

        except Exception as e:
            logger.warning(f"⚠️ 加载音频归一化参数失败: {e}，将使用默认参数")
            return {}

    def _extract_features(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """特征提取"""
        try:
            # 使用librosa提取FBank特征
            # Paraformer模型需要560维特征（根据ONNX模型规格）
            fbank = librosa.feature.melspectrogram(
                y=audio_array,
                sr=sample_rate,    # 使用实际采样率
                n_fft=400,
                hop_length=160,    # 10ms hop
                n_mels=560,        # 560维特征（模型要求）
                fmin=20,
                fmax=sample_rate // 2  # Nyquist频率
            )

            # 转换为分贝
            fbank = librosa.power_to_db(fbank, ref=np.max)

            # 应用CMVN（如果有参数）
            if self.am_mvn:
                # 这里应该应用均值和方差归一化
                # 暂时简化处理
                pass

            # 添加时间维度并转置为 (T, F)
            features = fbank.T  # (time, freq)

            logger.info(f"✅ 特征提取完成，形状: {features.shape}")
            return features.astype(np.float32)

        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
            return None

    def _run_inference(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """ONNX模型推理"""
        try:
            if not self.session:
                raise Exception("ONNX session未初始化")

            # 获取模型输入名称
            input_names = [input.name for input in self.session.get_inputs()]
            logger.info(f"🔍 模型输入: {input_names}")

            # 准备输入数据
            # Paraformer通常需要多个输入，这里需要根据实际模型调整
            inputs = {}

            # 主要输入：语音特征
            if 'speech' in input_names:
                # 添加批次维度
                speech_input = np.expand_dims(features, axis=0)  # (1, T, F)
                inputs['speech'] = speech_input
                logger.info(f"📊 语音输入形状: {speech_input.shape}")

            # 语音长度
            if 'speech_lengths' in input_names:
                speech_lengths = np.array([features.shape[0]], dtype=np.int32)
                inputs['speech_lengths'] = speech_lengths

            # 执行推理
            outputs = self.session.run(None, inputs)

            # 解析输出
            output_names = [output.name for output in self.session.get_outputs()]
            logger.info(f"🔍 模型输出: {output_names}")

            result = {}
            for name, output in zip(output_names, outputs):
                result[name] = output

            return result

        except Exception as e:
            logger.error(f"❌ ONNX推理失败: {e}")
            raise

    def _decode_result(self, result: Dict[str, np.ndarray]) -> str:
        """解码推理结果"""
        try:
            # Paraformer通常使用CTC解码或Attention解码
            # 这里需要根据实际模型输出调整

            # 查找可能的输出
            if 'logits' in result:
                logits = result['logits'][0]  # 移除批次维度
            elif 'output' in result:
                logits = result['output'][0]
            else:
                # 使用第一个输出
                logits = list(result.values())[0][0]

            logger.info(f"🔍 Logits形状: {logits.shape}")

            # CTC解码（贪婪解码）
            if len(logits.shape) == 2:  # (T, C)
                # 获取每个时间步的预测token
                pred_tokens = np.argmax(logits, axis=1)

                # 移除连续重复的token
                decoded_tokens = []
                prev_token = -1
                for token in pred_tokens:
                    if token != prev_token and token != 0:  # 0通常是blank token
                        decoded_tokens.append(token)
                    prev_token = token

                # 转换为文本
                if self.token_list:
                    text = ""
                    for token in decoded_tokens:
                        if 0 <= token < len(self.token_list):
                            token_text = self.token_list[token]
                            if isinstance(token_text, str):
                                text += token_text
                            else:
                                text += str(token_text)
                        else:
                            logger.warning(f"⚠️ Token {token} 超出词汇表范围")
                else:
                    text = " ".join([str(t) for t in decoded_tokens])

                return text
            else:
                logger.warning(f"⚠️ 意外的logits形状: {logits.shape}")
                return ""

        except Exception as e:
            logger.error(f"❌ 结果解码失败: {e}")
            return ""

    def _postprocess_text(self, text: str) -> str:
        """文本后处理"""
        if not text:
            return ""

        # 移除多余的空格
        text = " ".join(text.split())

        # 确保以中文标点结束（如果没有标点）
        if text and not any(p in text[-1] for p in "。！？，；："):
            # 检查是否可能是句子结尾
            if len(text) > 10:  # 较长的文本
                text += "。"

        return text.strip()

    def is_configured(self) -> bool:
        """检查服务是否已配置"""
        return self.initialized and self.session is not None

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            return {
                "service": "paraformer_stt",
                "configured": self.is_configured(),
                "model_dir": self.model_dir,
                "model_loaded": self.model is not None,
                "initialized": self.initialized
            }
        except Exception as e:
            return {
                "service": "paraformer_stt",
                "configured": False,
                "error": str(e)
            }

# 全局服务实例
paraformer_stt_service = ParaformerSTTService(
    model_dir=os.path.join(os.path.dirname(__file__), "../../../asr_zh")
)
