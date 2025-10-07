"""
语音服务主入口
统一管理STT、TTS和语音交互功能
"""

import json
import asyncio
import logging
import os
import sys
import traceback
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# 添加GPT_SoVITS到Python路径（必须在导入之前）
current_dir = os.path.dirname(os.path.abspath(__file__))
gpt_sovits_path = os.path.abspath(os.path.join(current_dir, "../../../GPT_SoVITS"))
if gpt_sovits_path not in sys.path:
    sys.path.insert(0, gpt_sovits_path)
    logger.info(f"✅ 添加GPT_SoVITS路径到Python路径: {gpt_sovits_path}")

# 计算绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../../../")
default_config_path = os.path.join(project_root, "voice_config.json")

class VoiceService:
    """语音服务主控制器"""

    def __init__(self, config_path: str = default_config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.input_mode = "text"  # text 或 voice

        # 服务组件
        self.stt_service = None
        self.tts_service = None
        self.social_service = None

        # 模型缓存
        self.models_cache = {}

    def _load_config(self) -> Dict:
        """加载语音配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"❌ 配置文件 {self.config_path} 不存在")
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            raise Exception(f"加载配置文件失败: {e}")

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "model_paths": {
                "gpt_weights_dir": "./GPT_weights_v2Pro",
                "sovits_weights_dir": "./SoVITS_weights_v2Pro"
            },
            "role_voice_mapping": {},
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
        """初始化语音服务"""
        try:
            logger.info("🚀 初始化语音服务...")

            # 初始化语音识别服务
            await self._initialize_stt_service()

            # 初始化GPT-SoVITS服务
            await self._initialize_tts_service()

            logger.info("✅ 语音服务初始化完成")
            return True

        except Exception as e:
            logger.error(f"❌ 语音服务初始化失败: {e}")
            return False

    async def _initialize_stt_service(self):
        """初始化语音识别服务"""
        # 仅使用Whisper STT服务
        try:
            logger.info("🎯 初始化Whisper STT服务...")
            from .whisper_stt_service import WhisperSTTService
            model_path = os.path.join(os.path.dirname(__file__), "../../../openai-whisper-large-v3")
            self.stt_service = WhisperSTTService(model_path)
            success = await self.stt_service.initialize()
            if success:
                logger.info("✅ Whisper Large v3 STT服务初始化完成")
            else:
                logger.warning("⚠️ Whisper STT服务初始化失败，将使用纯文本模式")
                self.stt_service = None
        except ImportError as e:
            logger.warning(f"⚠️ Whisper STT服务导入失败: {e}，将使用纯文本模式")
            self.stt_service = None
        except Exception as e:
            logger.warning(f"⚠️ Whisper STT服务初始化异常: {e}，将使用纯文本模式")
            self.stt_service = None

    async def _initialize_tts_service(self):
        """初始化语音合成服务"""
        try:
            logger.info("🎯 开始初始化GPT-SoVITS服务...")
            logger.info(f"📁 当前Python路径包含GPT_SoVITS: {'GPT_SoVITS' in str(sys.path)}")
            logger.info(f"📂 sys.path: {[p for p in sys.path if 'GPT_SoVITS' in p or 'mental-health' in p]}")

            from .gpt_sovits_service import GPTSoVITSService
            logger.info("✅ GPT-SoVITS模块导入成功")

            self.tts_service = GPTSoVITSService(self.config_path)
            logger.info("✅ GPT-SoVITS服务实例创建成功")

            await self.tts_service.initialize()
            logger.info("✅ GPT-SoVITS服务初始化完成")
        except Exception as e:
            logger.warning(f"⚠️ GPT-SoVITS服务初始化失败: {e}，将使用纯文本模式")
            logger.warning(f"🔍 详细错误信息: {traceback.format_exc()}")
            self.tts_service = None

    async def process_input(self, input_data: Any, input_type: str = "text") -> Tuple[str, str]:
        """
        处理用户输入（支持文本和语音）

        Args:
            input_data: 输入数据（文本字符串或音频字节数据）
            input_type: 输入类型 ("text" 或 "voice")

        Returns:
            Tuple[processed_text, actual_input_type]
        """
        try:
            if input_type == "voice" and self.stt_service:
                # 语音输入：使用Whisper STT转换
                logger.info("🎤 处理语音输入...")
                text = await self.stt_service.speech_to_text(input_data)
                return text, "voice"
            else:
                # 文本输入：直接使用
                logger.info("📝 处理文本输入...")
                return str(input_data), "text"

        except Exception as e:
            logger.error(f"❌ 输入处理失败: {e}")
            # 降级到文本模式
            if input_type == "voice":
                return "抱歉，我没有听清楚，请用文本回复。", "text"
            return str(input_data), "text"

    async def generate_response(
        self,
        text: str,
        role_name: str,
        enable_voice: bool = True,
        emotion_params: Dict = None
    ) -> Dict:
        """
        生成AI回复（支持语音合成）

        Args:
            text: 用户输入文本
            role_name: AI角色名称
            enable_voice: 是否启用语音合成
            emotion_params: 情绪参数

        Returns:
            包含文本回复和语音数据的字典
        """
        try:
            # 获取角色配置
            role_config = self._get_role_config(role_name)
            if not role_config:
                return {
                    "success": False,
                    "error": f"角色 '{role_name}' 配置不存在",
                    "text": "抱歉，角色配置有误。",
                    "audio": None,
                    "role": role_name
                }

            # 生成文本回复（使用社交实验室服务）
            if self.social_service:
                response_data = await self.social_service.generate_ai_response(
                    1, text, None, None  # session_id, voice_emotions, face_emotions
                )
            else:
                # 简化的回复生成
                response_data = await self._generate_simple_response(text, role_name)

            if "error" in response_data:
                return {
                    "success": False,
                    "error": response_data["error"],
                    "text": "抱歉，暂时无法生成回复。",
                    "audio": None,
                    "role": role_name
                }

            ai_text = response_data.get("response", "你好，请继续练习。")

            # 生成语音回复（如果启用）
            audio_data = None
            if enable_voice and self.tts_service:
                try:
                    logger.info(f"🎵 为角色 '{role_name}' 生成语音回复...")
                    audio_data = await self.tts_service.synthesize_speech(
                        ai_text, role_name, emotion_params
                    )
                except Exception as e:
                    logger.error(f"❌ 语音合成失败: {e}")
                    # 语音合成失败不影响文本回复

            return {
                "success": True,
                "text": ai_text,
                "audio": audio_data,
                "role": role_name,
                "input_mode": self.input_mode
            }

        except Exception as e:
            logger.error(f"❌ 生成回复失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "抱歉，服务暂时不可用。",
                "audio": None,
                "role": role_name
            }

    async def _generate_simple_response(self, user_text: str, role_name: str) -> Dict:
        """简化的回复生成（当社交服务不可用时）"""
        try:
            from app.services.deepseek_service import deepseek_service

            prompt = f"""
你正在扮演一个{role_name}，与用户进行社交练习对话。

用户输入："{user_text}"

请以{role_name}的身份，给出一个自然、适当的回复，帮助用户练习社交技能。
回复要简洁、有建设性。
"""

            messages = [
                {"role": "system", "content": "你是一个专业的社交技能教练AI助手。"},
                {"role": "user", "content": prompt}
            ]

            result = await deepseek_service.chat_completion(messages, temperature=0.8)

            if result["success"]:
                return {
                    "response": result["response"],
                    "role": role_name,
                    "timestamp": asyncio.get_event_loop().time()
                }
            else:
                return {
                    "response": "我理解你的分享，请继续练习。",
                    "role": role_name,
                    "timestamp": asyncio.get_event_loop().time()
                }

        except Exception as e:
            logger.error(f"❌ 简化回复生成失败: {e}")
            return {
                "response": "请继续练习，你做得很好。",
                "role": role_name,
                "timestamp": asyncio.get_event_loop().time()
            }

    def _get_role_config(self, role_name: str) -> Optional[Dict]:
        """获取角色配置"""
        return self.config.get("role_voice_mapping", {}).get(role_name)

    def switch_input_mode(self, mode: str) -> bool:
        """切换输入模式"""
        if mode in ["text", "voice"]:
            self.input_mode = mode
            logger.info(f"🔄 切换输入模式到: {mode}")
            return True
        return False

    def get_available_roles(self) -> List[Dict]:
        """获取可用角色列表"""
        roles = []
        for role_name, config in self.config.get("role_voice_mapping", {}).items():
            roles.append({
                "name": role_name,
                "description": config.get("description", ""),
                "voice_enabled": self.tts_service is not None,
                "gpt_model": config.get("gpt_model"),
                "sovits_model": config.get("sovits_model")
            })
        return roles

    def get_input_modes(self) -> Dict:
        """获取输入模式状态"""
        return {
            "current_mode": self.input_mode,
            "available_modes": self.config.get("input_modes", {}),
            "stt_available": self.stt_service is not None,
            "tts_available": self.tts_service is not None
        }

    async def health_check(self) -> Dict:
        """健康检查"""
        return {
            "service_status": "healthy",
            "input_mode": self.input_mode,
            "stt_service": self.stt_service is not None,
            "tts_service": self.tts_service is not None,
            "available_roles": len(self.get_available_roles()),
            "config_loaded": self.config is not None
        }

# 全局语音服务实例
voice_service = VoiceService()
