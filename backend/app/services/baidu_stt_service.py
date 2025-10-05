"""
百度语音识别服务
提供实时语音转文字功能
"""

import json
import logging
import os
import base64
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, Optional
import aiohttp

logger = logging.getLogger(__name__)

class BaiduSTTService:
    """百度语音识别服务"""

    def __init__(self):
        # 从环境变量获取百度API凭证
        self.api_key = os.getenv("BAIDU_API_KEY")
        self.secret_key = os.getenv("BAIDU_SECRET_KEY")
        self.access_token = None
        self.token_expire_time = 0

        # API配置
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.stt_url = "https://vop.baidu.com/server_api"

        # 语音识别参数
        self.format = "wav"  # 音频格式：wav, pcm, amr, m4a
        self.rate = 16000   # 采样率：8000, 16000
        self.channel = 1    # 声道数：1, 2
        self.dev_pid = 1537 # 语言模型：1537(普通话), 1737(英语), 1637(粤语), 1837(四川话)

    async def _get_access_token(self) -> Optional[str]:
        """获取百度API访问令牌"""
        try:
            if not self.api_key or not self.secret_key:
                logger.error("❌ 百度API凭证未配置")
                return None

            # 检查令牌是否过期
            current_time = time.time()
            if self.access_token and current_time < self.token_expire_time:
                return self.access_token

            # 构造请求参数
            params = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.token_url, data=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data.get("access_token")
                        # 令牌有效期通常是30天，但这里设置为25天以确保安全
                        self.token_expire_time = current_time + 25 * 24 * 3600

                        logger.info("✅ 百度访问令牌获取成功")
                        return self.access_token
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 获取百度访问令牌失败: {response.status} - {error_text}")
                        return None

        except Exception as e:
            logger.error(f"❌ 获取百度访问令牌异常: {e}")
            return None

    async def speech_to_text(self, audio_data: bytes, format: str = "wav") -> str:
        """
        将语音数据转换为文字

        Args:
            audio_data: 音频字节数据
            format: 音频格式

        Returns:
            识别出的文字
        """
        try:
            # 获取访问令牌
            access_token = await self._get_access_token()
            if not access_token:
                return "语音识别服务暂时不可用，请用文本回复。"

            # 构造请求头
            headers = {
                "Content-Type": f"audio/{format};rate={self.rate}"
            }

            # 构造请求参数
            params = {
                "cuid": "mental-health-app",  # 用户唯一标识
                "token": access_token,
                "dev_pid": self.dev_pid
            }

            # 发送语音识别请求
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.stt_url,
                    headers=headers,
                    params=params,
                    data=audio_data
                ) as response:

                    if response.status == 200:
                        result = await response.json()

                        # 检查识别结果
                        if result.get("err_no") == 0:
                            # 识别成功
                            text = result.get("result", [""])[0]
                            logger.info(f"✅ 语音识别成功: {text}")
                            return text.strip()
                        else:
                            # 识别失败
                            error_msg = result.get("err_msg", "未知错误")
                            logger.error(f"❌ 语音识别失败: {error_msg}")
                            return "抱歉，我没有听清楚，请再说一遍或用文本回复。"
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ 语音识别请求失败: {response.status} - {error_text}")
                        return "语音识别服务暂时不可用，请用文本回复。"

        except Exception as e:
            logger.error(f"❌ 语音识别异常: {e}")
            return "语音识别服务异常，请用文本回复。"

    async def speech_to_text_with_punctuation(self, audio_data: bytes) -> str:
        """
        语音转文字（带标点符号）

        百度语音识别默认会添加标点，这里提供一个包装方法
        """
        text = await self.speech_to_text(audio_data)

        # 简单的标点优化（可以后续增强）
        if text and not any(p in text for p in "。！？，"):
            # 如果没有中文标点，添加句号
            text = text.rstrip("。！？，") + "。"

        return text

    def is_configured(self) -> bool:
        """检查服务是否已配置"""
        return bool(self.api_key and self.secret_key)

    async def health_check(self) -> Dict:
        """健康检查"""
        try:
            access_token = await self._get_access_token()
            return {
                "service": "baidu_stt",
                "configured": self.is_configured(),
                "token_valid": access_token is not None,
                "api_key_set": bool(self.api_key),
                "secret_key_set": bool(self.secret_key)
            }
        except Exception as e:
            return {
                "service": "baidu_stt",
                "configured": False,
                "error": str(e)
            }

# 全局服务实例
baidu_stt_service = BaiduSTTService()
