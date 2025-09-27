"""
DeepSeek API服务
用于处理对话生成和文本分析
"""

import os
import httpx
import json
from typing import Dict, List, Optional
from pydantic import BaseModel
import asyncio
from datetime import datetime

class DeepSeekConfig:
    """DeepSeek API配置"""
    BASE_URL = "https://api.deepseek.com/v1"
    MODEL = "deepseek-chat"  # 或 deepseek-coder

    @classmethod
    def get_api_key(cls) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        return api_key

class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str  # "user", "assistant", "system"
    content: str

class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[ChatMessage]
    model: str = DeepSeekConfig.MODEL
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

class EmotionAnalysisRequest(BaseModel):
    """情绪分析请求"""
    text: str
    context: Optional[str] = None

class DeepSeekService:
    """DeepSeek API服务类"""

    def __init__(self):
        self.api_key = DeepSeekConfig.get_api_key()
        self.base_url = DeepSeekConfig.BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def chat_completion(self, messages: List[Dict], temperature: float = 0.7) -> Dict:
        """
        对话生成
        """
        try:
            # 转换消息格式
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })

            payload = {
                "model": DeepSeekConfig.MODEL,
                "messages": formatted_messages,
                "temperature": temperature,
                "max_tokens": 1000,
                "stream": False
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )

                if response.status_code != 200:
                    raise Exception(f"API request failed: {response.status_code} - {response.text}")

                result = response.json()

                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"DeepSeek API调用失败: {str(e)}",
                "fallback_response": self._get_fallback_response()
            }

    async def analyze_emotion_from_text(self, text: str, context: Optional[str] = None) -> Dict:
        """
        基于文本分析情绪
        """
        try:
            system_prompt = """你是一个专业的情绪分析助手。请分析用户提供的文本，识别主要情绪状态。

请以JSON格式返回结果，包含以下字段：
- primary_emotion: 主要情绪 (happy, sad, angry, surprised, neutral, anxious, excited, frustrated)
- confidence: 置信度 (0.0-1.0)
- intensity: 情绪强度 (0.0-1.0)
- reasoning: 分析理由
- suggestions: 建议 (数组，2-3个建议)

只返回JSON，不要其他内容。"""

            user_prompt = f"请分析这段文字的情绪：\n\n{text}"
            if context:
                user_prompt += f"\n\n上下文：{context}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            result = await self.chat_completion(messages, temperature=0.3)

            if result["success"]:
                try:
                    # 解析JSON响应
                    analysis = json.loads(result["response"])
                    return {
                        "success": True,
                        "analysis": analysis,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except json.JSONDecodeError:
                    # 如果JSON解析失败，返回结构化数据
                    return {
                        "success": True,
                        "analysis": {
                            "primary_emotion": "neutral",
                            "confidence": 0.5,
                            "intensity": 0.5,
                            "reasoning": "文本分析完成，但响应格式异常",
                            "suggestions": ["建议您详细描述当前感受", "可以尝试记录每日情绪日记"]
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }
            else:
                return result

        except Exception as e:
            return {
                "success": False,
                "error": f"情绪分析失败: {str(e)}",
                "analysis": {
                    "primary_emotion": "neutral",
                    "confidence": 0.3,
                    "intensity": 0.3,
                    "reasoning": "分析服务暂时不可用",
                    "suggestions": ["请稍后重试", "可以尝试描述您的感受"]
                }
            }

    async def generate_social_response(self, user_input: str, scenario: str, emotion_context: Optional[str] = None) -> Dict:
        """
        生成社交场景回复
        """
        try:
            system_prompt = f"""你是一个专业的社交技能教练。现在的场景是：{scenario}

你的任务是：
1. 以自然、鼓励的方式回复用户
2. 提供建设性的反馈
3. 给出改进建议
4. 保持积极支持的态度

请以JSON格式返回：
- response: 你的回复内容
- feedback: 反馈意见
- suggestions: 改进建议 (数组)
- score: 表现评分 (0-100)"""

            user_prompt = f"用户输入：{user_input}"
            if emotion_context:
                user_prompt += f"\n当前情绪状态：{emotion_context}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            result = await self.chat_completion(messages, temperature=0.8)

            if result["success"]:
                try:
                    analysis = json.loads(result["response"])
                    return {
                        "success": True,
                        "response": analysis.get("response", "很好，请继续练习！"),
                        "feedback": analysis.get("feedback", "表现不错"),
                        "suggestions": analysis.get("suggestions", ["保持练习"]),
                        "score": analysis.get("score", 75),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "response": result["response"],
                        "feedback": "回复生成成功",
                        "suggestions": ["继续保持这种交流方式"],
                        "score": 80,
                        "timestamp": datetime.utcnow().isoformat()
                    }
            else:
                return result

        except Exception as e:
            return {
                "success": False,
                "error": f"社交回复生成失败: {str(e)}",
                "response": "很抱歉，暂时无法生成回复。请稍后重试。",
                "feedback": "服务暂时不可用",
                "suggestions": ["请稍后重试"],
                "score": 50
            }

    def _get_fallback_response(self) -> str:
        """获取后备回复"""
        fallbacks = [
            "我理解你的感受。请继续分享你的想法。",
            "谢谢你与我分享这些。你的感受是很重要的。",
            "我在这里倾听。请告诉我更多关于你的情况。",
            "你的分享对我很有帮助。让我们继续这个对话。"
        ]
        import random
        return random.choice(fallbacks)

# 全局服务实例
deepseek_service = DeepSeekService()
