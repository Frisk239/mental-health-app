from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import time

router = APIRouter()

class EmotionData(BaseModel):
    happy: float
    sad: float
    angry: float
    surprised: float
    neutral: float
    timestamp: float

class EmotionAnalysisRequest(BaseModel):
    face_image: str  # base64 encoded image
    audio_data: str = None  # base64 encoded audio
    text_input: str = None

@router.post("/analyze", response_model=Dict)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """
    多模态情绪分析接口
    暂时返回模拟数据，后续集成真实AI模型
    """
    try:
        # 模拟AI处理时间
        time.sleep(0.5)

        # 模拟情绪分析结果
        emotions = {
            "happy": 0.7,
            "sad": 0.1,
            "angry": 0.05,
            "surprised": 0.1,
            "neutral": 0.05
        }

        # 确定主要情绪
        primary_emotion = max(emotions.items(), key=lambda x: x[1])

        # 生成个性化建议
        suggestions = {
            "happy": "太好了！保持这种积极的状态。建议分享您的快乐给身边的人。",
            "sad": "我理解您现在的心情。建议深呼吸几次，或写下您的感受。",
            "angry": "愤怒是正常的反应。建议找个安静的地方冷静一下。",
            "surprised": "惊喜的感觉很棒！这通常伴随着好奇心。",
            "neutral": "平静的状态有助于专注。建议继续保持。"
        }

        return {
            "success": True,
            "emotions": emotions,
            "primary_emotion": primary_emotion[0],
            "confidence": primary_emotion[1],
            "suggestion": suggestions.get(primary_emotion[0], "继续关注您的情绪变化。"),
            "timestamp": time.time()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"情绪分析失败: {str(e)}")

@router.get("/history", response_model=List[Dict])
async def get_emotion_history(user_id: str = "default", days: int = 7):
    """
    获取用户情绪历史记录
    """
    # 模拟历史数据
    import random
    from datetime import datetime, timedelta

    history = []
    base_date = datetime.now()

    for i in range(days):
        date = base_date - timedelta(days=i)
        emotions = {
            "happy": random.uniform(0.1, 0.9),
            "sad": random.uniform(0.0, 0.3),
            "angry": random.uniform(0.0, 0.2),
            "surprised": random.uniform(0.0, 0.4),
            "neutral": random.uniform(0.1, 0.6)
        }
        # 归一化
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}

        history.append({
            "date": date.strftime("%Y-%m-%d"),
            "emotions": emotions,
            "primary_emotion": max(emotions.items(), key=lambda x: x[1])[0]
        })

    return history
