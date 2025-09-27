from fastapi import APIRouter
from typing import Dict, List
import random
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/summary", response_model=Dict)
async def get_history_summary(user_id: str = "default", days: int = 30):
    """获取历史数据摘要"""
    # 模拟历史摘要数据
    return {
        "total_sessions": random.randint(50, 200),
        "average_mood": round(random.uniform(6.5, 8.5), 1),
        "most_common_emotion": random.choice(["happy", "neutral", "sad"]),
        "improvement_trend": random.choice(["improving", "stable", "needs_attention"]),
        "farm_crops_planted": random.randint(10, 50),
        "social_sessions_completed": random.randint(5, 25),
        "period": f"{days}天"
    }

@router.get("/timeline", response_model=List[Dict])
async def get_timeline(user_id: str = "default", limit: int = 20):
    """获取时间线数据"""
    timeline = []
    base_date = datetime.now()

    activities = [
        "情绪监测", "农场种植", "社交练习", "日志记录",
        "冥想练习", "呼吸训练", "情绪分析"
    ]

    emotions = ["happy", "sad", "angry", "neutral", "surprised"]

    for i in range(limit):
        date = base_date - timedelta(hours=i*2)
        activity = random.choice(activities)
        emotion = random.choice(emotions)

        timeline.append({
            "id": f"activity_{i}",
            "timestamp": date.isoformat(),
            "activity": activity,
            "emotion": emotion,
            "description": f"完成了{activity}活动",
            "impact": random.choice(["正面", "中性", "需要关注"])
        })

    return timeline

@router.get("/trends", response_model=Dict)
async def get_trends(user_id: str = "default", period: str = "week"):
    """获取趋势分析数据"""
    # 模拟趋势数据
    periods = {
        "week": 7,
        "month": 30,
        "quarter": 90
    }

    days = periods.get(period, 7)

    mood_trend = []
    activity_trend = []

    for i in range(days):
        mood_trend.append({
            "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
            "average_mood": round(random.uniform(5.0, 9.0), 1),
            "sessions_count": random.randint(1, 5)
        })

        activity_trend.append({
            "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
            "farm_activities": random.randint(0, 3),
            "social_activities": random.randint(0, 2),
            "meditation_sessions": random.randint(0, 2)
        })

    return {
        "mood_trend": mood_trend,
        "activity_trend": activity_trend,
        "insights": [
            "您的情绪稳定性有所提高",
            "建议增加社交练习频率",
            "农场活动对情绪有积极影响"
        ]
    }
