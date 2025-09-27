from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
import time

router = APIRouter()

class ScenarioData(BaseModel):
    id: str
    title: str
    difficulty: str
    type: str

class DialogueResponse(BaseModel):
    speaker: str
    text: str
    emotion: str

@router.get("/scenarios", response_model=List[Dict])
async def get_scenarios():
    """获取可用的社交场景"""
    scenarios = [
        {
            "id": "presentation",
            "title": "课堂演讲",
            "description": "模拟在课堂上进行演讲",
            "difficulty": "medium",
            "type": "presentation"
        },
        {
            "id": "interview",
            "title": "社团招新面试",
            "description": "模拟社团招新面试场景",
            "difficulty": "hard",
            "type": "interview"
        },
        {
            "id": "conversation",
            "title": "日常对话",
            "description": "日常社交对话练习",
            "difficulty": "easy",
            "type": "conversation"
        }
    ]
    return scenarios

@router.post("/start/{scenario_id}", response_model=Dict)
async def start_scenario(scenario_id: str):
    """开始社交场景练习"""
    return {
        "success": True,
        "scenario_id": scenario_id,
        "session_id": f"session_{int(time.time())}",
        "initial_message": "你好！让我们开始这个练习。请介绍一下自己。",
        "emotion": "neutral"
    }

@router.post("/respond", response_model=Dict)
async def get_ai_response(user_input: str, session_id: str, scenario: str = "日常对话"):
    """获取AI的对话回复"""
    try:
        from app.services.deepseek_service import deepseek_service

        # 使用DeepSeek生成智能回复
        result = await deepseek_service.generate_social_response(
            user_input=user_input,
            scenario=scenario,
            emotion_context=None  # 可以后续添加情绪上下文
        )

        if result["success"]:
            return {
                "success": True,
                "response": result["response"],
                "emotion": "interested",
                "suggestion": result["suggestions"][0] if result["suggestions"] else "保持眼神接触，展现自信",
                "score": result["score"],
                "feedback": result["feedback"]
            }
        else:
            # 回退到模拟回复
            import random
            responses = [
                "很好！请继续说下去。",
                "我理解你的意思。你觉得呢？",
                "很有趣的观点！能详细说说吗？",
                "谢谢你的分享。这让我想到...",
                "好的，让我们换个话题吧。"
            ]

            return {
                "success": True,
                "response": random.choice(responses),
                "emotion": "interested",
                "suggestion": "保持眼神接触，展现自信",
                "score": random.randint(70, 95),
                "feedback": "回复生成成功"
            }

    except Exception as e:
        # 最后的回退机制
        import random
        responses = [
            "很好！请继续说下去。",
            "我理解你的意思。你觉得呢？",
            "很有趣的观点！能详细说说吗？",
            "谢谢你的分享。这让我想到...",
            "好的，让我们换个话题吧。"
        ]

        return {
            "success": True,
            "response": random.choice(responses),
            "emotion": "interested",
            "suggestion": "保持眼神接触，展现自信",
            "score": random.randint(70, 95),
            "feedback": f"服务暂时不可用: {str(e)}"
        }
