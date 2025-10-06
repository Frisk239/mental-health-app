"""
社交实验室路由
提供场景管理、对话练习、反馈分析等API
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import json
import logging
from typing import Dict, List
from datetime import datetime

from app.services.social_lab_service import social_lab_service

router = APIRouter()

# 设置日志
logger = logging.getLogger(__name__)

# 存储活跃的对话会话
active_sessions: Dict[str, Dict] = {}

@router.on_event("startup")
async def startup_event():
    """应用启动时初始化社交实验室服务"""
    success = await social_lab_service.initialize()
    if not success:
        logger.error("❌ 社交实验室服务初始化失败")
    else:
        logger.info("✅ 社交实验室服务初始化成功")

@router.get("/scenarios")
async def get_scenarios():
    """获取可用的练习场景"""
    try:
        scenarios = await social_lab_service.get_available_scenarios()
        return {"scenarios": scenarios}
    except Exception as e:
        logger.error(f"❌ 获取场景列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取场景列表失败")

@router.post("/sessions/start")
async def start_session(request: Dict):
    """开始新的练习会话"""
    try:
        scenario_id = request.get('scenario_id')

        if not scenario_id:
            raise HTTPException(status_code=400, detail="缺少场景ID")

        session = await social_lab_service.start_practice_session(scenario_id)

        if not session:
            raise HTTPException(status_code=404, detail="场景不存在或不可用")

        # 存储活跃会话
        session_key = f"session_{session['session_id']}"
        active_sessions[session_key] = {
            'session_id': session['session_id'],
            'scenario_id': scenario_id,
            'start_time': datetime.now(),
            'dialogue_history': session.get('dialogue_history', [])
        }

        logger.info(f"🎯 开始练习会话: {session['session_id']} - 场景: {scenario_id}")
        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 开始会话失败: {e}")
        raise HTTPException(status_code=500, detail="开始会话失败")

@router.websocket("/sessions/{session_id}/chat")
async def chat_websocket(websocket: WebSocket, session_id: int):
    """实时对话WebSocket"""
    await websocket.accept()

    session_key = f"session_{session_id}"
    logger.info(f"🔗 对话WebSocket连接: {session_id}")

    try:
        while True:
            # 接收用户消息
            data = await websocket.receive_json()

            user_message = data.get('message', '')
            voice_emotions = data.get('voice_emotions')
            face_emotions = data.get('face_emotions')

            if not user_message:
                await websocket.send_json({'error': '消息不能为空'})
                continue

            # 检查会话是否存在
            if session_key not in active_sessions:
                await websocket.send_json({'error': '会话不存在或已结束'})
                continue

            # 记录用户消息到对话历史
            active_sessions[session_key]['dialogue_history'].append({
                'role': 'user',
                'message': user_message,
                'timestamp': datetime.now().isoformat(),
                'voice_emotions': voice_emotions,
                'face_emotions': face_emotions
            })

            # 生成AI回复
            ai_response = await social_lab_service.generate_ai_response(
                session_id, user_message, voice_emotions, face_emotions
            )

            if 'error' in ai_response:
                await websocket.send_json({'error': ai_response['error']})
                continue

            # 记录AI回复到对话历史
            active_sessions[session_key]['dialogue_history'].append({
                'role': 'assistant',
                'message': ai_response['response'],
                'timestamp': ai_response['timestamp']
            })

            # 发送AI回复
            await websocket.send_json({
                'response': ai_response['response'],
                'role': ai_response['role'],
                'timestamp': ai_response['timestamp']
            })

    except WebSocketDisconnect:
        logger.info(f"🔌 对话WebSocket断开: {session_id}")
    except Exception as e:
        logger.error(f"❌ 对话WebSocket错误: {e}")
        try:
            await websocket.send_json({'error': '对话服务异常'})
        except:
            pass

@router.post("/sessions/{session_id}/end")
async def end_session(session_id: int, request: Dict = None):
    """结束练习会话"""
    try:
        session_key = f"session_{session_id}"

        if session_key not in active_sessions:
            raise HTTPException(status_code=404, detail="会话不存在或已结束")

        session_data = active_sessions[session_key]

        # 获取对话历史
        dialogue_history = session_data.get('dialogue_history', [])

        # 获取情绪数据（从最后几条消息中提取）
        voice_emotions = None
        face_emotions = None

        for message in reversed(dialogue_history):
            if message.get('voice_emotions'):
                voice_emotions = message['voice_emotions']
                break

        for message in reversed(dialogue_history):
            if message.get('face_emotions'):
                face_emotions = message['face_emotions']
                break

        # 结束会话并生成反馈
        feedback = await social_lab_service.end_practice_session(
            session_id, dialogue_history, voice_emotions, face_emotions
        )

        # 清理活跃会话
        del active_sessions[session_key]

        logger.info(f"🏁 结束练习会话: {session_id}")
        return {
            'session_id': session_id,
            'feedback': feedback,
            'dialogue_count': len(dialogue_history)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 结束会话失败: {e}")
        raise HTTPException(status_code=500, detail="结束会话失败")

@router.get("/history")
async def get_session_history(user_id: int = 1, limit: int = 10):
    """获取会话历史"""
    try:
        history = await social_lab_service.get_session_history(user_id, limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"❌ 获取会话历史失败: {e}")
        raise HTTPException(status_code=500, detail="获取历史失败")

@router.get("/stats")
async def get_social_lab_stats():
    """获取社交实验室统计信息"""
    try:
        from app.models.database import db_manager

        # 总用户数
        total_users = db_manager.execute_query("SELECT COUNT(*) as count FROM users")[0]['count']

        # 总会话数
        total_sessions = db_manager.execute_query("SELECT COUNT(*) as count FROM practice_sessions")[0]['count']

        # 平均得分
        avg_score = db_manager.execute_query("""
            SELECT AVG(total_score) as avg FROM practice_sessions
            WHERE total_score IS NOT NULL
        """)[0]['avg'] or 0

        # 场景使用统计
        scenario_stats = db_manager.execute_query("""
            SELECT s.name, COUNT(ps.id) as count
            FROM scenarios s
            LEFT JOIN practice_sessions ps ON s.id = ps.scenario_id
            GROUP BY s.id, s.name
            ORDER BY count DESC
        """)

        return {
            'total_users': total_users,
            'total_sessions': total_sessions,
            'average_score': round(avg_score, 1),
            'scenario_stats': [dict(stat) for stat in scenario_stats],
            'active_sessions': len(active_sessions)
        }

    except Exception as e:
        logger.error(f"❌ 获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取统计失败")
