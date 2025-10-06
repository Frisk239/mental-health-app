u"""
表情识别WebSocket路由
处理实时视频流并返回表情识别结果
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
import asyncio
from typing import Dict, List
import numpy as np
import cv2
from app.services.emotion_recognition import emotion_service

router = APIRouter()

# 设置日志
logger = logging.getLogger(__name__)

# 存储活跃的WebSocket连接
active_connections: Dict[str, WebSocket] = {}

@router.websocket("/ws/emotion")
async def emotion_detection_websocket(websocket: WebSocket):
    """
    表情识别WebSocket端点
    接收前端视频帧，返回实时表情识别结果
    """
    client_id = f"client_{id(websocket)}"
    await websocket.accept()

    # 添加到活跃连接
    active_connections[client_id] = websocket
    logger.info(f"🔗 表情识别客户端连接: {client_id} (活跃连接数: {len(active_connections)})")

    try:
        # 初始化表情识别服务
        if not emotion_service.is_initialized:
            logger.info("🔧 初始化表情识别服务...")
            success = await emotion_service.initialize()
            if not success:
                await websocket.send_json({
                    "error": "表情识别服务初始化失败",
                    "type": "init_failed"
                })
                return

        while True:
            try:
                # 接收前端发送的视频帧数据
                frame_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)

                logger.debug(f"📹 接收到视频帧数据，大小: {len(frame_data)} bytes")

                # 处理视频帧
                result = await emotion_service.process_frame(frame_data)

                # 发送结果回前端
                await websocket.send_json(result)

                logger.debug(f"📤 发送表情识别结果: {result.get('emotion_chinese', '未知')}")

            except asyncio.TimeoutError:
                # 发送心跳包保持连接
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": asyncio.get_event_loop().time()
                })
                continue

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON解析失败: {e}")
                await websocket.send_json({
                    "error": "数据格式错误",
                    "type": "json_error"
                })

            except Exception as e:
                logger.error(f"❌ 处理视频帧时出错: {e}")
                await websocket.send_json({
                    "error": f"处理失败: {str(e)}",
                    "type": "process_error"
                })

    except WebSocketDisconnect:
        logger.info(f"🔌 表情识别客户端断开连接: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket连接异常: {e}")
    finally:
        # 从活跃连接中移除
        if client_id in active_connections:
            del active_connections[client_id]
        logger.info(f"🏁 清理连接: {client_id} (剩余连接数: {len(active_connections)})")

@router.get("/emotion/status")
async def get_emotion_service_status():
    """
    获取表情识别服务状态
    """
    return {
        "is_initialized": emotion_service.is_initialized,
        "model_info": emotion_service.get_model_info(),
        "active_connections": len(active_connections),
        "supported_emotions": list(emotion_service.emotion_labels_chinese.values())
    }

@router.post("/emotion/test")
async def test_emotion_detection():
    """
    测试表情识别功能
    """
    try:
        # 创建一个测试图像 (224x224的灰色图像)
        test_image = np.ones((224, 224, 3), dtype=np.uint8) * 128

        # 转换为字节数据
        _, encoded_img = cv2.imencode('.jpg', test_image)
        frame_data = encoded_img.tobytes()

        # 处理测试图像
        result = await emotion_service.process_frame(frame_data)

        return {
            "success": True,
            "result": result,
            "message": "表情识别测试成功"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "表情识别测试失败"
        }

@router.get("/emotion/models")
async def list_emotion_models():
    """
    列出可用的表情识别模型
    """
    return {
        "current_model": emotion_service.get_model_info(),
        "available_models": [
            {
                "name": "BEiT-Large",
                "description": "基于BEiT架构的表情识别模型",
                "accuracy": "76.2%",
                "size": "~1.2GB",
                "input_size": "224x224"
            }
        ]
    }
