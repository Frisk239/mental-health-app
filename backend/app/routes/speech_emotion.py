"""
语音情绪识别路由
支持实时麦克风录音和音频文件上传
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
import json
import logging
import asyncio
from typing import Dict, List
import io
from app.services.speech_emotion_recognition import speech_emotion_service

router = APIRouter()

# 设置日志
logger = logging.getLogger(__name__)

# 存储活跃的WebSocket连接
active_connections: Dict[str, WebSocket] = {}

@router.websocket("/ws/speech-emotion")
async def speech_emotion_websocket(websocket: WebSocket):
    """
    语音情绪识别WebSocket端点
    接收实时音频流，返回情绪识别结果
    """
    client_id = f"speech_client_{id(websocket)}"
    await websocket.accept()

    # 添加到活跃连接
    active_connections[client_id] = websocket
    logger.info(f"🎤 语音情绪识别客户端连接: {client_id} (活跃连接数: {len(active_connections)})")

    try:
        # 初始化语音情绪识别服务
        if not speech_emotion_service.is_initialized:
            logger.info("🔧 初始化语音情绪识别服务...")
            success = await speech_emotion_service.initialize()
            if not success:
                await websocket.send_json({
                    "error": "语音情绪识别服务初始化失败",
                    "type": "init_failed"
                })
                return

        # 音频数据缓冲区
        audio_buffer = bytearray()
        sample_rate = 16000  # Whisper默认采样率

        while True:
            try:
                # 接收前端发送的音频数据
                audio_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=10.0)

                logger.debug(f"🎵 接收到音频数据，大小: {len(audio_data)} bytes")

                # 累积音频数据
                audio_buffer.extend(audio_data)

                # 当缓冲区达到一定大小或收到特殊标记时进行处理
                # 这里简化处理：每收到数据就立即处理
                if len(audio_buffer) > 0:
                    try:
                        # 处理音频数据
                        result = await speech_emotion_service.process_audio_data(
                            bytes(audio_buffer),
                            sample_rate
                        )

                        # 发送结果回前端
                        await websocket.send_json(result)

                        logger.debug(f"📤 发送语音情绪识别结果: {result.get('emotion_chinese', '未知')}")

                        # 清空缓冲区，准备下一段音频
                        audio_buffer.clear()

                    except Exception as e:
                        logger.error(f"❌ 处理音频数据时出错: {e}")
                        await websocket.send_json({
                            "error": f"处理失败: {str(e)}",
                            "type": "process_error"
                        })

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

    except WebSocketDisconnect:
        logger.info(f"🔌 语音情绪识别客户端断开连接: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket连接异常: {e}")
    finally:
        # 从活跃连接中移除
        if client_id in active_connections:
            del active_connections[client_id]
        logger.info(f"🏁 清理语音连接: {client_id} (剩余连接数: {len(active_connections)})")

@router.post("/speech-emotion/upload")
async def upload_audio_file(file: UploadFile = File(...)):
    """
    上传音频文件进行情绪识别

    Args:
        file: 上传的音频文件

    Returns:
        Dict: 情绪识别结果
    """
    try:
        # 检查文件类型
        allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/x-wav", "audio/wave"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}。支持的类型: WAV, MP3"
            )

        # 检查文件大小 (限制为50MB)
        file_size = 0
        content = await file.read()
        file_size = len(content)

        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=400,
                detail="文件过大，请上传小于50MB的音频文件"
            )

        logger.info(f"📁 接收到音频文件: {file.filename}, 大小: {file_size} bytes, 类型: {file.content_type}")

        # 初始化服务（如果未初始化）
        if not speech_emotion_service.is_initialized:
            logger.info("🔧 初始化语音情绪识别服务...")
            success = await speech_emotion_service.initialize()
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="语音情绪识别服务初始化失败"
                )

        # 处理音频文件
        result = await speech_emotion_service.process_audio_data(content)

        logger.info(f"✅ 音频文件处理完成: {file.filename} -> {result.get('emotion_chinese', '未知')}")

        return {
            "success": True,
            "result": result,
            "file_info": {
                "filename": file.filename,
                "size": file_size,
                "content_type": file.content_type
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 处理上传文件失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"处理文件失败: {str(e)}"
        )

@router.get("/speech-emotion/status")
async def get_speech_emotion_service_status():
    """
    获取语音情绪识别服务状态
    """
    return {
        "is_initialized": speech_emotion_service.is_initialized,
        "model_info": speech_emotion_service.get_model_info(),
        "active_connections": len(active_connections),
        "supported_emotions": list(speech_emotion_service.emotion_labels_chinese.values())
    }

@router.post("/speech-emotion/test")
async def test_speech_emotion():
    """
    测试语音情绪识别功能（使用空音频）
    """
    try:
        # 初始化服务（如果未初始化）
        if not speech_emotion_service.is_initialized:
            logger.info("🔧 初始化语音情绪识别服务...")
            success = await speech_emotion_service.initialize()
            if not success:
                return {
                    "success": False,
                    "error": "语音情绪识别服务初始化失败",
                    "message": "服务初始化失败"
                }

        # 创建一个简单的测试音频（静音）
        sample_rate = 16000
        duration = 1.0  # 1秒
        silent_audio = np.zeros(int(sample_rate * duration), dtype=np.float32)

        # 转换为字节数据
        audio_buffer = io.BytesIO()
        # 这里简化处理，实际应该使用适当的音频编码
        # 为了测试，我们直接传递numpy数组的字节表示
        audio_bytes = silent_audio.tobytes()

        # 处理测试音频
        result = await speech_emotion_service.process_audio_data(audio_bytes, sample_rate)

        return {
            "success": True,
            "result": result,
            "message": "语音情绪识别测试成功"
        }

    except Exception as e:
        logger.error(f"❌ 语音情绪识别测试失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "语音情绪识别测试失败"
        }
