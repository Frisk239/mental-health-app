"""
调试日志WebSocket路由
接收前端发送的调试信息并输出到服务器控制台
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
from datetime import datetime
from typing import Dict, Any

router = APIRouter()

# 设置日志
logger = logging.getLogger("debug_ws")
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_handler.setFormatter(formatter)

# 添加处理器到logger
logger.addHandler(console_handler)

# 存储活跃连接
active_connections: Dict[str, WebSocket] = {}

@router.websocket("/ws/debug")
async def debug_websocket(websocket: WebSocket):
    """
    调试日志WebSocket端点
    接收前端发送的调试信息并输出到服务器控制台
    """
    client_id = f"client_{id(websocket)}"
    await websocket.accept()

    # 添加到活跃连接
    active_connections[client_id] = websocket
    logger.info(f"🔗 调试客户端连接: {client_id} (活跃连接数: {len(active_connections)})")

    try:
        while True:
            # 接收前端发送的数据
            data = await websocket.receive_text()

            try:
                # 解析JSON数据
                log_entry: Dict[str, Any] = json.loads(data)

                # 提取日志信息
                timestamp = log_entry.get('timestamp', datetime.utcnow().isoformat())
                level = log_entry.get('level', 'info').upper()
                message = log_entry.get('message', 'No message')
                source = log_entry.get('source', 'unknown')
                log_data = log_entry.get('data')

                # 格式化日志消息
                formatted_message = f"[{level}] [{source}] {message}"

                # 根据日志级别输出到控制台
                if level == 'ERROR':
                    logger.error(formatted_message)
                    if log_data:
                        logger.error(f"📊 数据: {json.dumps(log_data, indent=2, ensure_ascii=False)}")
                elif level == 'WARN' or level == 'WARNING':
                    logger.warning(formatted_message)
                    if log_data:
                        logger.warning(f"📊 数据: {json.dumps(log_data, indent=2, ensure_ascii=False)}")
                else:
                    logger.info(formatted_message)
                    if log_data:
                        logger.info(f"📊 数据: {json.dumps(log_data, indent=2, ensure_ascii=False)}")

                # 发送确认消息给前端
                await websocket.send_text(json.dumps({
                    "status": "received",
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": f"日志已接收: {message[:50]}..."
                }, ensure_ascii=False))

            except json.JSONDecodeError as e:
                logger.error(f"❌ 解析调试日志失败: {e}")
                logger.error(f"原始数据: {data[:200]}...")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": f"JSON解析失败: {str(e)}"
                }, ensure_ascii=False))

            except Exception as e:
                logger.error(f"❌ 处理调试日志时出错: {e}")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": f"处理失败: {str(e)}"
                }, ensure_ascii=False))

    except WebSocketDisconnect:
        logger.info(f"🔌 调试客户端断开连接: {client_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket连接异常: {e}")
    finally:
        # 从活跃连接中移除
        if client_id in active_connections:
            del active_connections[client_id]
        logger.info(f"🏁 清理连接: {client_id} (剩余连接数: {len(active_connections)})")

@router.get("/debug/connections")
async def get_debug_connections():
    """
    获取当前活跃的调试连接数量
    """
    return {
        "active_connections": len(active_connections),
        "connections": list(active_connections.keys())
    }

@router.post("/debug/broadcast")
async def broadcast_debug_message(message: str, level: str = "info"):
    """
    广播调试消息给所有连接的客户端
    """
    sent_count = 0
    for client_id, websocket in active_connections.items():
        try:
            await websocket.send_text(json.dumps({
                "type": "broadcast",
                "level": level,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }, ensure_ascii=False))
            sent_count += 1
        except Exception as e:
            logger.error(f"❌ 发送广播消息失败 {client_id}: {e}")

    return {
        "message": f"广播消息发送完成",
        "sent_to": sent_count,
        "total_connections": len(active_connections)
    }
