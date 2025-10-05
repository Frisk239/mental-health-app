"""
语音服务API路由
提供语音识别、合成和交互的HTTP接口
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import logging
from typing import Dict, Optional

from app.services.voice_service import voice_service

router = APIRouter()

# 设置日志
logger = logging.getLogger(__name__)

@router.on_event("startup")
async def startup_event():
    """应用启动时初始化语音服务"""
    success = await voice_service.initialize()
    if not success:
        logger.error("❌ 语音服务初始化失败")
    else:
        logger.info("✅ 语音服务初始化成功")

@router.post("/process-input")
async def process_input(
    input_data: str = Form(...),
    input_type: str = Form("text")
):
    """
    处理用户输入（文本或语音）

    Args:
        input_data: 输入数据（文本或音频文件路径）
        input_type: 输入类型 ("text" 或 "voice")

    Returns:
        处理后的文本和输入类型
    """
    try:
        if input_type == "voice":
            # 语音输入：需要上传音频文件
            raise HTTPException(
                status_code=400,
                detail="语音输入需要上传音频文件，请使用 /process-voice-input 接口"
            )
        else:
            # 文本输入
            processed_text, actual_type = await voice_service.process_input(
                input_data, "text"
            )
            return {
                "processed_text": processed_text,
                "input_type": actual_type,
                "success": True
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 输入处理失败: {e}")
        raise HTTPException(status_code=500, detail="输入处理失败")

@router.post("/process-voice-input")
async def process_voice_input(
    audio_file: UploadFile = File(...),
    enable_stt: bool = Form(True)
):
    """
    处理语音输入

    Args:
        audio_file: 音频文件
        enable_stt: 是否启用语音识别

    Returns:
        识别出的文本
    """
    try:
        if not enable_stt or not voice_service.stt_service:
            raise HTTPException(
                status_code=400,
                detail="语音识别服务未配置或已禁用"
            )

        # 读取音频文件内容
        audio_data = await audio_file.read()

        if not audio_data:
            raise HTTPException(status_code=400, detail="音频文件为空")

        # 处理语音输入
        processed_text, actual_type = await voice_service.process_input(
            audio_data, "voice"
        )

        return {
            "processed_text": processed_text,
            "input_type": actual_type,
            "success": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 语音输入处理失败: {e}")
        raise HTTPException(status_code=500, detail="语音输入处理失败")

@router.post("/generate-response")
async def generate_response(
    text: str = Form(...),
    role_name: str = Form(...),
    enable_voice: bool = Form(True),
    input_mode: str = Form("text")
):
    """
    生成AI回复（支持语音合成）

    Args:
        text: 用户输入文本
        role_name: AI角色名称
        enable_voice: 是否启用语音合成
        input_mode: 输入模式

    Returns:
        AI回复（文本+语音）
    """
    try:
        # 切换输入模式
        voice_service.switch_input_mode(input_mode)

        # 生成回复
        response = await voice_service.generate_response(
            text, role_name, enable_voice
        )

        if not response.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=response.get("error", "生成回复失败")
            )

        return {
            "text": response.get("text"),
            "audio": response.get("audio"),  # 语音数据（如果启用）
            "role": response.get("role"),
            "input_mode": response.get("input_mode"),
            "success": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 生成回复失败: {e}")
        raise HTTPException(status_code=500, detail="生成回复失败")

@router.get("/roles")
async def get_available_roles():
    """获取可用角色列表"""
    try:
        roles = voice_service.get_available_roles()
        return {
            "roles": roles,
            "total": len(roles),
            "voice_enabled": voice_service.tts_service is not None
        }
    except Exception as e:
        logger.error(f"❌ 获取角色列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取角色列表失败")

@router.get("/scene-roles/{scenario_id}")
async def get_scene_roles(scenario_id: str):
    """获取指定场景的可用角色"""
    try:
        config = voice_service.config
        scene_config = config.get("scene_voice_mapping", {}).get(scenario_id)

        if not scene_config:
            raise HTTPException(status_code=404, detail="场景不存在")

        available_roles = []
        for role_name in scene_config.get("available_roles", []):
            role_config = config.get("role_voice_mapping", {}).get(role_name)
            if role_config:
                available_roles.append({
                    "name": role_name,
                    "description": role_config.get("description", ""),
                    "voice_enabled": voice_service.tts_service is not None,
                    "gpt_model": role_config.get("gpt_model"),
                    "sovits_model": role_config.get("sovits_model")
                })

        return {
            "scene_id": scenario_id,
            "default_role": scene_config.get("default_role"),
            "available_roles": available_roles,
            "description": scene_config.get("description")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 获取场景角色失败: {e}")
        raise HTTPException(status_code=500, detail="获取场景角色失败")

@router.post("/scene-voice-config")
async def update_scene_voice_config(request: Dict):
    """更新场景音色配置"""
    try:
        scene_id = request.get("scene_id")
        role_name = request.get("role_name")

        if not scene_id or not role_name:
            raise HTTPException(status_code=400, detail="缺少场景ID或角色名称")

        # 更新配置文件
        config = voice_service.config
        if "scene_voice_mapping" not in config:
            config["scene_voice_mapping"] = {}

        if scene_id not in config["scene_voice_mapping"]:
            config["scene_voice_mapping"][scene_id] = {}

        config["scene_voice_mapping"][scene_id]["default_role"] = role_name

        # 保存配置到文件
        with open(voice_service.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 重新加载配置
        voice_service.config = config

        logger.info(f"✅ 更新场景 {scene_id} 音色配置为: {role_name}")
        return {
            "success": True,
            "scene_id": scene_id,
            "role_name": role_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 更新场景音色配置失败: {e}")
        raise HTTPException(status_code=500, detail="更新配置失败")

@router.get("/input-modes")
async def get_input_modes():
    """获取输入模式状态"""
    try:
        modes = voice_service.get_input_modes()
        return modes
    except Exception as e:
        logger.error(f"❌ 获取输入模式失败: {e}")
        raise HTTPException(status_code=500, detail="获取输入模式失败")

@router.post("/switch-mode")
async def switch_input_mode(mode: str = Form(...)):
    """
    切换输入模式

    Args:
        mode: 目标模式 ("text" 或 "voice")

    Returns:
        切换结果
    """
    try:
        success = voice_service.switch_input_mode(mode)

        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的输入模式: {mode}"
            )

        return {
            "current_mode": voice_service.input_mode,
            "success": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 切换输入模式失败: {e}")
        raise HTTPException(status_code=500, detail="切换输入模式失败")

@router.get("/health")
async def health_check():
    """语音服务健康检查"""
    try:
        health = await voice_service.health_check()
        return health
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")

@router.get("/stt-health")
async def stt_health_check():
    """百度STT服务健康检查"""
    try:
        if voice_service.stt_service:
            health = await voice_service.stt_service.health_check()
            return health
        else:
            return {
                "service": "baidu_stt",
                "configured": False,
                "error": "STT服务未初始化"
            }
    except Exception as e:
        logger.error(f"❌ STT健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="STT健康检查失败")

@router.get("/tts-health")
async def tts_health_check():
    """GPT-SoVITS服务健康检查"""
    try:
        if voice_service.tts_service:
            health = await voice_service.tts_service.health_check()
            return health
        else:
            return {
                "service": "gpt_sovits",
                "configured": False,
                "error": "TTS服务未初始化"
            }
    except Exception as e:
        logger.error(f"❌ TTS健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="TTS健康检查失败")

@router.post("/synthesize-speech")
async def synthesize_speech(
    text: str = Form(...),
    role_name: str = Form(...),
    speed: float = Form(1.0),
    pitch_shift: float = Form(1.0),
    emotion_intensity: float = Form(0.7)
):
    """
    语音合成（单独接口）

    Args:
        text: 要合成的文本
        role_name: 角色名称
        speed: 语速 (0.5-2.0)
        pitch_shift: 音调偏移 (0.8-1.2)
        emotion_intensity: 情感强度 (0.0-1.0)

    Returns:
        合成的音频数据
    """
    try:
        if not voice_service.tts_service:
            raise HTTPException(
                status_code=400,
                detail="语音合成服务未配置"
            )

        # 构建情绪参数
        emotion_params = {
            "speed": speed,
            "pitch_shift": pitch_shift,
            "emotion_intensity": emotion_intensity
        }

        # 生成语音
        audio_data = await voice_service.tts_service.synthesize_speech(
            text, role_name, emotion_params
        )

        if not audio_data:
            raise HTTPException(
                status_code=500,
                detail="语音合成失败"
            )

        # 返回音频数据流
        from fastapi.responses import StreamingResponse
        import io
        from urllib.parse import quote

        audio_stream = io.BytesIO(audio_data)

        # 对中文文件名进行URL编码
        safe_filename = quote(f"{role_name}_{len(text)}.wav")

        return StreamingResponse(
            audio_stream,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 语音合成失败: {e}")
        raise HTTPException(status_code=500, detail="语音合成失败")

@router.get("/config")
async def get_voice_config():
    """获取语音服务配置"""
    try:
        return {
            "config": voice_service.config,
            "input_mode": voice_service.input_mode,
            "available_roles": voice_service.get_available_roles(),
            "input_modes": voice_service.get_input_modes()
        }
    except Exception as e:
        logger.error(f"❌ 获取配置失败: {e}")
        raise HTTPException(status_code=500, detail="获取配置失败")
