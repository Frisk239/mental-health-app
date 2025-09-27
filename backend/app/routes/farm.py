from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
import time

router = APIRouter()

class CropData(BaseModel):
    id: str
    type: str
    growth: float
    emotion: str

@router.get("/crops", response_model=List[Dict])
async def get_crops():
    """获取农场作物状态"""
    # 模拟作物数据
    crops = [
        {
            "id": "crop_1",
            "type": "向日葵",
            "growth": 0.8,
            "emotion": "happy",
            "planted_at": time.time() - 86400  # 1天前
        },
        {
            "id": "crop_2",
            "type": "郁金香",
            "growth": 0.6,
            "emotion": "sad",
            "planted_at": time.time() - 43200  # 12小时前
        }
    ]
    return crops

@router.post("/plant", response_model=Dict)
async def plant_crop(crop: CropData):
    """种植新作物"""
    return {
        "success": True,
        "crop": {
            **crop.dict(),
            "planted_at": time.time(),
            "growth": 0.1
        },
        "message": f"成功种植{crop.type}！"
    }

@router.post("/water/{crop_id}", response_model=Dict)
async def water_crop(crop_id: str):
    """浇水作物，促进生长"""
    return {
        "success": True,
        "crop_id": crop_id,
        "growth_increase": 0.1,
        "message": "作物生长加快了！"
    }
