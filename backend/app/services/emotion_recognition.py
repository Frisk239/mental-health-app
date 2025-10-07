"""
面部表情识别服务
使用OpenCV进行面部检测，ViT模型进行表情分类
"""

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import logging
import os
from typing import Dict, List, Tuple, Optional
import json
import random

logger = logging.getLogger(__name__)

class EmotionRecognitionService:
    """
    面部表情识别服务类
    整合OpenCV面部检测和ViT深度学习模型
    """

    def __init__(self, model_path: str = None):
        """
        初始化表情识别服务

        Args:
            model_path: ViT模型路径，如果为None则使用本地模型路径
        """
        # 使用本地模型绝对路径
        if model_path is None:
            # 从当前文件位置向上查找项目根目录
            current_file = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file)  # backend/app/services
            current_dir = os.path.dirname(current_dir)  # backend/app
            current_dir = os.path.dirname(current_dir)  # backend
            project_root = os.path.dirname(current_dir)  # 项目根目录
            self.model_path = os.path.join(project_root, "facial_emotions_image_detection", "checkpoint-15740")
        else:
            self.model_path = model_path
        self.processor = None
        self.model = None
        self.face_cascade = None
        self.is_initialized = False

        # 情绪标签映射
        self.emotion_labels = {
            0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }

        # 中文情绪标签映射
        self.emotion_labels_chinese = {
            'anger': '愤怒',
            'disgust': '厌恶',
            'fear': '恐惧',
            'happy': '开心',
            'neutral': '平静',
            'sad': '悲伤',
            'surprise': '惊讶'
        }

    async def initialize(self) -> bool:
        """
        初始化模型和检测器

        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("🚀 开始初始化表情识别服务...")

            # 初始化OpenCV面部检测器
            logger.info("📷 加载OpenCV面部检测器...")
            # 使用内置的Haar Cascade分类器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                logger.error("❌ OpenCV面部检测器加载失败")
                return False

            logger.info("✅ OpenCV面部检测器加载成功")

            # 初始化ViT模型
            logger.info("🤖 加载ViT表情识别模型...")
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            self.model = ViTForImageClassification.from_pretrained(self.model_path)

            # 新模型已有正确的标签映射，无需手动设置
            logger.info(f"📋 模型标签映射: {self.model.config.id2label}")

            # 移动模型到GPU（如果可用）
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("🚀 模型已移动到GPU")
            else:
                logger.info("💻 使用CPU进行推理")

            self.is_initialized = True
            logger.info("✅ 表情识别服务初始化完成")

            return True

        except Exception as e:
            logger.error(f"❌ 表情识别服务初始化失败: {e}")
            return False

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的面部

        Args:
            frame: 输入图像帧 (BGR格式)

        Returns:
            List[Tuple[int, int, int, int]]: 面部边界框列表 (x, y, w, h)
        """
        if not self.is_initialized or self.face_cascade is None:
            logger.error("❌ 表情识别服务未初始化")
            return []

        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 检测面部
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,    # 搜索比例
                minNeighbors=5,     # 最小邻居数
                minSize=(30, 30),   # 最小面部大小
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            logger.info(f"📊 检测到 {len(faces)} 个人脸")

            # 确保 faces 是正确的类型再转换
            if hasattr(faces, 'tolist'):
                return faces.tolist()
            elif isinstance(faces, (list, tuple)):
                return list(faces)
            else:
                logger.warning(f"⚠️ 意外的面部检测结果类型: {type(faces)}")
                return []

        except Exception as e:
            logger.error(f"❌ 面部检测失败: {e}")
            return []

    def preprocess_face(self, face_img: np.ndarray) -> Image.Image:
        """
        预处理面部图像以适配ViT模型

        Args:
            face_img: 面部图像 (BGR格式)

        Returns:
            PIL.Image: 预处理后的图像
        """
        try:
            # 确保图像是正方形
            height, width = face_img.shape[:2]
            if width != height:
                # 计算正方形边界框
                size = min(width, height)
                x = (width - size) // 2
                y = (height - size) // 2
                face_img = face_img[y:y+size, x:x+size]

            # 调整大小为224x224 (ViT模型输入)
            face_resized = cv2.resize(face_img, (224, 224))

            # 转换为RGB格式
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

            # 转换为PIL Image
            pil_image = Image.fromarray(face_rgb)

            return pil_image

        except Exception as e:
            logger.error(f"❌ 面部图像预处理失败: {e}")
            # 返回一个空白图像作为后备
            return Image.new('RGB', (224, 224), color='black')

    def predict_emotion(self, face_image: Image.Image) -> Dict:
        """
        使用ViT模型预测表情

        Args:
            face_image: 预处理后的面部图像

        Returns:
            Dict: 包含情绪预测结果的字典
        """
        if not self.is_initialized or self.processor is None or self.model is None:
            logger.error("❌ ViT模型未初始化")
            return self._get_default_result()

        try:
            # 预处理图像
            inputs = self.processor(images=face_image, return_tensors="pt")

            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 获取预测结果
            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            probabilities = torch.softmax(logits, dim=1)[0]

            # 构建结果
            emotion_english = self.emotion_labels[predicted_class]
            emotion_chinese = self.emotion_labels_chinese[emotion_english]
            confidence = probabilities[predicted_class].item()

            probabilities_dict = {
                label: prob.item()
                for label, prob in zip(self.emotion_labels.values(), probabilities)
            }

            result = {
                "emotion": emotion_english,
                "emotion_chinese": emotion_chinese,
                "confidence": confidence,
                "probabilities": probabilities_dict,
                "timestamp": random.randint(1000000, 9999999)
            }

            logger.info(f"😊 表情识别结果: {emotion_chinese} ({confidence:.3f})")
            logger.info(f"📊 详细概率: {probabilities_dict}")
            return result

        except Exception as e:
            logger.error(f"❌ 表情预测失败: {e}")
            return self._get_default_result()

    def _get_default_result(self) -> Dict:
        """
        获取默认的预测结果（用于错误情况）

        Returns:
            Dict: 默认结果
        """
        return {
            "emotion": "neutral",
            "emotion_chinese": "平静",
            "confidence": 0.5,
            "probabilities": {
                "anger": 0.1,
                "disgust": 0.1,
                "fear": 0.1,
                "happy": 0.2,
                "neutral": 0.3,
                "sad": 0.1,
                "surprise": 0.1
            },
            "timestamp": random.randint(1000000, 9999999)
        }

    async def process_frame(self, frame_data: bytes) -> Dict:
        """
        处理视频帧并返回表情识别结果

        Args:
            frame_data: JPEG格式的图像字节数据

        Returns:
            Dict: 表情识别结果
        """
        try:
            logger.info(f"📹 开始处理视频帧，大小: {len(frame_data)} bytes")

            # 解码图像
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error("❌ 图像解码失败")
                return self._get_default_result()

            logger.info(f"✅ 图像解码成功，尺寸: {frame.shape}")

            # 检测面部
            faces = self.detect_faces(frame)

            if len(faces) == 0:
                logger.info("👤 未检测到面部")
                return {
                    "emotion": "neutral",
                    "emotion_chinese": "平静",
                    "confidence": 0.0,
                    "message": "未检测到面部",
                    "timestamp": random.randint(1000000, 9999999)
                }

            logger.info(f"🎯 检测到 {len(faces)} 个人脸，开始处理第一个")

            # 处理第一个检测到的人脸
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]

            logger.info(f"✂️ 裁剪面部区域: x={x}, y={y}, w={w}, h={h}")

            # 预处理面部图像
            processed_face = self.preprocess_face(face_img)

            # 预测表情
            result = self.predict_emotion(processed_face)

            # 添加面部框信息
            result["face_box"] = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            result["faces_count"] = len(faces)

            logger.info(f"🎉 帧处理完成，返回结果: {result.get('emotion_chinese', '未知')}")
            return result

        except Exception as e:
            logger.error(f"❌ 帧处理失败: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_result()

    def get_model_info(self) -> Dict:
        """
        获取模型信息

        Returns:
            Dict: 模型信息
        """
        return {
            "model_name": "ViT-Large Facial Emotion Recognition (checkpoint-15740)",
            "model_path": self.model_path,
            "supported_emotions": list(self.emotion_labels_chinese.values()),
            "input_size": "224x224",
            "accuracy": "预计80%+",
            "is_initialized": self.is_initialized,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

# 全局服务实例
emotion_service = EmotionRecognitionService()
