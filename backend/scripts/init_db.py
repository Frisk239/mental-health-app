#!/usr/bin/env python3
"""
数据库初始化脚本
用于创建PostgreSQL数据库和表结构
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import bcrypt

# 数据库配置
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mental_health.db")

Base = declarative_base()

class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    avatar_url = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    mood_entries = relationship("MoodEntry", back_populates="user")
    farm_crops = relationship("FarmCrop", back_populates="user")
    social_sessions = relationship("SocialSession", back_populates="user")

class MoodEntry(Base):
    """情绪记录表"""
    __tablename__ = "mood_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    mood_level = Column(String(20), nullable=False)  # very_low, low, okay, good, great
    context = Column(String(100))
    notes = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # 多模态数据
    face_emotion = Column(String(50))  # happy, sad, angry, etc.
    voice_emotion = Column(String(50))
    text_emotion = Column(String(50))

    # 活动记录
    activities = Column(Text)  # JSON string of activities

    # 关系
    user = relationship("User", back_populates="mood_entries")

class FarmCrop(Base):
    """农场作物表"""
    __tablename__ = "farm_crops"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    crop_type = Column(String(50), nullable=False)  # sunflower, tulip, etc.
    emotion_type = Column(String(20), nullable=False)  # linked emotion
    growth_level = Column(Float, default=0.0)  # 0.0 to 1.0
    planted_at = Column(DateTime, default=datetime.utcnow)
    last_watered = Column(DateTime, default=datetime.utcnow)
    is_harvested = Column(Boolean, default=False)

    # 关系
    user = relationship("User", back_populates="farm_crops")

class SocialSession(Base):
    """社交训练会话表"""
    __tablename__ = "social_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scenario_id = Column(String(50), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    score = Column(Float)  # 0.0 to 100.0
    feedback = Column(Text)  # JSON string of feedback
    duration = Column(Integer)  # in seconds

    # 关系
    user = relationship("User", back_populates="social_sessions")

def create_database():
    """创建数据库和表"""
    try:
        print("🔄 正在创建数据库引擎...")
        engine = create_engine(DATABASE_URL)

        print("📋 正在创建表结构...")
        Base.metadata.create_all(bind=engine)

        print("✅ 数据库初始化完成！")

        # 创建会话
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()

        try:
            # 创建默认管理员用户
            print("👤 正在创建默认用户...")
            hashed_password = bcrypt.hashpw("admin123".encode('utf-8'), bcrypt.gensalt())

            default_user = User(
                username="admin",
                email="admin@mentalhealth.com",
                hashed_password=hashed_password.decode('utf-8'),
                full_name="系统管理员",
                is_active=True
            )

            db.merge(default_user)  # 使用merge避免重复插入
            db.commit()

            print("✅ 默认用户创建完成！")
            print("   用户名: admin")
            print("   密码: admin123")

        except Exception as e:
            print(f"⚠️  创建默认用户时出错: {e}")
            db.rollback()
        finally:
            db.close()

    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        print("请检查:")
        print("1. PostgreSQL服务是否运行")
        print("2. 数据库连接字符串是否正确")
        print("3. 用户权限是否足够")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 开始初始化心灵健康助手数据库...")
    create_database()
    print("🎉 数据库初始化完成！您可以开始使用应用了。")
