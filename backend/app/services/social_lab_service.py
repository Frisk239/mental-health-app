"""
社交实验室核心服务
处理场景管理、对话生成、反馈分析等功能
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from app.models.database import db_manager
from app.services.deepseek_service import deepseek_service

logger = logging.getLogger(__name__)

class SocialLabService:
    """社交实验室服务类"""

    def __init__(self):
        self.db = db_manager

    async def initialize(self) -> bool:
        """初始化服务"""
        try:
            # 初始化数据库
            self.db.initialize_database()
            logger.info("✅ 社交实验室数据库初始化完成")
            return True
        except Exception as e:
            logger.error(f"❌ 社交实验室服务初始化失败: {e}")
            return False

    async def get_available_scenarios(self, user_id: int = 1) -> List[Dict]:
        """获取用户可用的场景列表"""
        try:
            # 获取所有场景
            scenarios = self.db.execute_query("""
                SELECT * FROM scenarios ORDER BY difficulty, created_at
            """)

            # 获取用户已解锁的成就
            unlocked_achievements = self.db.execute_query("""
                SELECT achievement_id FROM user_achievements WHERE user_id = ?
            """, (user_id,))

            unlocked_achievement_ids = {row['achievement_id'] for row in unlocked_achievements}

            available_scenarios = []
            for scenario in scenarios:
                scenario_dict = dict(scenario)

                # 解析脚本
                if scenario_dict.get('script'):
                    try:
                        scenario_dict['script'] = json.loads(scenario_dict['script'])
                    except:
                        scenario_dict['script'] = {}

                # 直接解锁所有场景
                scenario_dict['is_unlocked'] = True

                available_scenarios.append(scenario_dict)

            return available_scenarios

        except Exception as e:
            logger.error(f"❌ 获取场景列表失败: {e}")
            return []

    async def start_practice_session(self, user_id: int, scenario_id: str) -> Optional[Dict]:
        """开始练习会话"""
        try:
            # 检查场景是否存在
            scenario = self.db.execute_query("""
                SELECT * FROM scenarios WHERE id = ?
            """, (scenario_id,))

            if not scenario:
                return None

            scenario_data = dict(scenario[0])

            # 创建新的练习会话
            session_id = self.db.execute_insert("""
                INSERT INTO practice_sessions (user_id, scenario_id, start_time)
                VALUES (?, ?, ?)
            """, (user_id, scenario_id, datetime.now()))

            # 解析场景脚本
            script = {}
            if scenario_data.get('script'):
                try:
                    script = json.loads(scenario_data['script'])
                except:
                    script = {}

            return {
                'session_id': session_id,
                'scenario': scenario_data,
                'script': script,
                'start_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 开始练习会话失败: {e}")
            return None

    async def generate_ai_response(self, session_id: int, user_message: str,
                                 voice_emotions: Optional[Dict] = None,
                                 face_emotions: Optional[Dict] = None) -> Dict:
        """生成AI回复"""
        try:
            # 获取会话信息
            session = self.db.execute_query("""
                SELECT ps.*, s.name as scenario_name, s.ai_role, s.script
                FROM practice_sessions ps
                JOIN scenarios s ON ps.scenario_id = s.id
                WHERE ps.id = ?
            """, (session_id,))

            if not session:
                return {'error': '会话不存在'}

            session_data = dict(session[0])

            # 解析场景脚本
            script = {}
            if session_data.get('script'):
                try:
                    script = json.loads(session_data['script'])
                except:
                    script = {}

            # 构建AI提示词
            ai_role = session_data.get('ai_role', '助手')
            scenario_name = session_data.get('scenario_name', '社交练习')

            prompt = f"""
你正在扮演一个{ai_role}，在一个{scenario_name}场景中与用户进行对话练习。

你的目标是：
1. 提供自然的对话体验
2. 帮助用户练习社交技能
3. 根据用户的表现给予适当的反馈
4. 保持对话的连贯性和教育性

当前用户信息：
- 用户消息: "{user_message}"
"""

            if voice_emotions:
                prompt += f"- 语音情绪: {json.dumps(voice_emotions, ensure_ascii=False)}\n"

            if face_emotions:
                prompt += f"- 面部表情: {json.dumps(face_emotions, ensure_ascii=False)}\n"

            prompt += """
请以{ai_role}的身份回复，回复要自然、适当，并有助于用户练习社交技能。
重要：请不要使用括号()来表示情绪或动作，如（微笑点头），请直接用正常对话的方式表达。
"""

            # 调用DeepSeek生成回复
            messages = [
                {"role": "system", "content": "你是一个专业的社交技能教练AI助手。请根据用户输入和场景要求，生成自然、适当的回复。"},
                {"role": "user", "content": prompt}
            ]
            ai_result = await deepseek_service.chat_completion(messages, temperature=0.8)

            if ai_result["success"]:
                ai_response = ai_result["response"]
            else:
                ai_response = "我理解你的分享，请继续练习。"

            return {
                'response': ai_response,
                'role': ai_role,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ 生成AI回复失败: {e}")
            return {'error': '生成回复失败'}

    async def end_practice_session(self, session_id: int, dialogue_history: List[Dict],
                                 voice_emotions: Optional[Dict] = None,
                                 face_emotions: Optional[Dict] = None) -> Dict:
        """结束练习会话并生成反馈"""
        try:
            # 更新会话结束时间
            self.db.execute_update("""
                UPDATE practice_sessions
                SET end_time = ?, voice_emotions = ?, face_emotions = ?, dialogue_history = ?
                WHERE id = ?
            """, (
                datetime.now(),
                json.dumps(voice_emotions) if voice_emotions else None,
                json.dumps(face_emotions) if face_emotions else None,
                json.dumps(dialogue_history),
                session_id
            ))

            # 生成综合反馈
            feedback = await self._generate_comprehensive_feedback(
                session_id, dialogue_history, voice_emotions, face_emotions
            )

            # 更新反馈信息
            self.db.execute_update("""
                UPDATE practice_sessions
                SET feedback = ?, improvement_suggestions = ?, total_score = ?
                WHERE id = ?
            """, (
                feedback.get('feedback', ''),
                feedback.get('suggestions', ''),
                feedback.get('score', 0),
                session_id
            ))

            # 检查成就解锁
            await self._check_achievements(session_id)

            return feedback

        except Exception as e:
            logger.error(f"❌ 结束练习会话失败: {e}")
            return {'error': '结束会话失败'}

    async def _generate_comprehensive_feedback(self, session_id: int,
                                             dialogue_history: List[Dict],
                                             voice_emotions: Optional[Dict],
                                             face_emotions: Optional[Dict]) -> Dict:
        """生成综合反馈分析"""
        try:
            # 获取会话信息
            session = self.db.execute_query("""
                SELECT ps.*, s.name as scenario_name
                FROM practice_sessions ps
                JOIN scenarios s ON ps.scenario_id = s.id
                WHERE ps.id = ?
            """, (session_id,))

            if not session:
                return {'error': '会话不存在'}

            session_data = dict(session[0])

            # 构建分析提示词
            prompt = f"""
请分析以下社交练习表现，并提供详细的反馈和改进建议：

场景: {session_data.get('scenario_name', '未知')}
对话轮数: {len(dialogue_history)}

对话历史:
{json.dumps(dialogue_history, ensure_ascii=False, indent=2)}

情绪数据:
"""

            if voice_emotions:
                prompt += f"语音情绪: {json.dumps(voice_emotions, ensure_ascii=False)}\n"

            if face_emotions:
                prompt += f"面部表情: {json.dumps(face_emotions, ensure_ascii=False)}\n"

            prompt += """
请从以下方面进行分析：
1. 沟通技巧和表达能力
2. 情绪管理和自信度
3. 对话的自然度和流畅性
4. 社交礼仪和互动质量

请提供：
- 综合评分 (0-100分)
- 详细反馈
- 具体改进建议
- 鼓励性话语

请用JSON格式返回，包含字段：score, feedback, suggestions, encouragement
"""

            # 调用DeepSeek进行分析
            messages = [
                {"role": "system", "content": "你是一个专业的情绪分析和社交技能教练。请分析用户的表现并提供建设性反馈。"},
                {"role": "user", "content": prompt}
            ]
            analysis_result = await deepseek_service.chat_completion(messages, temperature=0.7)

            if analysis_result["success"]:
                analysis_text = analysis_result["response"]
            else:
                analysis_text = '{"score": 75, "feedback": "练习完成，表现良好", "suggestions": "继续练习", "encouragement": "你做得很好"}'

            try:
                # 尝试解析JSON结果
                feedback_data = json.loads(analysis_text)
                return feedback_data
            except:
                # 如果解析失败，返回默认结构
                return {
                    'score': 75,
                    'feedback': analysis_text,
                    'suggestions': '继续练习，提高自信心',
                    'encouragement': '你做得很好，继续努力！'
                }

        except Exception as e:
            logger.error(f"❌ 生成综合反馈失败: {e}")
            return {
                'score': 70,
                'feedback': '练习完成，表现良好',
                'suggestions': '继续练习，提高社交技能',
                'encouragement': '保持进步，继续努力！'
            }

    async def _check_achievements(self, session_id: int):
        """检查成就解锁"""
        try:
            # 获取会话信息
            session = self.db.execute_query("""
                SELECT * FROM practice_sessions WHERE id = ?
            """, (session_id,))

            if not session:
                return

            session_data = dict(session[0])
            user_id = session_data['user_id']

            # 检查"初次尝试"成就
            first_session_count = self.db.execute_query("""
                SELECT COUNT(*) as count FROM practice_sessions WHERE user_id = ?
            """, (user_id,))

            if first_session_count and first_session_count[0]['count'] >= 1:
                await self._unlock_achievement(user_id, 'first_session')

            # 检查其他成就逻辑...

        except Exception as e:
            logger.error(f"❌ 检查成就失败: {e}")

    async def _unlock_achievement(self, user_id: int, achievement_id: str):
        """解锁成就"""
        try:
            # 检查是否已解锁
            existing = self.db.execute_query("""
                SELECT * FROM user_achievements
                WHERE user_id = ? AND achievement_id = ?
            """, (user_id, achievement_id))

            if existing:
                return  # 已解锁

            # 解锁成就
            self.db.execute_insert("""
                INSERT INTO user_achievements (user_id, achievement_id)
                VALUES (?, ?)
            """, (user_id, achievement_id))

            logger.info(f"🎉 用户 {user_id} 解锁成就: {achievement_id}")

        except Exception as e:
            logger.error(f"❌ 解锁成就失败: {e}")

    async def get_user_progress(self, user_id: int = 1) -> Dict:
        """获取用户进度统计"""
        try:
            # 总练习次数
            total_sessions = self.db.execute_query("""
                SELECT COUNT(*) as count FROM practice_sessions WHERE user_id = ?
            """, (user_id,))

            # 平均得分
            avg_score = self.db.execute_query("""
                SELECT AVG(total_score) as avg_score FROM practice_sessions
                WHERE user_id = ? AND total_score IS NOT NULL
            """, (user_id,))

            # 已解锁成就
            achievements = self.db.execute_query("""
                SELECT a.* FROM achievements a
                JOIN user_achievements ua ON a.id = ua.achievement_id
                WHERE ua.user_id = ?
            """, (user_id,))

            return {
                'total_sessions': total_sessions[0]['count'] if total_sessions else 0,
                'average_score': avg_score[0]['avg_score'] if avg_score and avg_score[0]['avg_score'] else 0,
                'achievements': [dict(a) for a in achievements]
            }

        except Exception as e:
            logger.error(f"❌ 获取用户进度失败: {e}")
            return {
                'total_sessions': 0,
                'average_score': 0,
                'achievements': []
            }

    async def get_session_history(self, user_id: int = 1, limit: int = 10) -> List[Dict]:
        """获取会话历史"""
        try:
            sessions = self.db.execute_query("""
                SELECT ps.*, s.name as scenario_name
                FROM practice_sessions ps
                JOIN scenarios s ON ps.scenario_id = s.id
                WHERE ps.user_id = ?
                ORDER BY ps.created_at DESC
                LIMIT ?
            """, (user_id, limit))

            return [dict(session) for session in sessions]

        except Exception as e:
            logger.error(f"❌ 获取会话历史失败: {e}")
            return []

# 全局服务实例
social_lab_service = SocialLabService()
