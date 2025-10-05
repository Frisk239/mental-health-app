// 情绪相关类型
export interface EmotionData {
  happy: number
  sad: number
  angry: number
  surprised: number
  neutral: number
  disgust: number
  fear: number
  timestamp: Date
}

export interface MoodEntry {
  id: string
  level: 'very_low' | 'low' | 'okay' | 'good' | 'great'
  context: string
  activities: string[]
  notes: string
  timestamp: Date
}

// 用户相关类型
export interface User {
  id: string
  name: string
  email: string
  avatar?: string
  preferences: UserPreferences
}

export interface UserPreferences {
  theme: 'light' | 'dark'
  notifications: boolean
  language: string
}

// 游戏相关类型
export interface Crop {
  id: string
  type: string
  growth: number
  maxGrowth: number
  emotion: keyof EmotionData
  plantedAt: Date
}

export interface FarmState {
  crops: Crop[]
  score: number
  level: number
}

// API响应类型
export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

// 情绪检测配置
export interface EmotionDetectionConfig {
  faceMesh: boolean
  audioAnalysis: boolean
  textAnalysis: boolean
  realTime: boolean
}

// 社交实验室类型
export interface Scenario {
  id: string
  title: string
  description: string
  difficulty: 'easy' | 'medium' | 'hard'
  type: 'presentation' | 'conversation' | 'interview'
  script: DialogueLine[]
}

export interface DialogueLine {
  speaker: 'user' | 'ai'
  text: string
  emotion?: keyof EmotionData
  delay?: number
}

export interface LabSession {
  id: string
  scenarioId: string
  startTime: Date
  endTime?: Date
  score: number
  feedback: string[]
  emotions: EmotionData[]
}

// 语音情绪识别类型
export interface SpeechEmotionResult {
  emotion: string
  emotion_chinese: string
  confidence: number
  probabilities: Record<string, number>
  timestamp: number
}

export interface SpeechEmotionData {
  happy: number
  sad: number
  angry: number
  surprised: number
  neutral: number
  disgust: number
  fearful: number
  timestamp: Date
}

export interface AudioRecordingState {
  isRecording: boolean
  duration: number
  audioData: Blob | null
  emotionResult: SpeechEmotionData | null
}

// 社交实验室类型
export interface SocialLabScenario {
  id: string
  name: string
  description: string
  difficulty: 'easy' | 'medium' | 'hard'
  category: string
  unlocked_by: string | null
  ai_role: string
  script: {
    opening?: string
    questions?: string[]
    closing?: string
    interruptions?: string[]
    topics?: string[]
  }
  is_unlocked: boolean
  created_at: string
}

export interface PracticeSession {
  session_id: number
  scenario: SocialLabScenario
  script: any
  start_time: string
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  message: string
  timestamp: string
  voice_emotions?: SpeechEmotionData
  face_emotions?: EmotionData
}

export interface SessionFeedback {
  score: number
  feedback: string
  suggestions: string
  encouragement: string
}

export interface UserProgress {
  total_sessions: number
  average_score: number
  achievements: Achievement[]
}

export interface Achievement {
  id: string
  name: string
  description: string
  icon: string
  requirement: string
  reward_type: string
  reward_value: string
  unlocked_at?: string
}

export interface SocialLabStats {
  total_users: number
  total_sessions: number
  average_score: number
  scenario_stats: Array<{
    name: string
    count: number
  }>
  active_sessions: number
}
