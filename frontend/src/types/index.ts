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
