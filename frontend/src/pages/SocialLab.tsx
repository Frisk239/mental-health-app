import React, { useState, useEffect, useRef } from 'react'
import { MessageCircle, Users, Mic, MicOff, Camera, Play, Square, Send, Volume2, VolumeX, Settings, Phone, PhoneOff } from 'lucide-react'
import {
  SocialLabScenario,
  PracticeSession,
  ChatMessage,
  SessionFeedback
} from '../types'

const SocialLab: React.FC = () => {
  // 基础状态
  const [scenarios, setScenarios] = useState<SocialLabScenario[]>([])
  const [selectedScenario, setSelectedScenario] = useState<SocialLabScenario | null>(null)
  const [currentSession, setCurrentSession] = useState<PracticeSession | null>(null)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [userMessage, setUserMessage] = useState('')
  const [isSessionActive, setIsSessionActive] = useState(false)
  const [sessionFeedback, setSessionFeedback] = useState<SessionFeedback | null>(null)
  const [loading, setLoading] = useState(false)

  // 语音交互状态
  const [inputMode, setInputMode] = useState<'text' | 'voice'>('text')
  const [isRecording, setIsRecording] = useState(false)
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true)
  const [audioLevel, setAudioLevel] = useState(0)
  const [availableRoles, setAvailableRoles] = useState<any[]>([])
  const [voiceServiceStatus, setVoiceServiceStatus] = useState<any>(null)

  // 引用
  const websocketRef = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // 获取可用场景和语音服务状态
  useEffect(() => {
    fetchScenarios()
    fetchVoiceServiceStatus()
    fetchAvailableRoles()
  }, [])

  // 自动滚动到最新消息
  useEffect(() => {
    scrollToBottom()
  }, [chatMessages])

  // 音频可视化
  useEffect(() => {
    if (isRecording && audioContextRef.current && analyserRef.current) {
      const updateAudioLevel = () => {
        if (analyserRef.current) {
          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
          analyserRef.current.getByteFrequencyData(dataArray)
          const average = dataArray.reduce((a, b) => a + b) / dataArray.length
          setAudioLevel(average / 255)
        }
        if (isRecording) {
          animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
        }
      }
      updateAudioLevel()

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current)
        }
      }
    }
  }, [isRecording])

  const fetchScenarios = async () => {
    try {
      const response = await fetch('/api/social-lab/scenarios')
      const data = await response.json()
      setScenarios(data.scenarios)
    } catch (error) {
      console.error('获取场景失败:', error)
    }
  }



  const fetchVoiceServiceStatus = async () => {
    try {
      const response = await fetch('/api/voice/health')
      const status = await response.json()
      setVoiceServiceStatus(status)
    } catch (error) {
      console.error('获取语音服务状态失败:', error)
    }
  }

  const fetchAvailableRoles = async () => {
    try {
      const response = await fetch('/api/voice/roles')
      const data = await response.json()
      setAvailableRoles(data.roles)
    } catch (error) {
      console.error('获取可用角色失败:', error)
    }
  }

  const startPracticeSession = async (scenario: SocialLabScenario) => {
    // 如果有活跃会话，先结束当前会话
    if (isSessionActive && currentSession) {
      await endPracticeSession()
    }

    try {
      setLoading(true)
      const response = await fetch('/api/social-lab/sessions/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scenario_id: scenario.id
        })
      })

      const session = await response.json()
      setCurrentSession(session)
      setSelectedScenario(scenario)
      setIsSessionActive(true)
      setChatMessages([])
      setSessionFeedback(null)

      // 连接WebSocket
      connectWebSocket(session.session_id)

      // 添加AI开场白
      if (session.script?.opening) {
        setChatMessages([{
          role: 'assistant',
          message: session.script.opening,
          timestamp: new Date().toISOString()
        }])
      }

    } catch (error) {
      console.error('开始会话失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const connectWebSocket = (sessionId: number) => {
    const wsUrl = `ws://localhost:8000/api/social-lab/sessions/${sessionId}/chat`
    websocketRef.current = new WebSocket(wsUrl)

    websocketRef.current.onopen = () => {
      console.log('WebSocket连接成功')
    }

    websocketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.error) {
        console.error('WebSocket错误:', data.error)
        return
      }

      const message: ChatMessage = {
        role: 'assistant',
        message: data.response,
        timestamp: data.timestamp
      }
      setChatMessages(prev => [...prev, message])
    }

    websocketRef.current.onclose = () => {
      console.log('WebSocket连接关闭')
    }

    websocketRef.current.onerror = (error) => {
      console.error('WebSocket错误:', error)
    }
  }

  const sendMessage = () => {
    if (!userMessage.trim() || !websocketRef.current) return

    const message: ChatMessage = {
      role: 'user',
      message: userMessage,
      timestamp: new Date().toISOString()
    }

    setChatMessages(prev => [...prev, message])

    // 发送消息到WebSocket
    websocketRef.current.send(JSON.stringify({
      message: userMessage,
      voice_emotions: null, // 可以集成语音情绪
      face_emotions: null  // 可以集成面部情绪
    }))

    setUserMessage('')
  }

  const endPracticeSession = async () => {
    if (!currentSession) return

    try {
      setLoading(true)
      const response = await fetch(`/api/social-lab/sessions/${currentSession.session_id}/end`, {
        method: 'POST'
      })

      const result = await response.json()
      setSessionFeedback(result.feedback)
      setIsSessionActive(false)

      // 断开WebSocket
      if (websocketRef.current) {
        websocketRef.current.close()
        websocketRef.current = null
      }

    } catch (error) {
      console.error('结束会话失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'medium': return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'hard': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getDifficultyText = (difficulty: string) => {
    switch (difficulty) {
      case 'easy': return '简单'
      case 'medium': return '中等'
      case 'hard': return '困难'
      default: return difficulty
    }
  }

  const playVoiceMessage = async (text: string) => {
    try {
      console.log('🎵 开始播放语音消息:', text)

      // 使用当前场景的默认角色进行语音合成
      const defaultRole = selectedScenario?.ai_role || '张教授'
      console.log('🎭 使用角色:', defaultRole)

      const response = await fetch('/api/voice/synthesize-speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          text: text,
          role_name: defaultRole,
          speed: '1.0',
          pitch_shift: '1.0',
          emotion_intensity: '0.7'
        })
      })

      console.log('📡 API响应状态:', response.status, response.statusText)
      console.log('📡 响应头:', Object.fromEntries(response.headers.entries()))

      if (response.ok) {
        console.log('✅ API调用成功，开始处理音频数据')

        const audioBlob = await response.blob()
        console.log('📦 Blob创建成功，大小:', audioBlob.size, '类型:', audioBlob.type)

        if (audioBlob.size === 0) {
          console.error('❌ Blob大小为0，音频数据为空')
          return
        }

        const audioUrl = URL.createObjectURL(audioBlob)
        console.log('🔗 Audio URL创建成功:', audioUrl)

        const audio = new Audio(audioUrl)
        console.log('🎵 Audio对象创建成功')

        // 添加事件监听器
        audio.addEventListener('loadstart', () => console.log('⏳ 音频开始加载'))
        audio.addEventListener('canplay', () => console.log('✅ 音频可以播放'))
        audio.addEventListener('play', () => console.log('▶️ 音频开始播放'))
        audio.addEventListener('ended', () => {
          console.log('⏹️ 音频播放结束')
          URL.revokeObjectURL(audioUrl) // 清理URL
        })

        // 尝试播放
        try {
          const playPromise = audio.play()
          console.log('🎯 调用audio.play()')

          if (playPromise !== undefined) {
            playPromise
              .then(() => {
                console.log('✅ 音频播放成功启动')
              })
              .catch((playError) => {
                console.error('❌ 音频播放失败:', playError)
                console.error('播放错误详情:', {
                  name: playError.name,
                  message: playError.message,
                  code: playError.code
                })
              })
          } else {
            console.log('⚠️ audio.play()返回undefined（可能是旧版浏览器）')
          }
        } catch (playError) {
          console.error('❌ 调用audio.play()时出错:', playError)
        }
      } else {
        console.error('❌ API调用失败:', response.status, response.statusText)
        const errorText = await response.text()
        console.error('❌ 错误详情:', errorText)
      }
    } catch (error) {
      console.error('❌ 播放语音失败:', error)
      const err = error as Error
      console.error('错误详情:', {
        name: err.name,
        message: err.message,
        stack: err.stack
      })
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* 页面标题 */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            社交实验室
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            在AI陪伴下练习社交技能，逐步建立自信
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* 左侧：场景选择 */}
          <div className="lg:col-span-1 space-y-6">
            {/* 场景列表 */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                练习场景
              </h3>
              <div className="space-y-3">
                {scenarios.map((scenario) => (
                  <div
                    key={scenario.id}
                    className={`p-4 border rounded-lg cursor-pointer transition-all ${
                      scenario.is_unlocked
                        ? 'border-gray-200 dark:border-gray-600 hover:border-blue-300 hover:shadow-md'
                        : 'border-gray-100 dark:border-gray-700 opacity-60 cursor-not-allowed'
                    } ${selectedScenario?.id === scenario.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : ''}`}
                    onClick={() => scenario.is_unlocked && startPracticeSession(scenario)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h4 className="font-medium text-gray-900 dark:text-white">
                        {scenario.name}
                      </h4>
                      <span className={`px-2 py-1 text-xs rounded-full ${getDifficultyColor(scenario.difficulty)}`}>
                        {getDifficultyText(scenario.difficulty)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                      {scenario.description}
                    </p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center text-xs text-gray-500">
                        <Users className="w-3 h-3 mr-1" />
                        {scenario.ai_role}
                      </div>
                      {scenario.is_unlocked && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            // 打开音色配置面板
                          }}
                          className="px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded transition-colors"
                          title="配置音色"
                        >
                          ⚙️
                        </button>
                      )}
                    </div>
                    {!scenario.is_unlocked && (
                      <div className="mt-2 text-xs text-orange-600 dark:text-orange-400">
                        🔒 需要先完成其他场景解锁
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 右侧：对话界面 */}
          <div className="lg:col-span-2">
            {currentSession && isSessionActive ? (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg h-[600px] flex flex-col">
                {/* 会话头部 */}
                <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white">
                      {selectedScenario?.name}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      与 {selectedScenario?.ai_role} 对话练习中
                    </p>
                  </div>
                  <button
                    onClick={endPracticeSession}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm font-medium"
                    disabled={loading}
                  >
                    {loading ? '结束中...' : '结束练习'}
                  </button>
                </div>

                {/* 消息区域 */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                  {chatMessages.map((message, index) => (
                    <div
                      key={index}
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                          message.role === 'user'
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white'
                        }`}
                      >
                        <p className="text-sm">{message.message}</p>
                        <div className="flex items-center justify-between mt-1">
                          <p className="text-xs opacity-70">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </p>
                          {message.role === 'assistant' && voiceServiceStatus?.tts_service && (
                            <button
                              onClick={() => playVoiceMessage(message.message)}
                              className="ml-2 p-1 hover:bg-gray-300 dark:hover:bg-gray-600 rounded transition-colors"
                              title="播放语音"
                            >
                              <Volume2 className="w-3 h-3" />
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>

                {/* 双模态输入区域 */}
                <div className="p-4 border-t border-gray-200 dark:border-gray-700">
                  {/* 输入模式切换 */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setInputMode('text')}
                        className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${
                          inputMode === 'text'
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                        }`}
                      >
                        📝 文本输入
                      </button>
                      <button
                        onClick={() => setInputMode('voice')}
                        className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${
                          inputMode === 'voice'
                            ? 'bg-green-600 text-white'
                            : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                        }`}
                        disabled={!voiceServiceStatus?.stt_service}
                      >
                        🎤 语音输入
                      </button>
                    </div>

                    {/* 语音服务状态指示器 */}
                    <div className="flex items-center space-x-2 text-xs">
                      <div className={`w-2 h-2 rounded-full ${voiceServiceStatus?.tts_service ? 'bg-green-500' : 'bg-red-500'}`} />
                      <span className="text-gray-500">
                        {voiceServiceStatus?.tts_service ? '语音服务正常' : '语音服务离线'}
                      </span>
                    </div>
                  </div>

                  {/* 文本输入模式 */}
                  {inputMode === 'text' && (
                    <div className="space-y-2">
                      <div className="flex space-x-2">
                        <input
                          type="text"
                          value={userMessage}
                          onChange={(e) => setUserMessage(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                          placeholder="输入您的回复..."
                          className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                          disabled={!websocketRef.current}
                        />
                        <button
                          onClick={sendMessage}
                          disabled={!userMessage.trim() || !websocketRef.current}
                          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg flex items-center"
                        >
                          <Send className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>💡 尝试使用完整的句子，展现自信的语气</span>
                        <span>🎯 练习目标：自然流畅的对话</span>
                      </div>
                    </div>
                  )}

                  {/* 语音输入模式 */}
                  {inputMode === 'voice' && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-center space-x-4">
                        <button
                          onClick={() => {/* 开始录音 */}}
                          disabled={isRecording}
                          className={`px-6 py-3 rounded-lg font-medium transition-all ${
                            isRecording
                              ? 'bg-gray-400 cursor-not-allowed'
                              : 'bg-green-600 hover:bg-green-700 text-white'
                          }`}
                        >
                          {isRecording ? (
                            <>
                              <div className="flex items-center space-x-2">
                                <div className="flex space-x-1">
                                  {[...Array(5)].map((_, i) => (
                                    <div
                                      key={i}
                                      className="w-1 bg-white rounded-full transition-all"
                                      style={{
                                        height: `${Math.max(8, 20 * audioLevel)}px`
                                      }}
                                    />
                                  ))}
                                </div>
                                <span>录音中...</span>
                              </div>
                            </>
                          ) : (
                            <>
                              <Mic className="w-5 h-5 mr-2" />
                              开始录音
                            </>
                          )}
                        </button>

                        {isRecording && (
                          <button
                            onClick={() => {/* 停止录音 */}}
                            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium"
                          >
                            <Square className="w-4 h-4 mr-2" />
                            停止录音
                          </button>
                        )}
                      </div>

                      {/* 录音状态指示器 */}
                      {isRecording && (
                        <div className="flex items-center justify-center space-x-2 text-sm text-gray-600 dark:text-gray-300">
                          <div className="flex space-x-1">
                            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
                            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }} />
                          </div>
                          <span>正在录音，请说话...</span>
                        </div>
                      )}

                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>🎤 点击按钮开始语音输入</span>
                        <span>🔊 支持实时语音识别</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : sessionFeedback ? (
              /* 反馈结果 */
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
                <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
                  练习完成！🎉
                </h3>

                <div className="text-center mb-6">
                  <div className="text-6xl font-bold text-blue-600 mb-2">
                    {sessionFeedback.score}分
                  </div>
                  <div className="text-lg text-gray-600 dark:text-gray-300">
                    综合评分
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                    <h4 className="font-semibold text-green-800 dark:text-green-200 mb-2">
                      📝 详细反馈
                    </h4>
                    <p className="text-green-700 dark:text-green-300">
                      {sessionFeedback.feedback}
                    </p>
                  </div>

                  <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                    <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
                      💡 改进建议
                    </h4>
                    <p className="text-blue-700 dark:text-blue-300">
                      {sessionFeedback.suggestions}
                    </p>
                  </div>

                  <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                    <h4 className="font-semibold text-purple-800 dark:text-purple-200 mb-2">
                      🌟 鼓励的话
                    </h4>
                    <p className="text-purple-700 dark:text-purple-300">
                      {sessionFeedback.encouragement}
                    </p>
                  </div>
                </div>

                <div className="text-center mt-6">
                  <button
                    onClick={() => {
                      setSessionFeedback(null)
                      setCurrentSession(null)
                      setSelectedScenario(null)
                    }}
                    className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium"
                  >
                    选择其他场景继续练习
                  </button>
                </div>
              </div>
            ) : (
              /* 欢迎界面 */
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8">
                <div className="text-center mb-8">
                  <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                    <MessageCircle className="w-12 h-12 text-white" />
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                    开始您的社交练习
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-6">
                    选择左侧的场景开始与AI伙伴的对话练习。在安全的环境中提升您的社交技能。
                  </p>
                </div>

                {/* 功能特色展示 */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                  <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                    <div className="w-12 h-12 mx-auto mb-3 bg-blue-100 dark:bg-blue-800 rounded-full flex items-center justify-center">
                      <MessageCircle className="w-6 h-6 text-blue-600" />
                    </div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">智能对话</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      基于DeepSeek的AI伙伴，自然流畅的对话体验
                    </p>
                  </div>

                  <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="w-12 h-12 mx-auto mb-3 bg-green-100 dark:bg-green-800 rounded-full flex items-center justify-center">
                      <Mic className="w-6 h-6 text-green-600" />
                    </div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">语音交互</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      支持语音输入和合成，像真人对话一样自然
                    </p>
                  </div>

                  <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                    <div className="w-12 h-12 mx-auto mb-3 bg-purple-100 dark:bg-purple-800 rounded-full flex items-center justify-center">
                      <Camera className="w-6 h-6 text-purple-600" />
                    </div>
                    <h4 className="font-semibold text-gray-900 dark:text-white mb-2">情绪分析</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300">
                      实时分析语音和表情，提供个性化反馈建议
                    </p>
                  </div>


                </div>

                {/* 服务状态指示器 */}
                <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 mb-6">
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-3 flex items-center">
                    <Settings className="w-4 h-4 mr-2" />
                    服务状态
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-300">语音识别服务</span>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${voiceServiceStatus?.stt_service ? 'bg-green-500' : 'bg-red-500'}`} />
                        <span className={voiceServiceStatus?.stt_service ? 'text-green-600' : 'text-red-600'}>
                          {voiceServiceStatus?.stt_service ? '正常' : '离线'}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-300">语音合成服务</span>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${voiceServiceStatus?.tts_service ? 'bg-green-500' : 'bg-red-500'}`} />
                        <span className={voiceServiceStatus?.tts_service ? 'text-green-600' : 'text-red-600'}>
                          {voiceServiceStatus?.tts_service ? '正常' : '离线'}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 dark:text-gray-300">可用角色</span>
                      <span className="font-medium text-blue-600">
                        {Array.isArray(availableRoles) ? availableRoles.length : 0}个
                      </span>
                    </div>
                  </div>
                </div>

                {/* 快速开始提示 */}
                <div className="text-center">
                  <p className="text-gray-600 dark:text-gray-300 mb-4">
                    💡 <strong>新手建议</strong>：从"课堂发言"场景开始，熟悉对话流程后逐步尝试语音输入模式
                  </p>
                  <div className="flex items-center justify-center space-x-4 text-sm text-gray-500">
                    <span>🎯 支持文本输入</span>
                    <span>🎤 支持语音输入</span>
                    <span>🤖 AI智能反馈</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default SocialLab
