import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Mic, MicOff, Upload, Play, Square, AlertCircle } from 'lucide-react'
import { SpeechEmotionData, SpeechEmotionResult, AudioRecordingState } from '../types'
import { speechEmotionService } from '../services'

interface SpeechEmotionDetectorProps {
  onEmotionDetected: (emotions: SpeechEmotionData) => void
}

const SpeechEmotionDetector: React.FC<SpeechEmotionDetectorProps> = ({ onEmotionDetected }) => {
  const [recordingState, setRecordingState] = useState<AudioRecordingState>({
    isRecording: false,
    duration: 0,
    audioData: null,
    emotionResult: null
  })

  const [isConnected, setIsConnected] = useState(false)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const durationIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // 处理语音情绪识别结果
  const handleSpeechEmotionResult = useCallback((result: SpeechEmotionResult) => {
    console.log('🎤 接收到语音情绪识别结果:', result)

    // 转换格式以适配现有的SpeechEmotionData类型
    const emotionData: SpeechEmotionData = {
      happy: result.probabilities.happy || 0,
      sad: result.probabilities.sad || 0,
      angry: result.probabilities.anger || 0,
      surprised: result.probabilities.surprise || 0,
      neutral: result.probabilities.neutral || 0,
      disgust: result.probabilities.disgust || 0,
      fearful: result.probabilities.fearful || 0,
      timestamp: new Date()
    }

    setRecordingState(prev => ({
      ...prev,
      emotionResult: emotionData
    }))

    onEmotionDetected(emotionData)
  }, [onEmotionDetected])

  // 处理连接状态变化
  const handleConnectionChange = useCallback((connected: boolean) => {
    console.log('🔗 语音连接状态变化:', connected)
    setIsConnected(connected)
    if (!connected) {
      setConnectionError('语音服务连接失败')
    } else {
      setConnectionError(null)
    }
  }, [])

  // 处理连接错误
  const handleError = useCallback((error: string) => {
    console.error('❌ 语音识别错误:', error)
    setConnectionError(error)
  }, [])

  // 开始录音
  const startRecording = useCallback(async () => {
    try {
      console.log('🎤 开始录音...')

      // 请求麦克风权限
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      })

      streamRef.current = stream
      audioChunksRef.current = []

      // 创建MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      })

      mediaRecorderRef.current = mediaRecorder

      // 设置数据处理
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        setRecordingState(prev => ({
          ...prev,
          audioData: audioBlob,
          isRecording: false
        }))

        // 停止所有音轨
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop())
        }

        console.log('🎤 录音结束')
      }

      // 开始录音
      mediaRecorder.start(100) // 每100ms收集一次数据

      setRecordingState(prev => ({
        ...prev,
        isRecording: true,
        duration: 0,
        audioData: null,
        emotionResult: null
      }))

      // 开始计时
      durationIntervalRef.current = setInterval(() => {
        setRecordingState(prev => ({
          ...prev,
          duration: prev.duration + 0.1
        }))
      }, 100)

      console.log('✅ 录音已开始')

    } catch (error) {
      console.error('❌ 启动录音失败:', error)
      setConnectionError('无法访问麦克风，请检查权限设置')
    }
  }, [])

  // 停止录音
  const stopRecording = useCallback(() => {
    console.log('🛑 停止录音')

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
    }

    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current)
      durationIntervalRef.current = null
    }

    setRecordingState(prev => ({
      ...prev,
      isRecording: false
    }))
  }, [])

  // 分析录音结果
  const analyzeRecording = useCallback(async () => {
    if (!recordingState.audioData) {
      setConnectionError('没有录音数据')
      return
    }

    setIsProcessing(true)
    setConnectionError(null)

    try {
      console.log('🎵 开始分析录音...')

      const result = await speechEmotionService.analyzeAudio(recordingState.audioData)
      handleSpeechEmotionResult(result)

      console.log('✅ 录音分析完成')

    } catch (error: any) {
      console.error('❌ 分析录音失败:', error)
      setConnectionError(error.message || '分析失败')
    } finally {
      setIsProcessing(false)
    }
  }, [recordingState.audioData, handleSpeechEmotionResult])

  // 处理文件上传
  const handleFileUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // 检查文件类型
      const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/x-wav', 'audio/wave']
      if (!allowedTypes.includes(file.type)) {
        setConnectionError('不支持的文件类型，请上传 WAV 或 MP3 文件')
        return
      }

      // 检查文件大小 (限制为50MB)
      if (file.size > 50 * 1024 * 1024) {
        setConnectionError('文件过大，请上传小于50MB的音频文件')
        return
      }

      setUploadedFile(file)
      setConnectionError(null)
      console.log('📁 文件已选择:', file.name, file.size, 'bytes')
    }
  }, [])

  // 分析上传的文件
  const analyzeUploadedFile = useCallback(async () => {
    if (!uploadedFile) {
      setConnectionError('没有选择文件')
      return
    }

    setIsProcessing(true)
    setConnectionError(null)

    try {
      console.log('📁 开始分析上传文件...')

      const result = await speechEmotionService.analyzeAudioFile(uploadedFile)
      handleSpeechEmotionResult(result)

      console.log('✅ 文件分析完成')

    } catch (error: any) {
      console.error('❌ 分析文件失败:', error)
      setConnectionError(error.message || '分析失败')
    } finally {
      setIsProcessing(false)
    }
  }, [uploadedFile, handleSpeechEmotionResult])

  // 格式化时长显示
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 10)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`
  }

  // 获取主要情绪
  const getPrimaryEmotion = (emotions: SpeechEmotionData): string => {
    const emotionEntries = Object.entries(emotions).filter(([key]) => key !== 'timestamp')
    return emotionEntries.reduce((max, current) =>
      current[1] > max[1] ? current : max
    )[0]
  }

  // 获取情绪标签
  const getEmotionLabel = (emotion: string): string => {
    const labels: Record<string, string> = {
      happy: '开心',
      sad: '悲伤',
      angry: '愤怒',
      surprised: '惊讶',
      neutral: '平静',
      disgust: '厌恶',
      fearful: '恐惧'
    }
    return labels[emotion] || emotion
  }

  // 获取情绪颜色
  const getEmotionColor = (emotion: string): string => {
    const colors: Record<string, string> = {
      happy: 'text-green-600',
      sad: 'text-blue-600',
      angry: 'text-red-600',
      surprised: 'text-purple-600',
      neutral: 'text-gray-600',
      disgust: 'text-yellow-600',
      fearful: 'text-orange-600'
    }
    return colors[emotion] || 'text-gray-600'
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          语音情绪监测
        </h2>
        <div className="flex items-center space-x-2">
          {recordingState.isRecording && (
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse" />
              <span className="text-sm text-red-600">录音中...</span>
            </div>
          )}
          <div className="text-sm text-gray-600 dark:text-gray-300">
            状态: {isConnected ? '已连接' : '未连接'} |
            时长: {formatDuration(recordingState.duration)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 录音控制区域 */}
        <div className="space-y-4">
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              实时录音分析
            </h3>

            <div className="flex items-center space-x-4 mb-4">
              <button
                onClick={recordingState.isRecording ? stopRecording : startRecording}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium ${
                  recordingState.isRecording
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
                disabled={isProcessing}
              >
                {recordingState.isRecording ? (
                  <>
                    <Square className="w-5 h-5" />
                    <span>停止录音</span>
                  </>
                ) : (
                  <>
                    <Mic className="w-5 h-5" />
                    <span>开始录音</span>
                  </>
                )}
              </button>

              {recordingState.audioData && !recordingState.isRecording && (
                <button
                  onClick={analyzeRecording}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium"
                  disabled={isProcessing}
                >
                  <Play className="w-5 h-5" />
                  <span>{isProcessing ? '分析中...' : '分析录音'}</span>
                </button>
              )}
            </div>

            {recordingState.audioData && (
              <div className="text-sm text-gray-600 dark:text-gray-300">
                录音文件大小: {(recordingState.audioData.size / 1024).toFixed(1)} KB
              </div>
            )}
          </div>

          {/* 文件上传区域 */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              文件上传分析
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  选择音频文件
                </label>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="block w-full text-sm text-gray-500 dark:text-gray-400
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-lg file:border-0
                    file:text-sm file:font-medium
                    file:bg-blue-50 file:text-blue-700
                    dark:file:bg-blue-900 dark:file:text-blue-300
                    hover:file:bg-blue-100 dark:hover:file:bg-blue-800"
                />
              </div>

              {uploadedFile && (
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-600 dark:text-gray-300">
                    {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
                  </div>
                  <button
                    onClick={analyzeUploadedFile}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium text-sm"
                    disabled={isProcessing}
                  >
                    <Play className="w-4 h-4" />
                    <span>{isProcessing ? '分析中...' : '分析文件'}</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* 错误提示 */}
          {connectionError && (
            <div className="bg-red-100 dark:bg-red-900 border border-red-400 text-red-700 dark:text-red-200 px-4 py-3 rounded">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 mr-2" />
                <span>{connectionError}</span>
              </div>
            </div>
          )}
        </div>

        {/* 情绪分析结果 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            语音情绪分析结果
          </h3>

          {recordingState.emotionResult ? (
            <div className="space-y-4">
              {/* 主要情绪 */}
              <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
                    主要情绪
                  </span>
                  <span className={`text-lg font-bold ${getEmotionColor(getPrimaryEmotion(recordingState.emotionResult))}`}>
                    {getEmotionLabel(getPrimaryEmotion(recordingState.emotionResult))}
                  </span>
                </div>
                <div className="mt-2 text-2xl font-bold text-gray-900 dark:text-white">
                  {Math.round((recordingState.emotionResult as any)[getPrimaryEmotion(recordingState.emotionResult)] * 100)}%
                </div>
              </div>

              {/* 情绪分布 */}
              <div className="space-y-2">
                {Object.entries(recordingState.emotionResult)
                  .filter(([key]) => key !== 'timestamp')
                  .map(([emotion, value]) => (
                    <div key={emotion} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-300 capitalize">
                        {getEmotionLabel(emotion)}
                      </span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${value * 100}%` }}
                          />
                        </div>
                        <span className="text-sm font-medium text-gray-900 dark:text-white w-10 text-right">
                          {Math.round(value * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
              </div>

              {/* 时间戳 */}
              <div className="text-xs text-gray-500 dark:text-gray-400 pt-2 border-t border-gray-200 dark:border-gray-700">
                分析时间: {recordingState.emotionResult.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 dark:bg-gray-700 p-8 rounded-lg text-center">
              <Mic className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-gray-600 dark:text-gray-300">
                录音或上传音频文件后将显示分析结果
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 使用说明 */}
      <div className="mt-6 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
        <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-2">
          使用提示
        </h4>
        <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-1">
          <li>• 点击"开始录音"开始实时语音分析</li>
          <li>• 或者上传音频文件进行分析</li>
          <li>• 支持的格式: WAV, MP3</li>
          <li>• 文件大小限制: 50MB</li>
          <li>• 支持的情绪: 开心、悲伤、愤怒、惊讶、平静、厌恶、恐惧</li>
        </ul>
      </div>
    </div>
  )
}

export default SpeechEmotionDetector
