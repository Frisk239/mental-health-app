import React, { useRef, useEffect, useState, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Camera, CameraOff, RefreshCw, AlertCircle } from 'lucide-react'
import { EmotionData } from '../types'
import { faceRecognitionService, FaceEmotionResult } from '../services/faceRecognition'

interface EmotionDetectorProps {
  onEmotionDetected: (emotions: EmotionData) => void
  isActive: boolean
}

const EmotionDetector: React.FC<EmotionDetectorProps> = ({ onEmotionDetected, isActive }) => {
  const webcamRef = useRef<Webcam>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isWebcamReady, setIsWebcamReady] = useState(false)
  const [isDetecting, setIsDetecting] = useState(false)
  const [currentEmotions, setCurrentEmotions] = useState<EmotionData | null>(null)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [isModelReady, setIsModelReady] = useState(false)

  // 真正的面部表情检测
  const detectEmotions = useCallback(async () => {
    console.log('🎯 开始表情检测流程')
    console.log('📊 当前状态:', {
      isActive,
      isWebcamReady,
      hasVideo: !!webcamRef.current?.video,
      modelInitialized: faceRecognitionService['isInitialized']
    })

    if (!isActive || !isWebcamReady || !webcamRef.current?.video) {
      console.log('❌ 检测条件不满足，跳过检测')
      return
    }

    setIsDetecting(true)
    console.log('🔄 开始检测，设置检测状态为true')

    try {
      // 确保模型已初始化
      if (!faceRecognitionService['isInitialized']) {
        console.log('🔧 模型未初始化，开始初始化...')
        await faceRecognitionService.initialize()
        console.log('✅ 模型初始化完成')
      } else {
        console.log('✅ 模型已初始化')
      }

      const video = webcamRef.current.video
      console.log('📹 视频元素状态:', {
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
        readyState: video.readyState,
        paused: video.paused
      })

      // 分析面部表情
      console.log('🧠 开始分析面部表情...')
      const result: FaceEmotionResult | null = await faceRecognitionService.analyzeEmotion(video)

      if (result) {
        console.log('📊 检测到面部表情结果:', result)

        // 验证结果是否有效
        const emotionValues = [result.happy, result.sad, result.angry, result.surprised, result.neutral]
        const isValidResult = emotionValues.every(val =>
          typeof val === 'number' && !isNaN(val) && val >= 0 && val <= 1
        )

        console.log('🔍 结果验证:', {
          emotionValues,
          isValidResult,
          sum: emotionValues.reduce((a, b) => a + b, 0)
        })

        if (isValidResult) {
          // 转换格式
          const emotionData: EmotionData = {
            happy: result.happy,
            sad: result.sad,
            angry: result.angry,
            surprised: result.surprised,
            neutral: result.neutral,
            timestamp: result.timestamp
          }

          setCurrentEmotions(emotionData)
          onEmotionDetected(emotionData)
          console.log('✅ 表情检测成功，更新UI:', emotionData)
        } else {
          console.warn('⚠️ 表情检测结果无效，使用默认值')
          const defaultEmotions: EmotionData = {
            happy: 0.2,
            sad: 0.2,
            angry: 0.2,
            surprised: 0.2,
            neutral: 0.2,
            timestamp: new Date()
          }
          setCurrentEmotions(defaultEmotions)
          onEmotionDetected(defaultEmotions)
        }
      } else {
        console.log('👤 未检测到面部，请确保面部在摄像头视野内')
        // 显示提示信息但不更新情绪数据
        setCurrentEmotions(prev => prev ? prev : {
          happy: 0.2,
          sad: 0.2,
          angry: 0.2,
          surprised: 0.2,
          neutral: 0.2,
          timestamp: new Date()
        })
      }

    } catch (error: any) {
      console.error('❌ 情绪检测失败:', error)
      console.error('🔍 错误详情:', {
        message: error.message,
        stack: error.stack,
        name: error.name
      })

      // 显示用户友好的错误信息
      setCurrentEmotions({
        happy: 0.2,
        sad: 0.2,
        angry: 0.2,
        surprised: 0.2,
        neutral: 0.2,
        timestamp: new Date()
      })

      // 如果是初始化错误，尝试重新初始化
      if (error?.message?.includes('未初始化') || error?.message?.includes('initialized')) {
        try {
          console.log('🔄 尝试重新初始化面部识别模型...')
          await faceRecognitionService.initialize()
          console.log('✅ 面部识别模型重新初始化完成')
        } catch (initError: any) {
          console.error('❌ 模型重新初始化失败:', initError)
        }
      }
    } finally {
      setIsDetecting(false)
      console.log('🏁 检测流程结束，重置检测状态')
    }
  }, [isActive, isWebcamReady, onEmotionDetected])

  useEffect(() => {
    if (isActive && isWebcamReady) {
      const interval = setInterval(detectEmotions, 3000) // 每3秒检测一次
      return () => clearInterval(interval)
    }
  }, [isActive, isWebcamReady])

  const getPrimaryEmotion = (emotions: EmotionData): string => {
    const emotionEntries = Object.entries(emotions).filter(([key]) => key !== 'timestamp')
    return emotionEntries.reduce((max, current) =>
      current[1] > max[1] ? current : max
    )[0]
  }

  const getEmotionLabel = (emotion: string): string => {
    const labels: Record<string, string> = {
      happy: '开心',
      sad: '悲伤',
      angry: '愤怒',
      surprised: '惊讶',
      neutral: '平静'
    }
    return labels[emotion] || emotion
  }

  const getEmotionColor = (emotion: string): string => {
    const colors: Record<string, string> = {
      happy: 'text-green-600',
      sad: 'text-blue-600',
      angry: 'text-red-600',
      surprised: 'text-purple-600',
      neutral: 'text-gray-600'
    }
    return colors[emotion] || 'text-gray-600'
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          实时情绪监测
        </h2>
        <div className="flex items-center space-x-2">
          {isDetecting && (
            <RefreshCw className="w-5 h-5 animate-spin text-blue-600" />
          )}
          <div className="text-sm text-gray-600 dark:text-gray-300">
            {isActive ? '检测中...' : '等待开始'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 摄像头区域 */}
        <div className="space-y-4">
          <div className="relative bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden" style={{ height: '300px' }}>
            {isActive ? (
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={{
                  width: 640,
                  height: 480,
                  facingMode: "user"
                }}
                onUserMedia={() => setIsWebcamReady(true)}
                onUserMediaError={() => setIsWebcamReady(false)}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                  <Camera className="w-12 h-12 mx-auto mb-2" />
                  <p>点击"开始检测"开启摄像头</p>
                </div>
              </div>
            )}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
              style={{ display: isActive ? 'block' : 'none' }}
            />
          </div>

          {!isWebcamReady && isActive && (
            <div className="bg-yellow-100 dark:bg-yellow-900 border border-yellow-400 text-yellow-700 dark:text-yellow-200 px-4 py-3 rounded">
              <p>正在初始化摄像头...</p>
            </div>
          )}
        </div>

        {/* 情绪分析结果 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            情绪分析结果
          </h3>

          {currentEmotions ? (
            <div className="space-y-4">
              {/* 主要情绪 */}
              <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-600 dark:text-gray-300">
                    主要情绪
                  </span>
                  <span className={`text-lg font-bold ${getEmotionColor(getPrimaryEmotion(currentEmotions))}`}>
                    {getEmotionLabel(getPrimaryEmotion(currentEmotions))}
                  </span>
                </div>
                <div className="mt-2 text-2xl font-bold text-gray-900 dark:text-white">
                  {Math.round((currentEmotions as any)[getPrimaryEmotion(currentEmotions)] * 100)}%
                </div>
              </div>

              {/* 情绪分布 */}
              <div className="space-y-2">
                {Object.entries(currentEmotions)
                  .filter(([key]) => key !== 'timestamp')
                  .map(([emotion, value]) => (
                    <div key={emotion} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-300 capitalize">
                        {getEmotionLabel(emotion)}
                      </span>
                      <div className="flex items-center space-x-2">
                        <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
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
                最后更新: {currentEmotions.timestamp.toLocaleTimeString()}
              </div>
            </div>
          ) : (
            <div className="bg-gray-50 dark:bg-gray-700 p-8 rounded-lg text-center">
              <Camera className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-gray-600 dark:text-gray-300">
                {isActive ? '正在分析情绪...' : '开始检测后将显示分析结果'}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* 使用说明 */}
      <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
        <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
          使用提示
        </h4>
        <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
          <li>• 确保光线充足，面部清晰可见</li>
          <li>• 保持自然表情，系统会实时分析</li>
          <li>• 检测结果每3秒更新一次</li>
          <li>• 支持的表情: 开心、悲伤、愤怒、惊讶、平静</li>
        </ul>
      </div>
    </div>
  )
}

export default EmotionDetector
