import React, { useRef, useEffect, useState } from 'react'
import Webcam from 'react-webcam'
import { Camera, CameraOff, RefreshCw } from 'lucide-react'
import { EmotionData } from '../types'

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

  // 模拟情绪检测（实际项目中会使用TensorFlow.js + MediaPipe）
  const detectEmotions = async () => {
    if (!isActive || !isWebcamReady) return

    setIsDetecting(true)

    try {
      // 模拟AI处理时间
      await new Promise(resolve => setTimeout(resolve, 1000))

      // 模拟情绪检测结果
      const mockEmotions: EmotionData = {
        happy: Math.random() * 0.8 + 0.1,
        sad: Math.random() * 0.3,
        angry: Math.random() * 0.2,
        surprised: Math.random() * 0.4,
        neutral: Math.random() * 0.6 + 0.2,
        timestamp: new Date()
      }

      // 归一化
      const total = Object.values(mockEmotions).reduce((sum, val) => sum + val, 0) - mockEmotions.timestamp.getTime()
      Object.keys(mockEmotions).forEach(key => {
        if (key !== 'timestamp') {
          (mockEmotions as any)[key] = (mockEmotions as any)[key] / total
        }
      })

      setCurrentEmotions(mockEmotions)
      onEmotionDetected(mockEmotions)

    } catch (error) {
      console.error('情绪检测失败:', error)
    } finally {
      setIsDetecting(false)
    }
  }

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
          <button
            onClick={detectEmotions}
            disabled={!isWebcamReady || isDetecting}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg transition-colors"
          >
            {isActive ? <CameraOff className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
            <span>{isActive ? '停止检测' : '开始检测'}</span>
          </button>
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
