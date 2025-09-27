import React, { useState } from 'react'
import EmotionDetector from '../components/EmotionDetector'
import { EmotionData } from '../types'
import { MessageSquare, Mic, Camera } from 'lucide-react'

const EmotionDetection: React.FC = () => {
  const [currentEmotions, setCurrentEmotions] = useState<EmotionData | null>(null)
  const [detectionMode, setDetectionMode] = useState<'face' | 'voice' | 'text'>('face')
  const [isActive, setIsActive] = useState(false)

  const handleEmotionDetected = (emotions: EmotionData) => {
    setCurrentEmotions(emotions)
  }

  const getEmotionAdvice = (emotions: EmotionData): string => {
    const primaryEmotion = Object.entries(emotions)
      .filter(([key]) => key !== 'timestamp')
      .reduce((max, current) => current[1] > max[1] ? current : max)[0]

    const advice: Record<string, string> = {
      happy: "太好了！保持这种积极的状态。建议分享您的快乐给身边的人。",
      sad: "我理解您现在的心情。建议深呼吸几次，或写下您的感受。",
      angry: "愤怒是正常的反应。建议找个安静的地方冷静一下。",
      surprised: "惊喜的感觉很棒！这通常伴随着好奇心。",
      neutral: "平静的状态有助于专注。建议继续保持。"
    }

    return advice[primaryEmotion] || "继续关注您的情绪变化。"
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* 页面标题 */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            多模态情绪监测
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            通过面部表情、语音和文本分析，实时了解您的情绪状态
          </p>
        </div>

        {/* 检测模式选择 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            选择检测模式
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button
              onClick={() => setDetectionMode('face')}
              className={`p-4 rounded-lg border-2 transition-all ${
                detectionMode === 'face'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-blue-300'
              }`}
            >
              <Camera className="w-8 h-8 mx-auto mb-2 text-blue-600" />
              <div className="font-medium text-gray-900 dark:text-white">面部表情</div>
              <div className="text-sm text-gray-600 dark:text-gray-300">实时摄像头分析</div>
            </button>

            <button
              onClick={() => setDetectionMode('voice')}
              className={`p-4 rounded-lg border-2 transition-all ${
                detectionMode === 'voice'
                  ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-green-300'
              }`}
            >
              <Mic className="w-8 h-8 mx-auto mb-2 text-green-600" />
              <div className="font-medium text-gray-900 dark:text-white">语音分析</div>
              <div className="text-sm text-gray-600 dark:text-gray-300">语调情绪识别</div>
            </button>

            <button
              onClick={() => setDetectionMode('text')}
              className={`p-4 rounded-lg border-2 transition-all ${
                detectionMode === 'text'
                  ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-purple-300'
              }`}
            >
              <MessageSquare className="w-8 h-8 mx-auto mb-2 text-purple-600" />
              <div className="font-medium text-gray-900 dark:text-white">文本分析</div>
              <div className="text-sm text-gray-600 dark:text-gray-300">文字情绪挖掘</div>
            </button>
          </div>
        </div>

        {/* 情绪检测组件 */}
        {detectionMode === 'face' && (
          <EmotionDetector
            onEmotionDetected={handleEmotionDetected}
            isActive={isActive}
          />
        )}

        {detectionMode === 'voice' && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              语音情绪分析
            </h2>
            <div className="bg-yellow-100 dark:bg-yellow-900 border border-yellow-400 text-yellow-700 dark:text-yellow-200 px-4 py-3 rounded">
              <p>语音分析功能正在开发中...</p>
            </div>
          </div>
        )}

        {detectionMode === 'text' && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              文本情绪分析
            </h2>
            <div className="space-y-4">
              <textarea
                className="w-full h-32 p-3 border border-gray-300 dark:border-gray-600 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                placeholder="请输入您想分析的文字..."
              />
              <button className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                分析情绪
              </button>
            </div>
            <div className="bg-yellow-100 dark:bg-yellow-900 border border-yellow-400 text-yellow-700 dark:text-yellow-200 px-4 py-3 rounded mt-4">
              <p>文本分析功能正在开发中...</p>
            </div>
          </div>
        )}

        {/* 情绪建议 */}
        {currentEmotions && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
              个性化建议
            </h2>
            <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 p-4 rounded-lg">
              <p className="text-gray-800 dark:text-gray-200">
                {getEmotionAdvice(currentEmotions)}
              </p>
            </div>
          </div>
        )}

        {/* 检测控制 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                检测控制
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {isActive ? '正在进行实时情绪监测' : '点击开始进行情绪检测'}
              </p>
            </div>
            <button
              onClick={() => setIsActive(!isActive)}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                isActive
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isActive ? '停止检测' : '开始检测'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EmotionDetection
