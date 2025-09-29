import React, { useState } from 'react'
import EmotionDetector from '../components/EmotionDetector'
import { EmotionData } from '../types'

const EmotionDetection: React.FC = () => {
  const [currentEmotions, setCurrentEmotions] = useState<EmotionData | null>(null)
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
      neutral: "平静的状态有助于专注。建议继续保持。",
      disgust: "厌恶感通常源于某些不适。建议找出原因并寻求解决方案。",
      fear: "恐惧是自我保护的本能。建议深呼吸并理性分析情况。"
    }

    return advice[primaryEmotion] || "继续关注您的情绪变化。"
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* 页面标题 */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            面部表情分析
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            通过实时摄像头分析您的面部表情，识别当前情绪状态
          </p>
        </div>

        {/* 面部表情检测组件 */}
        <EmotionDetector
          onEmotionDetected={handleEmotionDetected}
          isActive={isActive}
        />

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
                {isActive ? '正在进行实时面部表情分析' : '点击开始进行面部表情检测'}
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

        {/* 技术说明 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            技术说明
          </h2>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p className="text-gray-600 dark:text-gray-300">
              本系统使用先进的AI模型实时分析您的面部表情，能够识别7种基本情绪：
              <strong>开心、悲伤、愤怒、惊讶、平静、厌恶、恐惧</strong>。
            </p>
            <p className="text-gray-600 dark:text-gray-300 mt-2">
              通过WebSocket实时通信，确保分析结果的及时性和准确性，为您的心理健康提供科学依据。
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EmotionDetection
