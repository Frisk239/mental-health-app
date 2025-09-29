import React from 'react'
import SpeechEmotionDetector from '../components/SpeechEmotionDetector'
import { SpeechEmotionData } from '../types'

const SpeechEmotionDetection: React.FC = () => {
  const handleEmotionDetected = (emotions: SpeechEmotionData) => {
    console.log('🎤 语音情绪检测结果:', emotions)
    // 这里可以添加保存到历史记录、更新统计等逻辑
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8 text-center">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            语音情绪检测
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-300">
            通过语音分析识别说话人的情绪状态，支持实时录音和文件上传两种方式
          </p>
        </div>

        <SpeechEmotionDetector onEmotionDetected={handleEmotionDetected} />

        {/* 功能说明 */}
        <div className="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            功能特性
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                🎤 实时录音分析
              </h3>
              <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-2">
                <li>• 点击"开始录音"实时分析语音情绪</li>
                <li>• 支持麦克风权限自动请求</li>
                <li>• 录音时长实时显示</li>
                <li>• 自动降噪和回声消除</li>
              </ul>
            </div>
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                📁 文件上传分析
              </h3>
              <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-2">
                <li>• 支持WAV、MP3等常见音频格式</li>
                <li>• 文件大小限制50MB</li>
                <li>• 快速批量分析</li>
                <li>• 详细的情绪概率分布</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 技术说明 */}
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            技术说明
          </h2>
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p className="text-gray-600 dark:text-gray-300">
              本系统使用OpenAI Whisper Large V3模型进行语音情绪识别，经过微调后能够识别7种基本情绪：
              <strong>开心、悲伤、愤怒、惊讶、平静、厌恶、恐惧</strong>。
            </p>
            <p className="text-gray-600 dark:text-gray-300 mt-2">
              模型准确率达91.99%，支持实时WebSocket通信和REST API调用，适用于心理健康监测、教育评估等多种场景。
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SpeechEmotionDetection
