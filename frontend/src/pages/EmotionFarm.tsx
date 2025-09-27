import React, { useState } from 'react'
import EmotionFarm from '../components/EmotionFarm'
import { EmotionData } from '../types'

const EmotionFarmPage: React.FC = () => {
  const [currentEmotion, setCurrentEmotion] = useState<string>('neutral')

  // 模拟从情绪检测获取当前情绪
  const handleEmotionUpdate = (emotions: EmotionData) => {
    const primaryEmotion = Object.entries(emotions)
      .filter(([key]) => key !== 'timestamp')
      .reduce((max, current) => current[1] > max[1] ? current : max)[0]

    setCurrentEmotion(primaryEmotion)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 p-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* 页面标题 */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            🌱 情绪农场
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            通过种植和照顾作物，学习管理情绪，建立积极的生活习惯
          </p>
        </div>

        {/* 情绪状态指示器 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                当前情绪状态
              </h2>
              <p className="text-gray-600 dark:text-gray-300">
                您当前的主要情绪会影响农场的作物生长
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-blue-600 mb-1 capitalize">
                {currentEmotion === 'happy' ? '😊 开心' :
                 currentEmotion === 'sad' ? '😢 悲伤' :
                 currentEmotion === 'angry' ? '😠 愤怒' :
                 currentEmotion === 'surprised' ? '😲 惊讶' : '😐 平静'}
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-300">
                影响作物生长速度
              </div>
            </div>
          </div>
        </div>

        {/* 情绪农场组件 */}
        <EmotionFarm currentEmotion={currentEmotion} />

        {/* 情绪与种植的关系说明 */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            情绪与种植的奥秘
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-gray-900 dark:text-white mb-3">
                🌻 情绪作物对应关系
              </h3>
              <div className="space-y-2 text-sm">
                <div className="flex items-center space-x-2">
                  <span className="text-yellow-500">🌻</span>
                  <span>向日葵 - 对应开心情绪</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-blue-500">🌷</span>
                  <span>郁金香 - 对应悲伤情绪</span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-red-500">🌹</span>
                  <span>玫瑰 - 对应愤怒情绪</span>
                </div>
              </div>
            </div>

            <div>
              <h3 className="font-medium text-gray-900 dark:text-white mb-3">
                🎯 农场益处
              </h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                <li>• 培养耐心和责任感</li>
                <li>• 学习情绪管理技巧</li>
                <li>• 建立积极的日常习惯</li>
                <li>• 通过成就感提升自信</li>
                <li>• 创造有意义的休闲活动</li>
              </ul>
            </div>
          </div>
        </div>

        {/* 今日目标 */}
        <div className="bg-gradient-to-r from-green-100 to-blue-100 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
            🎯 今日农场目标
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <div className="text-2xl mb-2">💧</div>
              <div className="font-medium text-gray-900 dark:text-white">浇水3次</div>
              <div className="text-sm text-gray-600 dark:text-gray-300">帮助作物生长</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <div className="text-2xl mb-2">🌱</div>
              <div className="font-medium text-gray-900 dark:text-white">收获1株</div>
              <div className="text-sm text-gray-600 dark:text-gray-300">庆祝成长成果</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg">
              <div className="text-2xl mb-2">📊</div>
              <div className="font-medium text-gray-900 dark:text-white">观察情绪</div>
              <div className="text-sm text-gray-600 dark:text-gray-300">了解心情变化</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default EmotionFarmPage
