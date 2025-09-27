import React, { useState, useEffect } from 'react'
import { Sprout, Droplets, Sun, Cloud, Heart } from 'lucide-react'
import { Crop, FarmState } from '../types'

interface EmotionFarmProps {
  currentEmotion?: string
}

const EmotionFarm: React.FC<EmotionFarmProps> = ({ currentEmotion }) => {
  const [farmState, setFarmState] = useState<FarmState>({
    crops: [
      {
        id: 'crop_1',
        type: '向日葵',
        growth: 0.3,
        maxGrowth: 1.0,
        emotion: 'happy',
        plantedAt: new Date(Date.now() - 2 * 60 * 60 * 1000) // 2小时前
      },
      {
        id: 'crop_2',
        type: '郁金香',
        growth: 0.6,
        maxGrowth: 1.0,
        emotion: 'sad',
        plantedAt: new Date(Date.now() - 4 * 60 * 60 * 1000) // 4小时前
      },
      {
        id: 'crop_3',
        type: '玫瑰',
        growth: 0.1,
        maxGrowth: 1.0,
        emotion: 'angry',
        plantedAt: new Date(Date.now() - 30 * 60 * 1000) // 30分钟前
      }
    ],
    score: 1250,
    level: 3
  })

  const [weather, setWeather] = useState<'sunny' | 'cloudy' | 'rainy'>('sunny')

  // 模拟天气变化
  useEffect(() => {
    const weatherInterval = setInterval(() => {
      const weathers: ('sunny' | 'cloudy' | 'rainy')[] = ['sunny', 'cloudy', 'rainy']
      setWeather(weathers[Math.floor(Math.random() * weathers.length)])
    }, 30000) // 每30秒改变天气

    return () => clearInterval(weatherInterval)
  }, [])

  // 模拟作物生长
  useEffect(() => {
    const growthInterval = setInterval(() => {
      setFarmState(prevState => ({
        ...prevState,
        crops: prevState.crops.map(crop => {
          if (crop.growth >= crop.maxGrowth) return crop

          // 根据当前情绪和天气调整生长速度
          let growthRate = 0.01 // 基础生长速度

          // 情绪影响
          if (currentEmotion === crop.emotion) {
            growthRate *= 1.5 // 匹配情绪加速生长
          }

          // 天气影响
          if (weather === 'sunny') {
            growthRate *= 1.2
          } else if (weather === 'rainy') {
            growthRate *= 0.8
          }

          return {
            ...crop,
            growth: Math.min(crop.maxGrowth, crop.growth + growthRate)
          }
        })
      }))
    }, 5000) // 每5秒更新生长

    return () => clearInterval(growthInterval)
  }, [currentEmotion, weather])

  const waterCrop = (cropId: string) => {
    setFarmState(prevState => ({
      ...prevState,
      crops: prevState.crops.map(crop =>
        crop.id === cropId
          ? { ...crop, growth: Math.min(crop.maxGrowth, crop.growth + 0.1) }
          : crop
      ),
      score: prevState.score + 10
    }))
  }

  const harvestCrop = (cropId: string) => {
    setFarmState(prevState => ({
      ...prevState,
      crops: prevState.crops.map(crop =>
        crop.id === cropId
          ? { ...crop, growth: 0, plantedAt: new Date() }
          : crop
      ),
      score: prevState.score + 50
    }))
  }

  const getCropEmoji = (type: string, growth: number) => {
    const emojis = {
      '向日葵': growth > 0.8 ? '🌻' : growth > 0.5 ? '🌱' : '🌿',
      '郁金香': growth > 0.8 ? '🌷' : growth > 0.5 ? '🌱' : '🌿',
      '玫瑰': growth > 0.8 ? '🌹' : growth > 0.5 ? '🌱' : '🌿'
    }
    return emojis[type as keyof typeof emojis] || '🌱'
  }

  const getEmotionColor = (emotion: string) => {
    const colors = {
      happy: 'bg-yellow-100 border-yellow-300',
      sad: 'bg-blue-100 border-blue-300',
      angry: 'bg-red-100 border-red-300',
      surprised: 'bg-purple-100 border-purple-300',
      neutral: 'bg-gray-100 border-gray-300'
    }
    return colors[emotion as keyof typeof colors] || colors.neutral
  }

  const getWeatherIcon = () => {
    switch (weather) {
      case 'sunny': return <Sun className="w-6 h-6 text-yellow-500" />
      case 'cloudy': return <Cloud className="w-6 h-6 text-gray-500" />
      case 'rainy': return <Droplets className="w-6 h-6 text-blue-500" />
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
      {/* 农场头部信息 */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            情绪农场
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            种植与情绪相关的作物，看着它们随您的心情生长
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {/* 天气显示 */}
          <div className="flex items-center space-x-2">
            {getWeatherIcon()}
            <span className="text-sm text-gray-600 dark:text-gray-300 capitalize">
              {weather === 'sunny' ? '晴天' : weather === 'cloudy' ? '多云' : '雨天'}
            </span>
          </div>

          {/* 分数和等级 */}
          <div className="text-right">
            <div className="text-lg font-bold text-green-600">分数: {farmState.score}</div>
            <div className="text-sm text-gray-600 dark:text-gray-300">等级 {farmState.level}</div>
          </div>
        </div>
      </div>

      {/* 当前情绪影响提示 */}
      {currentEmotion && (
        <div className="mb-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Heart className="w-5 h-5 text-red-500" />
            <span className="text-gray-800 dark:text-gray-200">
              当前情绪 <strong>{currentEmotion}</strong> 会影响作物生长速度！
            </span>
          </div>
        </div>
      )}

      {/* 农场网格 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {farmState.crops.map((crop) => (
          <div
            key={crop.id}
            className={`p-4 rounded-lg border-2 transition-all ${getEmotionColor(crop.emotion)}`}
          >
            {/* 作物信息 */}
            <div className="text-center mb-3">
              <div className="text-4xl mb-2">{getCropEmoji(crop.type, crop.growth)}</div>
              <h3 className="font-medium text-gray-900 dark:text-white">{crop.type}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                情绪: {crop.emotion}
              </p>
            </div>

            {/* 生长进度条 */}
            <div className="mb-3">
              <div className="flex justify-between text-sm mb-1">
                <span>生长进度</span>
                <span>{Math.round(crop.growth * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  className="bg-green-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${crop.growth * 100}%` }}
                />
              </div>
            </div>

            {/* 操作按钮 */}
            <div className="flex space-x-2">
              <button
                onClick={() => waterCrop(crop.id)}
                className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition-colors"
              >
                <Droplets className="w-4 h-4" />
                <span>浇水</span>
              </button>

              {crop.growth >= crop.maxGrowth && (
                <button
                  onClick={() => harvestCrop(crop.id)}
                  className="flex-1 flex items-center justify-center space-x-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded text-sm transition-colors"
                >
                  <Sprout className="w-4 h-4" />
                  <span>收获</span>
                </button>
              )}
            </div>

            {/* 种植时间 */}
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
              种植于: {crop.plantedAt.toLocaleString()}
            </div>
          </div>
        ))}
      </div>

      {/* 农场说明 */}
      <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
          农场规则
        </h4>
        <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-1">
          <li>• 作物生长速度受当前情绪和天气影响</li>
          <li>• 浇水可以加速生长 (+10 分数)</li>
          <li>• 收获成熟作物 (+50 分数)</li>
          <li>• 匹配情绪的作物生长更快</li>
          <li>• 晴天生长快，雨天生长慢</li>
        </ul>
      </div>
    </div>
  )
}

export default EmotionFarm
