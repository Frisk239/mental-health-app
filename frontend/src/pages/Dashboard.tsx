import React from 'react'
import { Link } from 'react-router-dom'
import { Heart, Camera, Sprout, Users, History, Settings, Sun, Moon } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'

const Dashboard: React.FC = () => {
  const { theme, toggleTheme } = useTheme()

  const features = [
    {
      icon: <Camera className="w-8 h-8" />,
      title: '情绪监测',
      description: '多模态实时情绪识别',
      path: '/detect',
      color: 'bg-blue-500'
    },
    {
      icon: <Sprout className="w-8 h-8" />,
      title: '情绪农场',
      description: '游戏化情绪调节',
      path: '/farm',
      color: 'bg-green-500'
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: '社交实验室',
      description: 'AI陪伴社交练习',
      path: '/social',
      color: 'bg-purple-500'
    },
    {
      icon: <History className="w-8 h-8" />,
      title: '历史记录',
      description: '查看情绪变化趋势',
      path: '/history',
      color: 'bg-orange-500'
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      {/* Header */}
      <header className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <Heart className="w-8 h-8 text-red-500" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                心灵健康助手
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={toggleTheme}
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
              </button>
              <Link
                to="/settings"
                className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                <Settings className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            欢迎回来！今天感觉如何？
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            让我们一起关注您的心理健康，通过AI技术提供个性化的支持
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {features.map((feature, index) => (
            <Link
              key={index}
              to={feature.path}
              className="group bg-white dark:bg-gray-800 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1 border border-gray-200 dark:border-gray-700"
            >
              <div className="p-6">
                <div className={`inline-flex p-3 rounded-lg ${feature.color} text-white mb-4`}>
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300">
                  {feature.description}
                </p>
              </div>
            </Link>
          ))}
        </div>

        {/* Quick Stats */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
            本周概览
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">7</div>
              <div className="text-gray-600 dark:text-gray-300">情绪记录</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">85%</div>
              <div className="text-gray-600 dark:text-gray-300">积极情绪</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">3</div>
              <div className="text-gray-600 dark:text-gray-300">农场作物</div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default Dashboard
