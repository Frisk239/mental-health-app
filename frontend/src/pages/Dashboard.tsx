import React from 'react'
import { Link } from 'react-router-dom'
import { Heart, Camera, Mic, Users, Settings, Sun, Moon } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'

const Dashboard: React.FC = () => {
  const { theme, toggleTheme } = useTheme()

  const features = [
    {
      icon: <Camera className="w-8 h-8" />,
      title: '面部表情分析',
      description: '实时摄像头分析情绪状态',
      path: '/detect',
      color: 'bg-blue-500'
    },
    {
      icon: <Mic className="w-8 h-8" />,
      title: '语音情绪检测',
      description: '通过语音分析情绪状态',
      path: '/speech-detect',
      color: 'bg-red-500'
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: '社交实验室',
      description: 'AI陪伴社交练习',
      path: '/social',
      color: 'bg-purple-500'
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-4xl mx-auto">
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
      </main>
    </div>
  )
}

export default Dashboard
