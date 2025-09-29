import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ThemeProvider } from './hooks/useTheme'
import Dashboard from './pages/Dashboard'
import EmotionDetection from './pages/EmotionDetection'
import SpeechEmotionDetection from './pages/SpeechEmotionDetection'
import SocialLab from './pages/SocialLab'
import Settings from './pages/Settings'
import './App.css'

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/detect" element={<EmotionDetection />} />
            <Route path="/speech-detect" element={<SpeechEmotionDetection />} />
            <Route path="/social" element={<SocialLab />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </div>
      </Router>
    </ThemeProvider>
  )
}

export default App
