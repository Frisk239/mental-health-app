/**
 * 语音情绪识别服务
 * 负责与后端语音情绪识别API进行通信
 */

import { SpeechEmotionResult } from '../types'

export class SpeechEmotionService {
  private baseUrl: string
  private websocket: WebSocket | null = null
  private isConnected = false
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval: NodeJS.Timeout | null = null

  // 回调函数
  private onEmotionResult?: (result: SpeechEmotionResult) => void
  private onConnectionChange?: (connected: boolean) => void
  private onError?: (error: string) => void

  constructor(baseUrl: string = 'http://localhost:8000') {
    this.baseUrl = baseUrl
  }

  /**
   * 连接到后端WebSocket
   */
  async connect(
    onEmotionResult?: (result: SpeechEmotionResult) => void,
    onConnectionChange?: (connected: boolean) => void,
    onError?: (error: string) => void
  ): Promise<boolean> {
    this.onEmotionResult = onEmotionResult
    this.onConnectionChange = onConnectionChange
    this.onError = onError

    try {
      console.log('🎤 连接到语音情绪识别WebSocket...')

      this.websocket = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/api/speech-emotion/ws/speech-emotion`)

      return new Promise((resolve, reject) => {
        if (!this.websocket) {
          reject(new Error('WebSocket创建失败'))
          return
        }

        this.websocket.onopen = () => {
          console.log('✅ 语音情绪识别WebSocket连接成功')
          this.isConnected = true
          this.reconnectAttempts = 0
          this.onConnectionChange?.(true)
          resolve(true)
        }

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)

            // 处理心跳包
            if (data.type === 'heartbeat') {
              return
            }

            // 处理错误消息
            if (data.error || data.type === 'init_failed' || data.type === 'process_error') {
              const errorMsg = data.error || '处理失败'
              console.error('❌ 语音情绪识别错误:', errorMsg)
              this.onError?.(errorMsg)
              return
            }

            // 处理情绪识别结果
            if (data.emotion && data.confidence !== undefined) {
              const result: SpeechEmotionResult = {
                emotion: data.emotion,
                emotion_chinese: data.emotion_chinese,
                confidence: data.confidence,
                probabilities: data.probabilities,
                timestamp: data.timestamp
              }

              console.log('📊 接收到语音情绪识别结果:', result)
              this.onEmotionResult?.(result)
            }

          } catch (error) {
            console.error('❌ 解析语音情绪识别结果失败:', error)
            this.onError?.('解析结果失败')
          }
        }

        this.websocket.onclose = (event) => {
          console.log('❌ 语音情绪识别WebSocket连接已断开:', event.code, event.reason)
          this.isConnected = false
          this.onConnectionChange?.(false)
          this.handleReconnect()
        }

        this.websocket.onerror = (error) => {
          console.error('❌ 语音情绪识别WebSocket连接错误:', error)
          this.onError?.('连接错误')
          reject(error)
        }

        // 连接超时
        setTimeout(() => {
          if (!this.isConnected) {
            reject(new Error('连接超时'))
          }
        }, 5000)
      })

    } catch (error) {
      console.error('❌ 创建语音WebSocket连接失败:', error)
      this.onError?.('创建连接失败')
      return false
    }
  }

  /**
   * 处理重连逻辑
   */
  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`🔄 尝试重连语音情绪识别服务 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

      this.reconnectInterval = setTimeout(() => {
        this.connect(this.onEmotionResult, this.onConnectionChange, this.onError)
      }, 2000)
    } else {
      console.error('❌ 达到最大重连次数，停止重连')
      this.onError?.('连接失败，已达到最大重连次数')
    }
  }

  /**
   * 发送音频数据进行实时分析
   */
  async sendAudioData(audioData: ArrayBuffer): Promise<void> {
    if (!this.websocket || !this.isConnected) {
      throw new Error('WebSocket未连接')
    }

    try {
      this.websocket.send(audioData)
      console.log('📤 已发送音频数据进行实时分析')
    } catch (error) {
      console.error('❌ 发送音频数据失败:', error)
      throw error
    }
  }

  /**
   * 分析音频Blob数据
   */
  async analyzeAudio(audioBlob: Blob): Promise<SpeechEmotionResult> {
    try {
      console.log('🎵 开始分析音频Blob...')

      // 将Blob转换为ArrayBuffer
      const arrayBuffer = await audioBlob.arrayBuffer()

      // 发送到WebSocket进行实时分析
      if (this.isConnected && this.websocket) {
        await this.sendAudioData(arrayBuffer)

        // 等待结果（简化处理，实际应该使用Promise等待结果）
        return new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error('分析超时'))
          }, 10000)

          const originalOnEmotionResult = this.onEmotionResult
          this.onEmotionResult = (result) => {
            clearTimeout(timeout)
            this.onEmotionResult = originalOnEmotionResult
            resolve(result)
          }
        })
      } else {
        // 如果WebSocket未连接，使用REST API
        return await this.analyzeAudioViaRest(arrayBuffer)
      }

    } catch (error) {
      console.error('❌ 分析音频Blob失败:', error)
      throw error
    }
  }

  /**
   * 通过REST API分析音频数据
   */
  private async analyzeAudioViaRest(audioData: ArrayBuffer): Promise<SpeechEmotionResult> {
    try {
      console.log('🌐 通过REST API分析音频...')

      const formData = new FormData()
      const audioBlob = new Blob([audioData], { type: 'audio/webm' })
      formData.append('file', audioBlob, 'recording.webm')

      const response = await fetch(`${this.baseUrl}/api/speech-emotion/speech-emotion/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      const result = await response.json()

      if (!result.success) {
        throw new Error(result.error || '分析失败')
      }

      return result.result

    } catch (error) {
      console.error('❌ REST API分析失败:', error)
      throw error
    }
  }

  /**
   * 分析上传的音频文件
   */
  async analyzeAudioFile(file: File): Promise<SpeechEmotionResult> {
    try {
      console.log('📁 开始分析上传的音频文件:', file.name)

      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch(`${this.baseUrl}/api/speech-emotion/speech-emotion/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}`)
      }

      const result = await response.json()

      if (!result.success) {
        throw new Error(result.error || '分析失败')
      }

      console.log('✅ 文件分析完成:', result.result)
      return result.result

    } catch (error) {
      console.error('❌ 分析音频文件失败:', error)
      throw error
    }
  }

  /**
   * 获取服务状态
   */
  async getServiceStatus() {
    try {
      const response = await fetch(`${this.baseUrl}/api/speech-emotion/speech-emotion/status`)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('❌ 获取服务状态失败:', error)
      throw error
    }
  }

  /**
   * 测试服务
   */
  async testService() {
    try {
      const response = await fetch(`${this.baseUrl}/api/speech-emotion/speech-emotion/test`, {
        method: 'POST'
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      return await response.json()
    } catch (error) {
      console.error('❌ 测试服务失败:', error)
      throw error
    }
  }

  /**
   * 断开连接
   */
  disconnect() {
    console.log('🔌 断开语音情绪识别连接')

    if (this.reconnectInterval) {
      clearTimeout(this.reconnectInterval)
      this.reconnectInterval = null
    }

    if (this.websocket) {
      this.websocket.close()
      this.websocket = null
    }

    this.isConnected = false
    this.onConnectionChange?.(false)
  }

  /**
   * 获取连接状态
   */
  isWebSocketConnected(): boolean {
    return this.isConnected
  }
}

// 全局服务实例
export const speechEmotionService = new SpeechEmotionService()
