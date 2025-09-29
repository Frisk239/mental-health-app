/**
 * 视频流服务
 * 负责从摄像头捕获视频帧并发送到后端进行表情识别
 */

export interface EmotionResult {
  emotion: string
  emotion_chinese: string
  confidence: number
  probabilities: Record<string, number>
  timestamp: number
  face_box?: {
    x: number
    y: number
    width: number
    height: number
  }
  faces_count?: number
  message?: string
}

export class VideoStreamService {
  private websocket: WebSocket | null = null
  private mediaRecorder: MediaRecorder | null = null
  private canvas: HTMLCanvasElement | null = null
  private video: HTMLVideoElement | null = null
  private stream: MediaStream | null = null
  private wsConnected = false
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval: NodeJS.Timeout | null = null
  private isCapturing = false
  private captureTimeout: NodeJS.Timeout | null = null

  // 回调函数
  private onEmotionResult?: (result: EmotionResult) => void
  private onConnectionChange?: (connected: boolean) => void
  private onError?: (error: string) => void

  constructor() {
    this.initializeCanvas()
  }

  /**
   * 初始化画布用于图像处理
   */
  private initializeCanvas() {
    this.canvas = document.createElement('canvas')
    this.canvas.width = 640
    this.canvas.height = 480
  }

  /**
   * 连接到后端WebSocket
   */
  async connect(
    onEmotionResult?: (result: EmotionResult) => void,
    onConnectionChange?: (connected: boolean) => void,
    onError?: (error: string) => void
  ): Promise<boolean> {
    this.onEmotionResult = onEmotionResult
    this.onConnectionChange = onConnectionChange
    this.onError = onError

    try {
      console.log('🔌 连接到表情识别WebSocket: ws://localhost:8000/api/emotion/ws/emotion')

      this.websocket = new WebSocket('ws://localhost:8000/api/emotion/ws/emotion')

      return new Promise((resolve, reject) => {
        if (!this.websocket) {
          reject(new Error('WebSocket创建失败'))
          return
        }

        this.websocket.onopen = () => {
          console.log('✅ 表情识别WebSocket连接成功')
          this.wsConnected = true
          this.reconnectAttempts = 0
          this.onConnectionChange?.(true)
          resolve(true)
        }

        this.websocket.onmessage = (event) => {
          try {
            const result: EmotionResult = JSON.parse(event.data)
            console.log('📊 接收到表情识别结果:', result)
            this.onEmotionResult?.(result)
          } catch (error) {
            console.error('❌ 解析表情识别结果失败:', error)
            this.onError?.('解析结果失败')
          }
        }

        this.websocket.onclose = (event) => {
          console.log('❌ 表情识别WebSocket连接已断开:', event.code, event.reason)
          this.wsConnected = false
          this.onConnectionChange?.(false)
          this.handleReconnect()
        }

        this.websocket.onerror = (error) => {
          console.error('❌ 表情识别WebSocket连接错误:', error)
          this.onError?.('连接错误')
          reject(error)
        }

        // 连接超时
        setTimeout(() => {
          if (!this.wsConnected) {
            reject(new Error('连接超时'))
          }
        }, 5000)
      })

    } catch (error) {
      console.error('❌ 创建WebSocket连接失败:', error)
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
      console.log(`🔄 尝试重连表情识别服务 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

      this.reconnectInterval = setTimeout(() => {
        this.connect(this.onEmotionResult, this.onConnectionChange, this.onError)
      }, 2000)
    } else {
      console.error('❌ 达到最大重连次数，停止重连')
      this.onError?.('连接失败，已达到最大重连次数')
    }
  }

  /**
   * 启动摄像头并开始视频流
   */
  async startCamera(): Promise<boolean> {
    try {
      console.log('📹 请求摄像头权限...')

      // 请求摄像头权限
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      })

      console.log('✅ 摄像头权限获取成功')

      // 创建视频元素（如果需要）
      if (!this.video) {
        this.video = document.createElement('video')
        this.video.srcObject = this.stream
        this.video.play()
      }

      return true

    } catch (error) {
      console.error('❌ 摄像头启动失败:', error)
      this.onError?.('无法访问摄像头，请检查权限设置')
      return false
    }
  }

  /**
   * 停止摄像头
   */
  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop())
      this.stream = null
    }

    if (this.video) {
      this.video.srcObject = null
      this.video = null
    }

    console.log('📹 摄像头已停止')
  }

  /**
   * 开始实时表情识别
   */
  async startEmotionDetection(
    onResult?: (result: EmotionResult) => void,
    onConnectionChange?: (connected: boolean) => void,
    onError?: (error: string) => void
  ): Promise<boolean> {
    console.log('🚀 启动实时表情识别')

    // 连接到后端WebSocket
    const connected = await this.connect(onResult, onConnectionChange, onError)
    if (!connected) {
      return false
    }

    // 启动摄像头
    const cameraStarted = await this.startCamera()
    if (!cameraStarted) {
      return false
    }

    // 开始发送视频帧
    this.startFrameCapture()

    console.log('✅ 实时表情识别已启动')
    return true
  }

  /**
   * 开始捕获和发送视频帧
   */
  private startFrameCapture() {
    if (!this.canvas || !this.video || !this.websocket || !this.wsConnected) {
      console.error('❌ 无法开始帧捕获：缺少必要组件')
      return
    }

    const context = this.canvas.getContext('2d')
    if (!context) {
      console.error('❌ 无法获取Canvas上下文')
      return
    }

    console.log('📹 开始捕获视频帧...')
    this.isCapturing = true

    const captureFrame = async () => {
      // 检查是否应该停止捕获
      if (!this.isCapturing || !this.video || !this.websocket || !this.wsConnected) {
        console.log('🛑 停止视频帧捕获')
        return
      }

      try {
        // 在canvas上绘制当前视频帧
        context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height)

        // 转换为JPEG格式
        this.canvas.toBlob(async (blob) => {
          if (blob && this.websocket && this.wsConnected && this.isCapturing) {
            // 转换为字节数组
            const arrayBuffer = await blob.arrayBuffer()
            const uint8Array = new Uint8Array(arrayBuffer)

            // 发送到后端
            this.websocket.send(uint8Array)

            // 继续下一帧（如果还在捕获）
            if (this.isCapturing) {
              this.captureTimeout = setTimeout(captureFrame, 200) // 5 FPS
            }
          } else if (this.isCapturing) {
            // 如果blob为空但还在捕获，继续下一帧
            this.captureTimeout = setTimeout(captureFrame, 200)
          }
        }, 'image/jpeg', 0.8) // 80%质量

      } catch (error) {
        console.error('❌ 帧捕获失败:', error)
        // 出错时也继续捕获（如果还在运行）
        if (this.isCapturing) {
          this.captureTimeout = setTimeout(captureFrame, 200)
        }
      }
    }

    // 开始捕获
    captureFrame()
  }

  /**
   * 停止表情识别
   */
  stopEmotionDetection() {
    console.log('🛑 停止表情识别')

    // 停止帧捕获
    this.isCapturing = false
    if (this.captureTimeout) {
      clearTimeout(this.captureTimeout)
      this.captureTimeout = null
    }

    // 停止重连定时器
    if (this.reconnectInterval) {
      clearTimeout(this.reconnectInterval)
      this.reconnectInterval = null
    }

    // 关闭WebSocket连接
    if (this.websocket) {
      this.websocket.close()
      this.websocket = null
    }

    // 停止摄像头
    this.stopCamera()

    this.wsConnected = false
    this.onConnectionChange?.(false)

    console.log('✅ 表情识别已停止')
  }

  /**
   * 获取连接状态
   */
  isConnected(): boolean {
    return this.wsConnected
  }

  /**
   * 手动发送测试帧
   */
  async sendTestFrame(): Promise<boolean> {
    if (!this.canvas || !this.websocket || !this.wsConnected) {
      console.error('❌ 无法发送测试帧：WebSocket未连接')
      return false
    }

    try {
      // 创建一个测试图像（彩色条纹）
      const context = this.canvas.getContext('2d')
      if (!context) return false

      // 绘制彩色条纹
      for (let i = 0; i < 10; i++) {
        context.fillStyle = `hsl(${(i * 36) % 360}, 70%, 50%)`
        context.fillRect(i * 64, 0, 64, 480)
      }

      // 转换为blob
      return new Promise((resolve) => {
        if (!this.canvas) {
          resolve(false)
          return
        }

        this.canvas.toBlob((blob) => {
          if (blob && this.websocket && this.wsConnected) {
            blob.arrayBuffer().then(arrayBuffer => {
              const uint8Array = new Uint8Array(arrayBuffer)
              this.websocket.send(uint8Array)
              console.log('📤 测试帧已发送')
              resolve(true)
            }).catch(error => {
              console.error('❌ 数组缓冲区转换失败:', error)
              resolve(false)
            })
          } else {
            resolve(false)
          }
        }, 'image/jpeg', 0.8)
      })

    } catch (error) {
      console.error('❌ 发送测试帧失败:', error)
      return false
    }
  }
}

// 全局服务实例
export const videoStreamService = new VideoStreamService()
