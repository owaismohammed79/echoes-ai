type SocketCallbacks = {
  onMessage: (data: any) => void
  onAudioChunk: (chunk: ArrayBuffer) => void
  onError?: (e: Event) => void
}

export function createSocketService(callbacks: SocketCallbacks) {
  let socket = null

  function connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      socket = new WebSocket("ws://localhost:8000/ws/interview")
      socket.binaryType = "arraybuffer"

      socket.onopen = () => resolve()

      socket.onmessage = (event) => {
        if (typeof event.data === "string") {
          const data = JSON.parse(event.data)
          callbacks.onMessage(data)
        } else {
          callbacks.onAudioChunk(event.data as ArrayBuffer)
        }
      }

      socket.onerror = (e) => {
        console.error("[Socket] Error:", e)
        callbacks.onError?.(e)
        reject(e)
      }
    })
  }

  function send(data: Blob | string) {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(data)
    } else {
      console.warn("can't send data on non open socket")
    }
  }

  function sendJson(payload: object) {
    send(JSON.stringify(payload))
  }

  function disconnect() {
    socket?.close()
    socket = null
  }

  function isOpen(): boolean {
    return socket?.readyState === WebSocket.OPEN
  }

  return { connect, send, sendJson, disconnect, isOpen }
}