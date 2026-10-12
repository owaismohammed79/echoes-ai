import { createSocketService } from "../socketService"
import { createRecorderService } from "./recorderService"
import { createVADService } from "../vadService"
import { initMSE, setOnPlaybackEnd, signalEndOfStream, handleAudioChunk, flushBuffer } from "./audioPlayerService"

type SessionMessage = { type: string; text?: string }

export function createSessionController(onMessage: (data: SessionMessage) => void) {
  let socket: ReturnType<typeof createSocketService> | null = null
  let mediaRecorder: ReturnType<typeof createRecorderService> | null = null
  let vad: ReturnType<typeof createVADService> | null = null
  let stream: MediaStream | null = null

  let isAISpeaking = false
  let isInterviewActive = false
  let silenceTimer: ReturnType<typeof setTimeout> | null = null

  async function getStream(): Promise<MediaStream> {
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      })
    }
    console.log("Stream found")
    return stream
  }

  async function start() {
    isInterviewActive = true
    isAISpeaking = false

    const audioStream = await getStream()
    initMSE()

    socket = createSocketService({
      onMessage: (data) => {
        onMessage(data)
      },
      onAudioChunk: (chunk) => {
        isAISpeaking = true
        handleAudioChunk(chunk)
      },
    })
    await socket.connect()

    console.log("Websocket connection established")

    mediaRecorder = createRecorderService(audioStream, {
      onCommit: (blob) => {
        socket!.send(blob)
        socket!.sendJson({ type: "end" })
      },
      onStop: () => {
        if (!isInterviewActive) return
        if (!isAISpeaking) {
          mediaRecorder!.start()
        }
      },
    })
    mediaRecorder.start()
    console.log("Recorder starting to record")

    const vadCallbacks = {
      onSpeakingStart: () => {
        if (silenceTimer) {
          clearTimeout(silenceTimer)
          silenceTimer = null
          console.log("User continued speaking")
          return
        }

        if (!isAISpeaking) return
        console.log("User tried to interrupt")
        flushBuffer()
        socket!.sendJson({ type: "barge_in" })
        isAISpeaking = false
        if (mediaRecorder!.getState() === "inactive") mediaRecorder!.start()
      },
      onSpeakingEnd: () => {
        const isRecording = mediaRecorder?.getState() === "recording"
        if (socket?.isOpen() && !isAISpeaking && isRecording) {
          console.log("User quiet... waiting 2 secs to confirm")
          if (silenceTimer) clearTimeout(silenceTimer)

          silenceTimer = setTimeout(() => {
            console.log("Silence confirmed, stopping recorder")
            mediaRecorder!.stop()
            silenceTimer = null
          }, 2000)
        }
      },
    }

    if (!vad) {
      vad = createVADService(audioStream, vadCallbacks)
    } else {
      vad.updateCallbacks(vadCallbacks)
    }

    setOnPlaybackEnd(() => {
      isAISpeaking = false
      if (mediaRecorder?.getState() === "inactive") {
        mediaRecorder.start()
      }
    })
  }

  function stop() {
    isInterviewActive = false
    mediaRecorder?.stop()
    vad?.stop()
    vad = null

    socket?.sendJson({ type: "end" })
    socket?.disconnect()
    signalEndOfStream()
  }

  return { start, stop }
}
