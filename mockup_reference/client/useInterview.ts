import { useRef } from "react";
import { createSessionController } from "../services/sessionController"

type SessionMessage = { type: string; text?: string }

function useInterview(onMessage: (data: SessionMessage) => void) {
  const controllerRef = useRef<ReturnType<typeof createSessionController> | null>(null)

  const startInterview = async () => {
    if(!controllerRef.current) {
      controllerRef.current = createSessionController(onMessage)
    }
    await controllerRef.current.start()
  }
  
  const stopInterview = () => {
    controllerRef.current?.stop()
  }

  return { startInterview, stopInterview }
}

export default useInterview