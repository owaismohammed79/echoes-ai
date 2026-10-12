import hark from "hark"

export function createVADService(stream, callbacks = {}) {
  let speech = null
  let cb = {...callbacks}
  speech = hark(stream, { interval: 70, threshold: -50 })

  speech.on("speaking", () => cb.onSpeakingStart?.())
  speech.on("stopped_speaking", () => cb.onSpeakingEnd?.())

  function updateCallbacks(newCb) {
    callbacks = { ...cb, ...newCb }
  }

  function stop() {
    speech?.stop()
    speech = null
  }

  return { stop, updateCallbacks }
}