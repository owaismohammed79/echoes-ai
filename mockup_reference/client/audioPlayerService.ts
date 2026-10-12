let mediaSource
let sourceBuffer
let queue = []
let audio = new Audio()
let isAppending = false
let onPlaybackEnd = null
let objectUrl = null

export function setOnPlaybackEnd(callback) {
  onPlaybackEnd = callback
}

export function initMSE() {
  if(mediaSource && mediaSource.readyState !== 'closed' && sourceBuffer) return

    if(objectUrl) {
      URL.revokeObjectURL(objectUrl)
      objectUrl = null
    }

  mediaSource = new MediaSource()
  objectUrl = URL.createObjectURL(mediaSource)
  audio.src = objectUrl

  audio.onwaiting = () => {
    if(onPlaybackEnd && !isAppending && queue.length === 0) onPlaybackEnd()
  }

  mediaSource.addEventListener('sourceopen', () => {
    sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg')
    
    sourceBuffer.addEventListener('updateend', () => {
      isAppending = false
      if (queue.length > 0) {
        processQueue()
      }
    })
  }, {once: true})
  audio.play().catch(e => console.error("MSE Playback Error:", e))
}

function processQueue() {
  if (queue.length > 0 && !isAppending && sourceBuffer && !sourceBuffer.updating) {
    isAppending = true
    sourceBuffer.appendBuffer(queue.shift())
  }
}

export function handleAudioChunk(chunk) {
  //if stream was previously closed, we need to re init
  if (sourceBuffer && !sourceBuffer.updating && queue.length === 0) {
    isAppending = true
    sourceBuffer.appendBuffer(chunk)
  } else {
    queue.push(chunk)
  }
}

export function flushBuffer() {
  queue = [] //clear existing chunks
  isAppending = false

  if(sourceBuffer && !sourceBuffer.updating && mediaSource.readyState === 'open') {
    try {
      //remove all upcoming audio from buffer
      sourceBuffer.remove(0, 1000000) 
      
      //reset audio to the last timestamp if buffer exists otherwise shift it to 0th millisec
      audio.currentTime = audio.buffered.length > 0 ? audio.buffered.end(0) : 0
    } catch (e) {
      console.log("Flush failed, ", e)
    }
  }
}

export function signalEndOfStream() {
  if(mediaSource && mediaSource.readyState === 'open') {
    //wait until buffer isnt updating before closing 
    if(!sourceBuffer.updating) {
      mediaSource.endOfStream()
    } else{
      sourceBuffer.addEventListener('updateend', () => {
        if(mediaSource.readyState === 'open') mediaSource.endOfStream()
      }, { once: true })
    }
  }
}