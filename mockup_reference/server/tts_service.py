import asyncio
from functools import lru_cache
import re
import html
from google.cloud import texttospeech


@lru_cache(maxsize=1)
def get_tts_client() -> texttospeech.TextToSpeechClient:
    return texttospeech.TextToSpeechClient()


def build_ssml(text: str) -> str:
    text = re.sub(r'\.', '.<break time="300ms"/>', text)
    text = re.sub(r'\?', '?<break time="300ms"/>', text)
    text = re.sub(r'!', '!<break time="300ms"/>', text)
    text = re.sub(r',', ',<break time="150ms"/>', text)

    return f"""
    <speak>
        <prosody rate="95%" pitch="+2%">
            {text}
        </prosody>
    </speak>
    """


def synthesize_sync(text: str) -> bytes:
    """blocking TTS call"""
    clean_text = text.strip()
    if not clean_text:
        return b""

    client = get_tts_client()
    ssml = build_ssml(clean_text)

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(ssml=ssml),
        voice=texttospeech.VoiceSelectionParams(
            language_code="en-IN",
            name="en-IN-Wavenet-D"
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,   #Lower latency in mp3 due to compression
            speaking_rate=1.0,
            pitch=0.0,
        ),
    )
    return response.audio_content


async def synthesize_and_send(text: str, websocket) -> bytes:
    audio = await asyncio.to_thread(synthesize_sync, text)
    if audio:
        await websocket.send_bytes(audio)