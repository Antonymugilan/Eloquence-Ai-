from elevenlabs import ElevenLabs
import io
import pygame

# Initialize client with your API key
client = ElevenLabs(api_key="sk_9e0646009b2451a9f3bb7f0fb9ff31e66eadc1ba600e69e8")

# Initialize pygame mixer
pygame.mixer.init()

def get_audio_data(text):
    """Generate and return audio data from ElevenLabs"""
    try:
        # Convert text to audio (generator object)
        audio_stream = client.text_to_speech.convert(
            voice_id="EXAVITQu4vr4xnSDxMaL",  # "Rachel" voice ID
            model_id="eleven_monolingual_v1",
            text=text,
            output_format="mp3_44100_128"
        )
        
        # Collect all chunks into a single bytes object
        audio_data = b""
        for chunk in audio_stream:
            audio_data += chunk
            
        return audio_data
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        raise

def speak(text):
    """Play the audio for the given text"""
    try:
        # Get the audio data
        audio_data = get_audio_data(text)
        
        # Create a file-like object from the bytes
        audio_file = io.BytesIO(audio_data)
        
        # Load and play the audio
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        raise

if __name__ == "__main__":
    user_input = input("Enter the text you want to speak: ")
    speak(user_input)
