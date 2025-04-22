import torch
import queue
import threading
import time
import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np


class LiveSpeechTranscriber:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphones = sr.Microphone.list_microphone_names()
        self.selected_mic_index = None
        self.transcription_thread = None
        self.transcription_active = False
        self.spoken_text = ""
        self.transcription_queue = queue.Queue()
        self.audio_lock = threading.Lock()
        self.last_device_check = 0
        self.device_check_interval = 5  # Increased to reduce device checks
        self.last_audio_data = None

        # Enhanced GPU detection and usage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            # Set CUDA to use TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cudnn benchmarking for faster training
            torch.backends.cudnn.benchmark = True
        
        # Load Wav2Vec2 model (uses GPU if available)
        self.processor = Wav2Vec2Processor.from_pretrained("./final-wav2vec2-model")
        self.model = Wav2Vec2ForCTC.from_pretrained("./final-wav2vec2-model").to(self.device)
        
        # Optimize model for inference
        self.model.eval()
        if self.device.type == 'cuda':
            # Use TorchScript for faster inference on GPU
            try:
                self.model = torch.jit.script(self.model)
                print("Model successfully converted to TorchScript")
            except Exception as e:
                print(f"TorchScript conversion failed: {e}")
                print("Continuing with regular model")

    def refresh_audio_devices(self):
        current_time = time.time()
        if current_time - self.last_device_check >= self.device_check_interval:
            try:
                with self.audio_lock:
                    self.microphones = sr.Microphone.list_microphone_names()
                self.last_device_check = current_time
                return True
            except Exception as e:
                print(f"[Device Refresh Error] {e}")
        return False

    def select_best_microphone(self):
        self.refresh_audio_devices()

        device_keywords = {
            "Wired Headset": ['headset', 'wired', 'usb', 'earphones'],
            "Bluetooth Device": ['bluetooth', 'wireless', 'buds'],
            "Built-in Microphone": ['built-in', 'internal', 'laptop', 'microphone'],
            "Mobile Microphone": ['mobile', 'phone', 'android', 'ios', 'smartphone']
        }

        def get_mic_index(keywords):
            for index, mic_name in enumerate(self.microphones):
                if any(keyword.lower() in mic_name.lower() for keyword in keywords):
                    return index
            return None

        for device_type, keywords in device_keywords.items():
            mic_index = get_mic_index(keywords)
            if mic_index is not None:
                print(f"[Using {device_type}] {self.microphones[mic_index]} (Index {mic_index})")
                return mic_index

        try:
            default_mic = sr.Microphone().device_index
            print(f"[Using Default Microphone] Index {default_mic}")
            return default_mic
        except Exception as e:
            print(f"[Default Mic Error] {e}")

        if self.microphones:
            print(f"[Using First Available Mic] {self.microphones[0]}")
            return 0

        raise RuntimeError("No microphone found")

    def clear_transcription(self):
        """Clear the current transcription state"""
        self.spoken_text = ""
        # Clear the queue
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except queue.Empty:
                break

    def _transcription_loop(self):
        retry_count = 0
        max_retries = 3
        buffer_text = ""

        while self.transcription_active and retry_count < max_retries:
            try:
                self.selected_mic_index = self.select_best_microphone()
                with sr.Microphone(device_index=self.selected_mic_index, sample_rate=16000) as source:
                    # Reduce ambient noise adjustment time
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    print("[Listening] Start speaking...")

                    while self.transcription_active:
                        try:
                            # Reduce timeout and phrase time limit for faster response
                            audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                            audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0

                            # Store the audio data
                            self.last_audio_data = audio_data
                            
                            # Process audio in smaller chunks for faster response
                            chunk_size = 16000 * 2  # 2 seconds chunks
                            for i in range(0, len(audio_data), chunk_size):
                                chunk = audio_data[i:i + chunk_size]
                                
                                # Move data to GPU if available
                                input_values = self.processor(chunk, return_tensors="pt", sampling_rate=16000).input_values
                                if self.device.type == 'cuda':
                                    input_values = input_values.to(self.device)
                                
                                # Use torch.no_grad() for inference
                                with torch.no_grad():
                                    logits = self.model(input_values).logits
                                
                                # Move results back to CPU for processing
                                predicted_ids = torch.argmax(logits, dim=-1)
                                transcription = self.processor.batch_decode(predicted_ids)[0]
                                
                                if transcription.strip():
                                    buffer_text += " " + self.format_text(transcription)
                                    self.spoken_text = buffer_text.strip()
                                    # Store in queue but don't emit to client
                                    self.transcription_queue.put(self.spoken_text)

                        except sr.WaitTimeoutError:
                            continue
                        except Exception as e:
                            print(f"[Wav2Vec2 Transcription Error] {e}")
                            continue

                retry_count = 0
            except Exception as e:
                retry_count += 1
                print(f"[Retry {retry_count}] Transcription Error: {e}")
                time.sleep(1)

        print("[Transcription Stopped]")

    def start_transcription(self):
        if not self.transcription_active:
            self.transcription_active = True
            self.transcription_thread = threading.Thread(target=self._transcription_loop)
            self.transcription_thread.start()
        else:
            print("[Info] Transcription is already running.")

    def stop_transcription(self):
        if self.transcription_active:
            self.transcription_active = False
            self.transcription_thread.join(timeout=2)
            
            # Get the final text from the queue if available
            final_text = None
            try:
                while not self.transcription_queue.empty():
                    final_text = self.transcription_queue.get_nowait()
            except queue.Empty:
                pass
                
            # If no text in queue, use the current spoken text
            if not final_text:
                final_text = self.spoken_text
                
            return final_text
        return None

    def get_latest_transcription(self):
        try:
            latest_text = None
            while not self.transcription_queue.empty():
                latest_text = self.transcription_queue.get_nowait()
            return latest_text if latest_text else self.spoken_text
        except Exception as e:
            print(f"[Get Transcription Error] {e}")
            return self.spoken_text

    def get_last_audio(self):
        """Return the last recorded audio data"""
        return self.last_audio_data

    @staticmethod
    def format_text(text):
        return text.capitalize() if text else text
