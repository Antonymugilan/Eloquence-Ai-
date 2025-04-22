from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from live_transcriber import LiveSpeechTranscriber
import threading
import time
import torch
import json
import random
import os
from voice import speak, get_audio_data
import requests
from phonem_eval import compare_audio, format_result, compare_pronunciations
from insights_generator import generate_session_insights

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize transcriber
transcriber = LiveSpeechTranscriber()
thread = None

# Load sentences.json
def load_sentences():
    try:
        # Get absolute path to backend directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct path to sentences.json
        json_path = os.path.join(current_dir, 'sentences.json')
        
        # Check if file exists
        if not os.path.exists(json_path):
            print(f"Error: sentences.json not found at {json_path}")
            return None
            
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("Successfully loaded sentences.json")
            return data
            
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading sentences.json: {e}")
        return None

# Get random text based on difficulty level
def get_random_text(level):
    sentences = load_sentences()
    
    if not sentences:
        return "Error loading practice text. Please try again."
    
    try:
        if level == "easy":
            # For easy level, select only from single word categories
            single_word_categories = ["Nouns", "Verbs", "Adjectives", "Adverbs", "Pronouns", "Determiners", "Prepositions", "Conjunctions", "Interjections"]
            category = random.choice(single_word_categories)
            return random.choice(sentences["PartsOfSpeech"][category])
        
        elif level == "medium":
            # For medium level, select from Sentences
            category = random.choice(list(sentences["Sentences"].keys()))
            return random.choice(sentences["Sentences"][category])
        
        elif level == "hard":
            # For hard level, select from ComplicatedSentences
            category = random.choice(list(sentences["ComplicatedSentences"].keys()))
            return random.choice(sentences["ComplicatedSentences"][category])
        
        return "Invalid difficulty level"
    except Exception as e:
        print(f"Error getting random text: {e}")
        return "Error loading practice text. Please try again."

# Add this function to check internet connectivity
def check_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except requests.RequestException:
        return False

# New routes for the sentence selection feature
@app.route('/levels', methods=['GET'])
def get_levels():
    has_internet = check_internet()
    return jsonify({
        'levels': [
            {'id': 'easy', 'name': 'Easy - Parts of Speech'},
            {'id': 'medium', 'name': 'Medium - Sentences'},
            {'id': 'hard', 'name': 'Hard - Complicated Sentences'}
        ],
        'hasInternet': has_internet
    })

# Store reference audio globally
current_reference = {
    'text': None,
    'audio': None,
    'level': None  # Add level to track current difficulty
}

@app.route('/get-text/<level>', methods=['GET'])
def get_text(level):
    global current_reference
    
    # Always get a new random text
    text = get_random_text(level)
    if text:
        try:
            audio_data = get_audio_data(text)
            current_reference = {
                'text': text,
                'audio': audio_data,
                'level': level
            }
            return jsonify({'text': text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid level'}), 400

# Add new route for text-to-speech
@app.route('/speak-text', methods=['POST'])
def speak_text():
    if not check_internet():
        return jsonify({'error': 'No internet connection'}), 503
    
    try:
        text = request.json.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # This will play the audio directly without saving a file
        speak(text)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Print GPU information at startup
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
else:
    print("No GPU available. Using CPU for transcription.")

@socketio.on('start_transcription')
def start_transcription():
    global thread
    
    # Clear any previous transcription
    transcriber.clear_transcription()
    emit('transcription', {'status': 'listening', 'text': ''})

    def run_transcription():
        transcriber.start_transcription()
        last_emitted_text = ""
        
        while transcriber.transcription_active:
            # Get the latest transcription and send it to the client
            text = transcriber.get_latest_transcription()
            if text and text != last_emitted_text:
                socketio.emit('transcription', {'status': 'listening', 'text': text, 'is_final': False})
                last_emitted_text = text
            time.sleep(0.05)  # Reduced delay for faster updates
    
    thread = threading.Thread(target=run_transcription)
    thread.start()

@socketio.on('stop_transcription')
def stop_transcription():
    global current_reference
    emit('transcription', {'status': 'processing', 'text': ''})
    
    # Get the final text and audio
    final_text = transcriber.stop_transcription()
    final_audio = transcriber.get_last_audio()
    
    try:
        if final_audio is not None and current_reference['audio'] is not None:
            # Compare the audio with both reference audio and text
            result = compare_audio(
                current_reference['audio'], 
                final_audio,
                current_reference['text'],
                final_text
            )
            
            # Get text comparison data
            text_comparison = compare_pronunciations(current_reference['text'], final_text)
            
            # Format all metrics for display
            metrics = {
                'text_comparison': {
                    'reference_phonemes': text_comparison.get('reference_phonemes', ''),
                    'user_phonemes': text_comparison.get('user_phonemes', ''),
                    'word_correctness': f"{text_comparison.get('word_correctness', 0):.2f}%",
                    'phoneme_correctness': f"{text_comparison.get('phoneme_correctness', 0):.2f}%",
                    'text_similarity': f"{text_comparison.get('correctness_score', 0):.2f}%"
                },
                'audio_metrics': {
                    'mse_score': f"{result['metrics']['mse_score']:.2f}%",
                    'correlation': f"{result['metrics']['correlation']:.2f}%",
                    'cosine_similarity': f"{result['metrics']['cosine_similarity']:.2f}%"
                },
                'final_score': f"{result['score']:.2f}%"
            }
            
            emit('transcription', {
                'status': 'stopped',
                'text': final_text,
                'is_final': True,
                'metrics': metrics
            })
        else:
            emit('transcription', {
                'status': 'stopped',
                'text': final_text,
                'is_final': True,
                'error': 'Missing reference or user audio'
            })
    except Exception as e:
        print(f"Error in comparison: {e}")
        emit('transcription', {
            'status': 'stopped',
            'text': final_text,
            'is_final': True,
            'error': str(e)
        })

# Add new route to get pronunciation comparison
@app.route('/compare-pronunciation', methods=['POST'])
def compare_pronunciation():
    try:
        data = request.json
        reference_text = data.get('reference_text', '')
        user_text = data.get('user_text', '')
        
        if not reference_text or not user_text:
            return jsonify({'error': 'Both reference and user text are required'}), 400
            
        result = compare_pronunciation(reference_text, user_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a test route to verify the server is running
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Server is running!'})

@app.route('/get-insights', methods=['POST'])
def get_insights():
    try:
        results = request.json.get('results', [])
        print("Received results:", results)  # Debug log
        
        if not results:
            return jsonify({'insights': [], 'error': 'No results provided'}), 400
            
        insights = generate_session_insights(results)
        print("Generated insights:", insights)  # Debug log
        
        return jsonify({'insights': insights})
    except Exception as e:
        print("Error:", str(e))  # Debug log
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Set environment variables for better GPU utilization
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Optimize memory allocation
    
    # Run the app with threading mode for better performance
    socketio.run(app, debug=True, port=5000, use_reloader=False)