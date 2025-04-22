import librosa
from phonemizer import phonemize
from Levenshtein import distance as levenshtein_distance
import numpy as np
from typing import Tuple
from dtaidistance import dtw
import io
import soundfile as sf
from pydub import AudioSegment
import tempfile

def transcribe_to_text(audio_path):
    """
    Dummy transcription function. Replace with your ASR model or Whisper for real transcription.
    """
    # For now, simulate with dummy outputs (use Whisper or ASR here)
    if 'ref' in audio_path:
        return "cat is sleeping"
    else:
        return "kat iz sleeping"

def text_to_phonemes(text: str) -> str:
    """Convert text to phonemes using basic phoneme mapping"""
    try:
        # Simple mapping of common English phonemes
        phoneme_map = {
            'a': 'ah', 'e': 'eh', 'i': 'iy', 'o': 'ow', 'u': 'uw',
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
            'h': 'h', 'j': 'jh', 'k': 'k', 'l': 'l', 'm': 'm',
            'n': 'n', 'p': 'p', 'q': 'k', 'r': 'r', 's': 's',
            't': 't', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'y',
            'z': 'z', 'th': 'th', 'ch': 'ch', 'sh': 'sh',
            'ing': 'ihng', 'tion': 'shun', 'sion': 'zhun',
            'are': 'ahr', 'air': 'ehr', 'ear': 'ihr',
            'the': 'dhah', 'and': 'ahnd', 'for': 'fohr'
        }
        
        # Convert text to lowercase and split into words
        words = text.lower().split()
        phonemes = []
        
        for word in words:
            word_phonemes = []
            i = 0
            while i < len(word):
                # Try to match longer sequences first
                found = False
                for length in range(4, 1, -1):  # Try sequences up to 4 characters
                    if i + length <= len(word):
                        seq = word[i:i+length]
                        if seq in phoneme_map:
                            word_phonemes.append(phoneme_map[seq])
                            i += length
                            found = True
                            break
                
                # If no longer sequence found, try single character
                if not found:
                    char = word[i]
                    if char in phoneme_map:
                        word_phonemes.append(phoneme_map[char])
                    else:
                        word_phonemes.append(char)  # Keep original character if no mapping
                    i += 1
            
            phonemes.extend(word_phonemes)
            phonemes.append(' ')  # Add space between words
        
        return ' '.join(phonemes).strip()
        
    except Exception as e:
        print(f"Error in phoneme conversion: {e}")
        # If conversion fails, return the original text
        return text

def compute_per(ref_phonemes: str, hyp_phonemes: str):
    """Compute Phoneme Error Rate between reference and hypothesis phonemes"""
    # Split into individual phonemes
    ref_phones = ref_phonemes.split()
    hyp_phones = hyp_phonemes.split()
    
    if not ref_phones:
        return 1.0, 0.0
    
    # Calculate Levenshtein distance
    distance = levenshtein_distance(ref_phonemes, hyp_phonemes)
    
    # Calculate PER
    per = distance / len(ref_phones)
    
    # Calculate correctness score (0-100%)
    correctness = max(0, min(100, (1 - per) * 100))
    
    return per, correctness

def compare_pronunciations(reference_text: str, user_text: str) -> dict:
    """Compare the pronunciation of reference text with user's spoken text"""
    try:
        print(f"\nComparing pronunciations:")
        print(f"Reference text: {reference_text}")
        print(f"User text: {user_text}")
        
        # Convert texts to lowercase and split into words
        ref_words = reference_text.lower().split()
        user_words = user_text.lower().split()
        
        # Calculate word-level similarity
        word_matches = 0
        total_words = len(ref_words)
        
        # Check each word in reference against user's words
        for ref_word in ref_words:
            # Check for exact match
            if ref_word in user_words:
                word_matches += 1
            else:
                # Check for similar words using Levenshtein distance
                for user_word in user_words:
                    if levenshtein_distance(ref_word, user_word) <= 2:  # Allow small differences
                        word_matches += 1
                        break
        
        # Calculate word-level correctness
        word_correctness = (word_matches / total_words) * 100
        
        # Convert texts to phonemes for detailed comparison
        ref_phonemes = text_to_phonemes(reference_text.lower())
        user_phonemes = text_to_phonemes(user_text.lower())
        
        print(f"Reference phonemes: {ref_phonemes}")
        print(f"User phonemes: {user_phonemes}")
        
        # Calculate PER and phoneme-level correctness
        per, phoneme_correctness = compute_per(ref_phonemes, user_phonemes)
        
        # Combine word-level and phoneme-level scores
        # Weight word-level more heavily (60%) than phoneme-level (40%)
        final_correctness = (word_correctness * 0.6) + (phoneme_correctness * 0.4)
        
        print(f"Word-level correctness: {word_correctness:.2f}%")
        print(f"Phoneme-level correctness: {phoneme_correctness:.2f}%")
        print(f"Final correctness score: {final_correctness:.2f}%")
        
        return {
            'per': float(per),
            'correctness_score': float(final_correctness),
            'word_correctness': float(word_correctness),
            'phoneme_correctness': float(phoneme_correctness),
            'reference_phonemes': ref_phonemes,
            'user_phonemes': user_phonemes
        }
    except Exception as e:
        print(f"Error in pronunciation comparison: {e}")
        return {
            'error': str(e),
            'per': 1.0,
            'correctness_score': 0.0,
            'word_correctness': 0.0,
            'phoneme_correctness': 0.0
        }

def extract_features(audio_data):
    """Extract audio features from either bytes (ElevenLabs) or numpy array (live audio)"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            if isinstance(audio_data, bytes):
                # Handle ElevenLabs audio (MP3)
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
                audio_segment = audio_segment.set_channels(1)  # Convert to mono
                audio_segment = audio_segment.set_frame_rate(16000)  # Set sample rate
                audio_segment.export(temp_file.name, format='wav')
            else:
                # Handle live audio (numpy array)
                # Normalize the audio
                audio_data = audio_data / np.max(np.abs(audio_data))
                sf.write(temp_file.name, audio_data, 16000)

            # Load and process audio
            y, sr = librosa.load(temp_file.name, sr=16000, mono=True)
            
            # Normalize audio
            y = librosa.util.normalize(y)
            
            # Extract features
            # MFCCs for overall sound
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Spectral features for pronunciation characteristics
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Combine features
            features = np.vstack([
                mfcc,
                spectral_centroid,
                spectral_rolloff
            ])
            
            # Normalize features
            features = (features - np.mean(features)) / np.std(features)
            
            return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def calculate_similarity(ref_features, user_features):
    """Calculate similarity between two sets of features"""
    try:
        # Make sure features have the same length
        min_length = min(ref_features.shape[1], user_features.shape[1])
        ref_features = ref_features[:, :min_length]
        user_features = user_features[:, :min_length]

        # Calculate different similarity metrics
        # 1. Mean Squared Error
        mse = np.mean((ref_features - user_features) ** 2)
        
        # 2. Correlation coefficient
        correlation = np.corrcoef(ref_features.flatten(), user_features.flatten())[0, 1]
        
        # 3. Cosine similarity
        cos_sim = np.dot(ref_features.flatten(), user_features.flatten()) / (
            np.linalg.norm(ref_features.flatten()) * np.linalg.norm(user_features.flatten())
        )

        # Combine metrics into a final score
        # Convert MSE to similarity (0-1 range)
        mse_score = np.exp(-mse/100)  # Exponential decay
        
        # Ensure correlation and cosine similarity are positive
        correlation = max(0, correlation)
        cos_sim = max(0, cos_sim)
        
        # Weighted combination of scores
        final_score = (0.4 * mse_score + 0.3 * correlation + 0.3 * cos_sim) * 100
        
        return final_score, {
            'mse': float(mse),
            'mse_score': float(mse_score * 100),
            'correlation': float(correlation * 100),
            'cosine_similarity': float(cos_sim * 100)
        }

    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0, {}

def compare_audio(reference_audio, user_audio, reference_text=None, user_text=None):
    """Compare reference audio with user audio"""
    try:
        # Validate inputs
        if not reference_text or not user_text:
            raise ValueError("Both reference_text and user_text must be provided")

        print(f"\nText Comparison:")
        print(f"Reference: '{reference_text}'")
        print(f"User said: '{user_text}'")

        # First do text comparison
        text_comparison = compare_pronunciations(reference_text, user_text)
        text_score = text_comparison['correctness_score']
        
        print(f"Text similarity score: {text_score:.2f}%")
        
        # If texts are too different, return immediately with low score
        if text_score < 30:  # Very strict threshold
            print("Texts are too different - returning low score")
            return {
                'score': text_score,
                'error': 'Texts are too different',
                'metrics': {
                    'mse_score': 0,
                    'correlation': 0,
                    'cosine_similarity': 0
                }
            }

        # Only proceed with audio comparison if texts are similar enough
        ref_features = extract_features(reference_audio)
        user_features = extract_features(user_audio)

        if ref_features is None or user_features is None:
            raise ValueError("Failed to extract features from audio")

        # Calculate audio similarity
        audio_score, metrics = calculate_similarity(ref_features, user_features)

        print("\nAudio Metrics:")
        print(f"MSE Score: {metrics['mse_score']:.2f}%")
        print(f"Correlation: {metrics['correlation']:.2f}%")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.2f}%")

        # Calculate final score with heavy weight on text similarity
        final_score = (text_score * 0.8) + (audio_score * 0.2)
        print(f"Final combined score: {final_score:.2f}%")

        return {
            'score': float(final_score),
            'metrics': metrics,
            'text_score': text_score,
            'audio_score': audio_score
        }

    except Exception as e:
        print(f"Error in comparison: {e}")
        return {
            'score': 0,
            'error': str(e)
        }

def format_result(result):
    """Format the comparison result for display"""
    if 'error' in result:
        return f"Error in comparison: {result['error']}"
    
    metrics = result.get('metrics', {})
    return f"""
Pronunciation Score: {result['score']:.2f}%

Detailed Metrics:
- Acoustic Similarity: {metrics.get('mse_score', 0):.2f}%
- Pattern Matching: {metrics.get('correlation', 0):.2f}%
- Sound Alignment: {metrics.get('cosine_similarity', 0):.2f}%
"""

def main():
    # Load and transcribe both audio files
    ref_text = transcribe_to_text("ref_audio.wav")
    hyp_text = transcribe_to_text("test_audio.wav")

    print("Ref Text:", ref_text)
    print("Hyp Text:", hyp_text)

    # Convert both texts to phonemes
    ref_phonemes = text_to_phonemes(ref_text)
    hyp_phonemes = text_to_phonemes(hyp_text)

    print("Ref Phonemes:", ref_phonemes)
    print("Hyp Phonemes:", hyp_phonemes)

    # Compute PER
    per, correctness = compute_per(ref_phonemes, hyp_phonemes)

    print(f"\nPhoneme Error Rate (PER): {per * 100:.2f}%")
    print(f"Correctness Score: {correctness:.2f}%")

# Test function
if __name__ == "__main__":
    # Create test data
    duration = 1
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    
    # Generate two similar test signals
    test_ref = np.sin(2 * np.pi * 440 * t)  # 440 Hz
    test_user = np.sin(2 * np.pi * 445 * t)  # 445 Hz
    
    # Test comparison
    result = compare_audio(test_ref, test_user)
    print(format_result(result))
