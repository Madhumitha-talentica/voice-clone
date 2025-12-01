import torch
import torchaudio
import whisper
import jiwer
from speechbrain.inference.speaker import SpeakerRecognition
import os
import sys
import pandas as pd
import soundfile as sf

# Add the parent directory to the path to allow importing 'openvoice'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openvoice.api import ToneColorConverter


# ========================================================================================
# 1. Speaker Verification (Speaker Similarity)
# ========================================================================================

class SpeakerSimilarity:
    """
    A class to measure the similarity between two speakers in different audio files.
    It uses a pre-trained model from SpeechBrain to create speaker embeddings
    and then calculates the cosine similarity between them.
    """
    def __init__(self, device=None):
        """
        Initializes the SpeakerRecognition model.
        
        Args:
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). 
                                    Defaults to 'cuda' if available.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="../pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": self.device}
            )
        except Exception as e:
            print(f"Error loading SpeechBrain model: {e}")
            print("Please ensure you have a working internet connection to download the model.")
            self.model = None

    def calculate_similarity(self, file1_path, file2_path):
        """
        Calculates the cosine similarity between the speakers in two audio files.

        Args:
            file1_path (str): Path to the first audio file.
            file2_path (str): Path to the second audio file.

        Returns:
            float: A similarity score between -1 and 1. Higher is more similar.
                   Returns -1.0 if the model failed to load or files are not found.
        """
        if not self.model:
            return -1.0
            
        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            print(f"Error: One or both audio files not found ({file1_path}, {file2_path}).")
            return -1.0

        # Use SpeechBrain's verification API for compatibility across versions
        try:
            if hasattr(self.model, "verify_files"):
                score, prediction = self.model.verify_files(file1_path, file2_path)
                return float(score)
            # Some versions expose 'verify_paths' instead
            if hasattr(self.model, "verify_paths"):
                score, prediction = self.model.verify_paths(file1_path, file2_path)
                return float(score)
        except Exception as e:
            print(f"Speaker verification failed: {e}")
            return -1.0

        print("Error: SpeakerRecognition model does not support a verification API (verify_files/verify_paths).")
        return -1.0

# ========================================================================================
# 2. ASR (Content Preservation)
# ========================================================================================

class ContentPreservation:
    """
    A class to evaluate the content preservation of converted audio using ASR.
    It uses OpenAI's Whisper model to transcribe the audio and then calculates
    the Word Error Rate (WER) against a ground truth transcript.
    """
    def __init__(self, model_size="base", device=None):
        """
        Initializes the Whisper ASR model.

        Args:
            model_size (str, optional): The size of the Whisper model to use. 
                                        Defaults to "base".
            device (str, optional): The device to run the model on ('cuda' or 'cpu'). 
                                    Defaults to 'cuda' if available.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = whisper.load_model(model_size, device=self.device)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Please ensure you have ffmpeg installed (`pip install ffmpeg-python`).")
            self.model = None

    def calculate_wer(self, audio_path, ground_truth_text):
        """
        Transcribes an audio file and calculates the Word Error Rate (WER).

        Args:
            audio_path (str): Path to the audio file to be evaluated.
            ground_truth_text (str): The correct transcript for the audio.

        Returns:
            float: The Word Error Rate (0.0 is a perfect match).
                   Returns 1.0 if the model failed or the file is not found.
        """
        if not self.model:
            return 1.0
            
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return 1.0

        # Transcribe audio
        result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())
        transcribed_text = result['text']

        # Calculate WER
        wer = jiwer.wer(ground_truth_text, transcribed_text)
        
        print(f" - Ground Truth: {ground_truth_text}")
        print(f" - Transcribed:  {transcribed_text}")
        
        return wer

# ========================================================================================
# Main Batch Benchmarking Function
# ========================================================================================

def run_batch_benchmark(csv_path, config_path, ckpt_path, output_dir="outputs", batch_size=None):
    """
    Runs a full benchmark on a batch of audio files defined in a CSV.

    Args:
        csv_path (str): Path to the CSV file with benchmark data.
        config_path (str): Path to the model's config file.
        ckpt_path (str): Path to the model's checkpoint file.
        output_dir (str, optional): Directory to save converted audio. Defaults to "outputs".
        batch_size (int, optional): Number of rows to process from the CSV. Defaults to all.
    """
    print("="*80)
    print("Starting Batch Evaluation Benchmark")
    print("="*80)

    # --- 1. Load Models ---
    print("\n[1/4] Loading all necessary models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Tone Color Converter
    try:
        converter = ToneColorConverter(config_path, device=device)
        converter.load_ckpt(ckpt_path)
        print("ToneColorConverter loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load ToneColorConverter. Error: {e}")
        print("Please ensure the config and checkpoint paths are correct.")
        return

    # Load Evaluation Models
    speaker_eval = SpeakerSimilarity(device=device)
    content_eval = ContentPreservation(device=device)
    
    if not speaker_eval.model:
        print("WARNING: Speaker similarity model failed to load; similarity will be skipped.")
    if not content_eval.model:
        print("WARNING: ASR (Whisper) model failed to load; WER will be skipped.")
    if not speaker_eval.model and not content_eval.model:
        print("FATAL: No evaluation models available. Aborting.")
        return

    # --- 2. Read and Prepare Data ---
    print("\n[2/4] Reading and preparing data from CSV...")
    if not os.path.exists(csv_path):
        print(f"FATAL: CSV file not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if batch_size:
        df = df.head(batch_size)
    print(f"Found {len(df)} tasks to process.")
    
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # --- 3. Process Each Task ---
    print("\n[3/4] Processing conversion and evaluation tasks...")
    for index, row in df.iterrows():
        source_audio = row['source_audio']
        target_audio = row['target_audio']
        ground_truth = row['ground_truth_transcript']
        
        print(f"\n--- Task {index + 1}/{len(df)} ---")
        print(f"Source: {source_audio}, Target: {target_audio}")

        if not os.path.exists(source_audio) or not os.path.exists(target_audio):
            print("WARNING: Skipping task due to missing source or target audio file.")
            continue

        # Define output path for the converted audio
        base_name = os.path.basename(source_audio).replace('.wav', '')
        converted_audio_path = os.path.join(output_dir, f"{base_name}_converted.wav")

        # A. Run Voice Conversion
        print(" - Running voice conversion...")
        # Extract speaker embedding (SE) from source and target speakers
        # ToneColorConverter.extract_se returns a single tensor (embedding)
        target_se = converter.extract_se(target_audio, se_save_path=None)
        src_se = converter.extract_se(source_audio, se_save_path=None)
        # Convert the source audio with the target's tone color
        converter.convert(
            audio_src_path=source_audio, 
            src_se=src_se, 
            tgt_se=target_se, 
            output_path=converted_audio_path,
            tau=0.3 # A parameter for the conversion model
        )
        print(f" - Converted audio saved to {converted_audio_path}")

        # B. Evaluate Speaker Similarity (if available)
        similarity_score = None
        if speaker_eval.model:
            print(" - Evaluating speaker similarity...")
            similarity_score = speaker_eval.calculate_similarity(converted_audio_path, target_audio)
            print(f" - Similarity Score: {similarity_score:.4f}")
        else:
            print(" - Skipping speaker similarity (model unavailable).")

        # C. Evaluate Content Preservation (if available)
        wer_score = None
        if content_eval.model:
            print(" - Evaluating content preservation (WER)...")
            wer_score = content_eval.calculate_wer(converted_audio_path, ground_truth)
            print(f" - Word Error Rate (WER): {wer_score:.4f}")
        else:
            print(" - Skipping WER evaluation (ASR model unavailable).")
        
        results.append({
            "source": source_audio,
            "target": target_audio,
            "similarity": similarity_score,
            "wer": wer_score
        })

    # --- 4. Aggregate and Display Results ---
    print("\n[4/4] Aggregating results...")
    print("\n" + "="*80)
    print("Benchmark Complete: Final Results")
    print("="*80)

    if not results:
        print("No tasks were processed. Please check your CSV and file paths.")
        return

    # Create a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    
    # Calculate Averages (ignore None values)
    if 'similarity' in results_df.columns:
        avg_similarity = results_df['similarity'].dropna().mean()
    else:
        avg_similarity = None
    if 'wer' in results_df.columns:
        avg_wer = results_df['wer'].dropna().mean()
    else:
        avg_wer = None

    print("\n--- Individual Results ---")
    print(results_df.to_string(index=False))
    
    print("\n--- Average Scores ---")
    if avg_similarity is not None:
        print(f"Average Speaker Similarity: {avg_similarity:.4f}")
    else:
        print("Average Speaker Similarity: n/a (model unavailable)")
    if avg_wer is not None:
        print(f"Average Word Error Rate (WER): {avg_wer:.4f}")
    else:
        print("Average Word Error Rate (WER): n/a (model unavailable)")
    print("\nReminder: Higher similarity is better. Lower WER is better.")
    print("="*80)


def setup_dummy_environment():
    """Creates dummy audio files and directories for demonstration."""
    print("Setting up dummy audio files and directories for demonstration...")
    
    # Create dummy directories
    os.makedirs("test_audio/source", exist_ok=True)
    os.makedirs("test_audio/target", exist_ok=True)
    
    # Dummy audio generation
    sample_rate = 24000  # OpenVoice models often use 24kHz
    duration = 3
    t = torch.linspace(0., duration, int(sample_rate * duration), dtype=torch.float32)
    
    # Source files
    sf.write("test_audio/source/source1.wav", torch.sin(2 * torch.pi * 440 * t).numpy(), sample_rate)
    sf.write("test_audio/source/source2.wav", torch.sin(2 * torch.pi * 550 * t).numpy(), sample_rate)
    
    # Target files
    sf.write("test_audio/target/target1.wav", torch.sin(2 * torch.pi * 660 * t).numpy(), sample_rate)
    sf.write("test_audio/target/target2.wav", torch.sin(2 * torch.pi * 770 * t).numpy(), sample_rate)
    
    print("Dummy files created in 'test_audio' directory.")
    print("NOTE: The results will be nonsensical with dummy tones.")
    print("Replace the dummy files and CSV with your actual data for meaningful results.\n")


if __name__ == '__main__':
    # --- Configuration ---
    # Create dummy files for a first-time run.
    # In a real scenario, you would have your own data and can comment this out.
    setup_dummy_environment()

    # IMPORTANT: You MUST provide the correct paths to your model's config and checkpoint.
    # These are placeholders and will likely cause an error.
    # Find the correct paths in the OpenVoice repository, typically in 'checkpoints/converter/'.
    converter_config_path = "../checkpoints/converter/config.json"
    converter_ckpt_path = "../checkpoints/converter/checkpoint.pth"

    # --- Run the Batch Benchmark ---
    run_batch_benchmark(
        csv_path="benchmark_data.csv",
        config_path=converter_config_path,
        ckpt_path=converter_ckpt_path,
        batch_size=None  # Run all tasks in the CSV
    )

