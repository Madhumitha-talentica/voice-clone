import os
import pandas as pd
import random
import argparse
from tqdm import tqdm

def find_transcript(transcript_file, audio_filename):
    """Finds the transcript for a specific audio file within a chapter's transcript file."""
    base_name = os.path.splitext(audio_filename)[0]
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Format is often: 123_456_000001_000000 | 123_456 | "transcript"
                parts = line.strip().split('|')
                if len(parts) >= 3 and parts[0].strip() == base_name:
                    return parts[2].strip().strip('"')
    except FileNotFoundError:
        return None
    return None

def create_benchmark_csv(libritts_path, output_csv, num_samples=50):
    """
    Generates a CSV file for benchmarking from the LibriTTS dataset.

    Args:
        libritts_path (str): The root path to the extracted LibriTTS dataset 
                             (e.g., '.../LibriTTS/test-clean').
        output_csv (str): Path to save the generated CSV file.
        num_samples (int): The number of sample pairs to generate.
    """
    print(f"Scanning LibriTTS directory at: {libritts_path}")
    if not os.path.isdir(libritts_path):
        print(f"Error: Directory not found at {libritts_path}")
        print("Please download and extract the LibriTTS 'test-clean' or 'dev-clean' set and provide the correct path.")
        return

    # --- 1. Discover all speakers and their utterances ---
    speaker_files = {}
    for speaker_id in os.listdir(libritts_path):
        speaker_path = os.path.join(libritts_path, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
        
        speaker_files[speaker_id] = []
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
            
            # Find the transcript file for the chapter
            # It's usually named speaker_id-chapter_id.trans.tsv or similar
            transcript_file = None
            for f in os.listdir(chapter_path):
                if f.endswith('.tsv'):
                    transcript_file = os.path.join(chapter_path, f)
                    break
            
            if transcript_file is None:
                continue

            for filename in os.listdir(chapter_path):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(chapter_path, filename)
                    transcript = find_transcript(transcript_file, filename)
                    if transcript:
                        speaker_files[speaker_id].append({
                            "audio_path": audio_path,
                            "transcript": transcript
                        })

    speaker_ids = list(speaker_files.keys())
    if len(speaker_ids) < 2:
        print("Error: Fewer than 2 speakers found. Cannot create source/target pairs.")
        return

    print(f"Found {len(speaker_ids)} speakers.")

    # --- 2. Create benchmark pairs ---
    benchmark_data = []
    print(f"Generating {num_samples} benchmark samples...")
    
    with tqdm(total=num_samples) as pbar:
        while len(benchmark_data) < num_samples:
            try:
                # Pick a random source speaker and a different target speaker
                source_speaker_id, target_speaker_id = random.sample(speaker_ids, 2)

                # Pick a random utterance from the source speaker
                source_utterance = random.choice(speaker_files[source_speaker_id])
                
                # Pick a random utterance from the target speaker (for their voice timbre)
                target_utterance = random.choice(speaker_files[target_speaker_id])

                benchmark_data.append({
                    "source_audio": source_utterance["audio_path"],
                    "target_audio": target_utterance["audio_path"],
                    "ground_truth_transcript": source_utterance["transcript"]
                })
                pbar.update(1)
            except (IndexError, ValueError):
                # This can happen if a speaker has no valid utterances
                continue

    # --- 3. Save to CSV ---
    df = pd.DataFrame(benchmark_data)
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully created benchmark file at: {output_csv}")
    print(f"Total samples: {len(df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare benchmark data from LibriTTS.")
    parser.add_argument(
        "--libritts_path", 
        type=str, 
        required=True, 
        help="Path to the root of the LibriTTS dataset (e.g., 'D:/datasets/LibriTTS/test-clean')."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="benchmark_data.csv", 
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=50, 
        help="Number of benchmark samples to generate."
    )
    args = parser.parse_args()

    create_benchmark_csv(args.libritts_path, args.output_csv, args.num_samples)
