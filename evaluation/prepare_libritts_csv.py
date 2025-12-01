import os
import pandas as pd
import random
import argparse
from tqdm import tqdm

def find_transcript(chapter_path, audio_filename):
    """
    Finds the transcript for a specific audio file within a chapter's transcript file.
    This is made more robust to handle different naming conventions.
    """
    base_name = os.path.splitext(audio_filename)[0]
    
    # Find the transcript file, which could have different extensions
    transcript_file = None
    for f in os.listdir(chapter_path):
        if f.endswith(('.tsv', '.txt')) and 'trans' in f:
            transcript_file = os.path.join(chapter_path, f)
            break
    
    if not transcript_file:
        return None

    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Handle both tab-separated and pipe-separated formats
                parts = [p.strip() for p in line.strip().replace('|', '\t').split('\t') if p.strip()]
                
                if not parts:
                    continue
                
                # The audio file name is usually the first part
                file_id = parts[0]
                
                if file_id == base_name:
                    # The transcript is usually the last part
                    transcript = parts[-1]
                    # Clean up quotes
                    return transcript.strip().strip('"')
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
    print("Discovering speakers and utterances...")
    all_speaker_dirs = [os.path.join(libritts_path, d) for d in os.listdir(libritts_path) if os.path.isdir(os.path.join(libritts_path, d))]

    for speaker_path in tqdm(all_speaker_dirs, desc="Scanning Speakers"):
        speaker_id = os.path.basename(speaker_path)
        
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue

            for filename in os.listdir(chapter_path):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(chapter_path, filename)
                    transcript = find_transcript(chapter_path, filename)
                    if transcript:
                        if speaker_id not in speaker_files:
                            speaker_files[speaker_id] = []
                        speaker_files[speaker_id].append({
                            "audio_path": audio_path,
                            "transcript": transcript
                        })

    # Filter out speakers with no valid utterances
    valid_speaker_ids = [sid for sid, files in speaker_files.items() if files]
    
    if len(valid_speaker_ids) < 2:
        print("Error: Fewer than 2 speakers with valid utterances found. Cannot create source/target pairs.")
        print(f"Found {len(speaker_files)} total speaker folders, but only {len(valid_speaker_ids)} had usable content.")
        return

    print(f"Found {len(valid_speaker_ids)} speakers with valid audio and transcripts.")

    # --- 2. Create benchmark pairs ---
    benchmark_data = []
    print(f"Generating {num_samples} benchmark samples...")
    
    with tqdm(total=num_samples, desc="Generating Samples") as pbar:
        attempts = 0
        while len(benchmark_data) < num_samples and attempts < num_samples * 100:
            attempts += 1
            try:
                # Pick a random source speaker and a different target speaker
                source_speaker_id, target_speaker_id = random.sample(valid_speaker_ids, 2)

                # Pick a random utterance from the source speaker
                source_utterance = random.choice(speaker_files[source_speaker_id])
                
                # Pick a random utterance from the target speaker (for their voice timbre)
                target_utterance = random.choice(speaker_files[target_speaker_id])

                # Check for duplicates
                is_duplicate = False
                for item in benchmark_data:
                    if item['source_audio'] == source_utterance['audio_path'] and item['target_audio'] == target_utterance['audio_path']:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue

                benchmark_data.append({
                    "source_audio": source_utterance["audio_path"],
                    "target_audio": target_utterance["audio_path"],
                    "ground_truth_transcript": source_utterance["transcript"]
                })
                pbar.update(1)
            except (IndexError, ValueError):
                # This can happen if a speaker has no valid utterances, though we filtered already.
                continue
    
    if len(benchmark_data) < num_samples:
        print(f"\nWarning: Could only generate {len(benchmark_data)} samples out of the requested {num_samples}.")
        print("This might happen if the dataset is very small or speakers have few utterances.")

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
