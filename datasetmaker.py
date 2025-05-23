import os
from pathlib import Path
import subprocess
import logging
import sys
import pandas as pd
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch
import unicodedata
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import hashlib

# Configure global logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_maker.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DatasetMaker:
    def __init__(self, max_duration: float = 10.0, train_split: float = 0.8, max_workers: int = 4):
        self.max_duration = max_duration
        self.train_split = train_split
        self.max_workers = max_workers
        self.model = None
        self.logger = None

    def _setup_dataset_logger(self, dataset_dir: Path, dataset_index: int):
        """Set up a logger for a specific dataset."""
        log_file = dataset_dir / f"dataset-{dataset_index}.log"
        self.logger = logging.getLogger(f"Dataset-{dataset_index}")
        self.logger.setLevel(logging.INFO)
        # Remove existing handlers to avoid duplicate logging
        self.logger.handlers = []
        # Add file handler for dataset-specific log
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def _create_directory_structure(self, output_dir: Path):
        """Create dataset directory structure."""
        directories = [
            output_dir / "audio",
            output_dir / "vocals" / "temp_chunks",
            output_dir / "vocals" / "processed",
            output_dir / "wavs",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _remove_directory(self, dir_path: Path):
        """Recursively delete a directory and its contents."""
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                self.logger.info(f"Deleted directory {dir_path}")
        except Exception as e:
            self.logger.warning(f"Error deleting directory {dir_path}: {e}")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Shorten and sanitize filename using a hash to avoid long paths."""
        filename_hash = hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]
        short_name = filename[:20] if len(filename) > 20 else filename
        short_name = short_name.replace('đ', 'd').replace('Đ', 'D')
        normalized = ''.join(c for c in unicodedata.normalize('NFD', short_name)
                             if unicodedata.category(c) != 'Mn')
        sanitized = re.sub(r'[^a-zA-Z0-9.-]', '_', normalized)
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return f"{sanitized}_{filename_hash}"

    def extract_audio_from_video(self, input_path: Path, output_dir: Path, sanitized_stem: str) -> Path:
        """Extract audio from video file."""
        try:
            video = VideoFileClip(str(input_path))
            audio_path = output_dir / "audio" / f"{sanitized_stem}.mp3"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            video.audio.write_audiofile(str(audio_path))
            video.close()
            self.logger.info(f"Extracted audio from {input_path} to {audio_path}")
            return audio_path
        except Exception as e:
            self.logger.error(f"Error extracting audio from {input_path}: {str(e)}")
            raise

    def _process_demucs_chunk(self, chunk_path: Path, processed_dir: Path, model: str = "htdemucs") -> Path:
        """Process a single audio chunk with Demucs."""
        try:
            output_model_dir = processed_dir / model / chunk_path.stem
            output_model_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, "-m", "demucs",
                "--mp3",
                "-n", model,
                "--shifts=1",
                "-o", str(processed_dir),
                str(chunk_path)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='replace'
            )
            self.logger.info(f"Demucs output for {chunk_path} (model: {model}): {result.stdout}")
            vocals_path = processed_dir / model / chunk_path.stem / "vocals.mp3"
            if vocals_path.exists():
                return vocals_path
            else:
                raise FileNotFoundError(f"Vocals track not found for {chunk_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Demucs command failed for {chunk_path} (model: {model}): {e.stderr}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_path} with {model}: {str(e)}")
            raise

    def remove_background_music(self, input_path: Path, output_dir: Path, sanitized_stem: str) -> Path:
        """Remove background music using Demucs."""
        chunk_dir = output_dir / "vocals" / "temp_chunks"
        processed_dir = output_dir / "vocals" / "processed"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        try:
            audio = AudioSegment.from_file(input_path).set_frame_rate(44100).set_channels(2)
            chunk_length_ms = 60 * 1000
            min_chunk_duration_ms = 1000
            chunk_paths = []
            total_duration_ms = len(audio)

            num_chunks = max(1, (total_duration_ms + chunk_length_ms - 1) // chunk_length_ms)
            adjusted_chunk_length_ms = total_duration_ms // num_chunks if num_chunks > 1 else total_duration_ms

            for i in range(0, total_duration_ms, adjusted_chunk_length_ms):
                chunk = audio[i:i + adjusted_chunk_length_ms]
                if len(chunk) < min_chunk_duration_ms:
                    chunk = chunk + AudioSegment.silent(duration=min_chunk_duration_ms - len(chunk))
                chunk_path = chunk_dir / f"{sanitized_stem}_chunk_{i//adjusted_chunk_length_ms:04d}.mp3"
                chunk.export(chunk_path, format="mp3")
                chunk_paths.append(chunk_path)
                self.logger.info(f"Created chunk {chunk_path} (duration: {len(chunk)/1000}s, channels: {chunk.channels}, sample_rate: {chunk.frame_rate}Hz)")

            vocal_chunks = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for model in ["htdemucs", "mdx_extra"]:
                    future_to_chunk = {executor.submit(self._process_demucs_chunk, chunk_path, processed_dir, model): chunk_path for chunk_path in chunk_paths}
                    for future in as_completed(future_to_chunk):
                        chunk_path = future_to_chunk[future]
                        try:
                            vocals_path = future.result()
                            vocal_chunks.append(vocals_path)
                        except Exception as e:
                            self.logger.warning(f"Skipping chunk {chunk_path} with {model}: {str(e)}")
                            continue
                    if vocal_chunks:
                        break
                else:
                    raise RuntimeError("No vocal chunks processed")

            if not vocal_chunks:
                raise RuntimeError("No vocal chunks processed")

            vocal_chunks.sort(key=lambda x: int(re.search(r'_chunk_(\d+)', str(x)).group(1)))
            combined_vocals = AudioSegment.empty()
            for vocal_path in vocal_chunks:
                vocal_audio = AudioSegment.from_file(vocal_path)
                combined_vocals += vocal_audio
            output_vocals_path = output_dir / "vocals" / f"{sanitized_stem}_vocals.mp3"
            combined_vocals.export(output_vocals_path, format="mp3")
            self.logger.info(f"Combined vocals saved to {output_vocals_path}")
            return output_vocals_path
        except Exception as e:
            self.logger.error(f"Error removing background music: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()

    def split_into_sentences(self, input_path: Path, output_dir: Path) -> list:
        """Split audio into sentence-based chunks and name them chunk_{id}.wav."""
        try:
            audio = AudioSegment.from_file(input_path)
            chunks = split_on_silence(audio, min_silence_len=150, silence_thresh=-45, keep_silence=400)
            segment_paths = []
            segment_dir = output_dir / "wavs"
            current_chunk = AudioSegment.empty()
            chunk_index = 0

            for chunk in chunks:
                if current_chunk.duration_seconds + chunk.duration_seconds <= self.max_duration:
                    current_chunk += chunk
                else:
                    if current_chunk.duration_seconds > 0:
                        segment_path = segment_dir / f"chunk_{chunk_index:04d}.wav"
                        current_chunk.export(segment_path, format="wav")
                        segment_paths.append(segment_path)
                        chunk_index += 1
                    current_chunk = chunk

            if current_chunk.duration_seconds > 0:
                segment_path = segment_dir / f"chunk_{chunk_index:04d}.wav"
                current_chunk.export(segment_path, format="wav")
                segment_paths.append(segment_path)

            self.logger.info(f"Split audio into {len(segment_paths)} segments")
            return segment_paths
        except Exception as e:
            self.logger.error(f"Error splitting audio {input_path}: {str(e)}")
            raise

    def transcribe_segment(self, input_paths: list, output_dir: Path) -> list:
        """Transcribe audio segments using Whisper."""
        if self.model is None:
            self.logger.info("Loading Whisper model large-v3")
            self.model = whisper.load_model("large-v3")
            torch.cuda.empty_cache()

        metadata = []
        for segment_path in input_paths:
            self.logger.info(f"Transcribing segment {segment_path}")
            try:
                result = self.model.transcribe(str(segment_path), language="vi")
                segments = result.get("segments", [])
                confidence = np.mean([seg.get("confidence", 0.0) for seg in segments]) if segments else 0.0
                metadata.append({
                    "audio_file": f"wavs/{segment_path.name}",
                    "text": result["text"].lower(),
                    "language": "vi"
                })
            except Exception as e:
                self.logger.error(f"Error transcribing segment {segment_path}: {str(e)}")
        self.logger.info(f"Transcribed {len(metadata)} segments")
        return metadata

    def _save_metadata(self, split: str, metadata: list, output_dir: Path):
        """Save metadata to CSV."""
        try:
            df = pd.DataFrame(metadata, columns=["audio_file", "text", "language"])
            csv_path = output_dir / f"metadata_{split}.csv"
            df.to_csv(csv_path, index=False, sep="|", encoding='utf-8')
            self.logger.info(f"Saved {split} metadata to {csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving {split} metadata: {str(e)}")
            raise

    def process_file(self, input_path: Path, output_dir: Path, dataset_index: int):
        """Process a single file through all steps."""
        file_name = input_path.name
        sanitized_name = self._sanitize_filename(file_name)
        sanitized_stem = sanitized_name.rsplit('.', 1)[0] if '.' in sanitized_name else sanitized_name
        dataset_dir = output_dir / f"dataset-{dataset_index}"
        self._create_directory_structure(dataset_dir)
        self._setup_dataset_logger(dataset_dir, dataset_index)

        # Log input file and parameters
        self.logger.info(f"Processing input file: {input_path}")
        self.logger.info(f"Parameters: max_duration={self.max_duration}s, train_split={self.train_split}, max_workers={self.max_workers}")

        try:
            # Step 1: Extract audio
            if input_path.suffix.lower() in ('.mp4', '.mkv', '.avi'):
                current_input = self.extract_audio_from_video(input_path, dataset_dir, sanitized_stem)
            elif input_path.suffix.lower() in ('.mp3', '.wav'):
                audio_path = dataset_dir / "audio" / f"{sanitized_stem}.mp3"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                if input_path.suffix.lower() == '.wav':
                    AudioSegment.from_file(input_path).export(audio_path, format="mp3")
                else:
                    shutil.copy(input_path, audio_path)
                current_input = audio_path
                self.logger.info(f"Copied audio to {audio_path}")
            else:
                self.logger.error(f"Unsupported file: {input_path}")
                return

            # Step 2: Remove background music
            current_input = self.remove_background_music(current_input, dataset_dir, sanitized_stem)

            # Step 3: Split into sentences
            current_input = self.split_into_sentences(current_input, dataset_dir)

            # Step 4: Transcribe segments
            metadata = self.transcribe_segment(current_input, dataset_dir)
            if metadata:
                train_size = int(len(metadata) * self.train_split)
                train_metadata = metadata[:train_size]
                eval_metadata = metadata[train_size:]
                if train_metadata:
                    self._save_metadata("train", train_metadata, dataset_dir)
                if eval_metadata:
                    self._save_metadata("eval", eval_metadata, dataset_dir)
                self.logger.info(f"Generated {len(train_metadata)} train and {len(eval_metadata)} eval metadata entries")

            # Clean up intermediates
            for dir_name in ["audio", "vocals"]:
                self._remove_directory(dataset_dir / dir_name)

            self.logger.info(f"Completed processing for {input_path} in {dataset_dir}")
        except Exception as e:
            self.logger.error(f"Processing failed for {input_path}: {str(e)}")
            self._remove_directory(dataset_dir)

    def process(self, input_dir: str, output_dir: str):
        """Process all files in the input directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        supported_extensions = ('.mp4', '.mkv', '.avi', '.mp3', '.wav')

        if not input_dir.is_dir():
            logging.error(f"Input path is not a directory: {input_dir}")
            return

        # Get list of files with supported extensions
        files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        if not files:
            logging.warning(f"No supported files found in {input_dir}")
            return

        # Process each file with a unique dataset index
        for index, file_path in enumerate(files, start=1):
            logging.info(f"Processing file {file_path} as dataset-{index}")
            self.process_file(file_path, output_dir, index)

if __name__ == "__main__":
    # Hardcoded input and output paths using raw strings
    input_dir = r"D:\Ope Watson\Project\AudioProcessing\input_files"
    output_dir = r"D:\Ope Watson\Project\AudioProcessing"

    maker = DatasetMaker()
    maker.process(input_dir, output_dir)