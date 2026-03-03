"""Transcribe audio to MIDI using Spotify basic-pitch."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe_to_midi(wav_path, output_dir, onset_threshold=0.5,
                       frame_threshold=0.3, min_note_length=58.0):
    """Transcribe WAV to MIDI. Returns path to generated MIDI file."""
    from basic_pitch.inference import predict_and_save
    from basic_pitch import ICASSP_2022_MODEL_PATH

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = output_dir / (wav_path.stem + ".mid")

    if midi_path.exists():
        logger.debug("MIDI cache hit: %s", midi_path.name)
        return midi_path

    logger.info("Transcribing: %s", wav_path.name)

    predict_and_save(
        audio_path_list=[str(wav_path)],
        output_directory=str(output_dir),
        save_midi=True,
        save_model_outputs=False,
        save_notes=False,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
    )

    # basic-pitch names output as <stem>_basic_pitch.mid
    bp_path = output_dir / (wav_path.stem + "_basic_pitch.mid")
    if bp_path.exists():
        bp_path.rename(midi_path)

    if not midi_path.exists():
        raise FileNotFoundError("MIDI not created for " + str(wav_path))

    logger.info("MIDI saved: %s", midi_path.name)
    return midi_path
