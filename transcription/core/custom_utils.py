from silero_vad_module import SileroVAD


def vad_only(vad: SileroVAD, audio):
    audio_segment = audio["waveform"]
    sample_rate = audio["sample_rate"]
    out = []

    speech_timestamps = vad.get_speech_timestamps(audio_segment, vad.vad_model, sampling_rate=sample_rate)
    for ts in speech_timestamps:
        out.append(
            {
                "start": ts["start"] / sample_rate,  # in seconds
                "end": ts["end"] / sample_rate
            }
        )

    return out


def merge_over_min_length(vad_list):
    """
    Merge and trim VAD segments by speaker labels, enforcing constraints on segment length and merge gaps.

    Args:
        vad_list (list): List of VAD segments with start, end, and speaker labels.

    Returns:
        list: A list of updated VAD segments after merging and trimming.
    """
    MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge
    MIN_SEGMENT_LENGTH = 1  # min segment length in seconds
    MAX_SEGMENT_LENGTH = 30  # max segment length in seconds

    updated_list = []

    for idx, vad in enumerate(vad_list):
        last_start_time = updated_list[-1]["start"] if updated_list else None
        last_end_time = updated_list[-1]["end"] if updated_list else None

        if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH:
            updated_list.append(vad)
            continue

        if (
            last_start_time is None
            or last_end_time is None
            or vad["start"] - last_end_time >= MERGE_GAP
            or vad["end"] - last_start_time >= MAX_SEGMENT_LENGTH
        ):
            updated_list.append(vad)
        else:
            updated_list[-1]["end"] = vad["end"]  # merge the time

    print(
        f"merge_over_min_length > merged {len(vad_list) - len(updated_list)} segments"
    )

    filter_list = [
        vad for vad in updated_list if vad["end"] - vad["start"] >= MIN_SEGMENT_LENGTH
    ]

    print(
        f"merge_over_min_length > removed: {len(updated_list) - len(filter_list)} segments by length"
    )

    return filter_list
