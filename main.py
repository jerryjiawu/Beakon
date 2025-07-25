from src import *

def main():
    detector = AcousticDetector(
        sample_rate=44100,
        chunk_size=1024,
        low_freq=1000,      # 1 kHz
        high_freq=8000,     # 8 kHz
        spike_threshold=2.0, # 2 standard deviations above mean
        recording_duration=10.0  # 10 seconds
    )
    detector.start_monitoring()


if __name__ == "__main__":
    main()