# -*- coding: utf-8 -*-
# 🔢🎵 Ultimate FFT Analyzer v12.0 🚀
# 📦 Requires: pyqt5, pyqtgraph, numpy, pyaudio, scipy

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QKeyEvent
import pyaudio
from scipy.signal import find_peaks, butter, lfilter, lfilter_zi
from collections import deque
from datetime import datetime

# 🎛️ Core Configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 8192
WINDOW = np.blackman(BUFFER_SIZE)
MIN_DB = -70
PEAK_THRESHOLD = -68  # More sensitive overall
NOTE_TOLERANCE = 2.0  # Base tolerance percentage
UPDATE_INTERVAL = 30
NOISE_FLOOR_ALPHA = 0.90

TARGET_NOTES = {
    '174Hz': 174.00, '285Hz': 285.00, '396Hz': 396.00,
    '417Hz': 417.00, '432Hz': 432.00, '440Hz': 440.00,
    '528Hz': 528.00, '639Hz': 639.00, '741Hz': 741.00,
    '852Hz': 852.00, '963Hz': 963.00
}

SOLFEGGIO_COLORS = [
    '#C0C0C0',  # 174Hz: Silver
    '#D4AF37',  # 285Hz: Metallic Gold
    '#FF0000',  # 396Hz: Red
    '#FFA500',  # 417Hz: Orange
    '#0000FF',  # 432Hz: Blue
    '#800020',  # 440Hz: Burgundy
    '#FFFF00',  # 528Hz: Yellow
    '#00FF00',  # 639Hz: Green
    '#00FFFF',  # 741Hz: Cyan
    '#4B0082',  # 852Hz: Indigo
    '#FF00FF'  # 963Hz: Magenta
]

SOLFEGGIO_FREQS = list(TARGET_NOTES.items())


class RealTimeAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solfeggio Frequency Analyzer v12.0")
        self.setGeometry(50, 50, 1600, 800)
        self.init_components()
        self.init_audio()
        self.setup_ui()

        self.current_counts = {name: 0 for name in TARGET_NOTES}
        self.start_time = datetime.now()
        self.last_enter_time = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(UPDATE_INTERVAL)

    def init_components(self):
        self.p = pyaudio.PyAudio()
        self.frequencies = np.fft.rfftfreq(BUFFER_SIZE, 1 / SAMPLE_RATE)
        self.target_indices = [np.abs(self.frequencies - f).argmin() for _, f in SOLFEGGIO_FREQS]
        self.note_buffer = deque(maxlen=8)

        # Enhanced low-frequency bandpass filter (70Hz-4kHz)
        self.bp_b, self.bp_a = butter(4, [70 / (SAMPLE_RATE / 2), 4000 / (SAMPLE_RATE / 2)], btype='band')
        self.bp_zi = lfilter_zi(self.bp_b, self.bp_a) * 0.0

    def init_audio(self):
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=BUFFER_SIZE,
            start=False
        )
        self.stream.start_stream()
        self.calibrate_noise_floor()

    def calibrate_noise_floor(self):
        print("Calibrating adaptive noise floor...")
        noise_frames = []
        for _ in range(20):  # Reduced calibration frames for faster setup
            raw_data = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
            buffer = np.frombuffer(raw_data, dtype=np.float32)
            filtered, self.bp_zi = lfilter(self.bp_b, self.bp_a, buffer, zi=self.bp_zi)
            windowed = filtered * WINDOW
            noise_frames.append(np.abs(np.fft.rfft(windowed)))

        # Frequency-weighted noise floor
        self.noise_floor = np.minimum(
            np.percentile(noise_frames, 20, axis=0),
            np.median(noise_frames, axis=0) * 0.8
        ) / BUFFER_SIZE

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setLabel('left', 'Amplitude (dB)')
        self.spectrum_plot.setLabel('bottom', 'Frequency (Hz)')
        self.spectrum_plot.setLogMode(x=True)
        self.spectrum_plot.setYRange(MIN_DB, 0)
        self.spectrum_curve = self.spectrum_plot.plot(pen='#4ECDC4')

        self.bar_plot = pg.PlotWidget()
        self.bars = []
        for i, name in enumerate(TARGET_NOTES):
            bar = pg.BarGraphItem(
                x=[i], height=[MIN_DB], width=0.6,
                brush=SOLFEGGIO_COLORS[i]
            )
            self.bar_plot.addItem(bar)
            self.bars.append(bar)
        self.bar_plot.getAxis('bottom').setTicks([[(i, name) for i, name in enumerate(TARGET_NOTES)]])

        self.detection_label = QLabel("Active Frequencies: ")
        self.detection_label.setFont(QFont('Arial', 14))
        self.detection_label.setStyleSheet("color: #4ECDC4;")

        left_layout.addWidget(self.spectrum_plot)
        left_layout.addWidget(self.bar_plot)
        left_layout.addWidget(self.detection_label)
        left_panel.setLayout(left_layout)

        # Right Panel
        self.ratio_plot = pg.PlotWidget()
        self.ratio_plot.setLabel('left', 'Enhanced Distribution (%)')
        self.ratio_plot.setLabel('bottom', 'Solfeggio Frequencies')
        self.ratio_plot.setYRange(0, 160)  # Expanded range

        self.ratio_bars = pg.BarGraphItem(
            x=np.arange(len(SOLFEGGIO_FREQS)),
            height=[0] * len(SOLFEGGIO_FREQS),
            width=0.6,
            brushes=SOLFEGGIO_COLORS
        )
        self.ratio_plot.addItem(self.ratio_bars)
        self.ratio_plot.getAxis('bottom').setTicks([[(i, name) for i, (name, _) in enumerate(SOLFEGGIO_FREQS)]])

        main_layout.addWidget(left_panel, 60)
        main_layout.addWidget(self.ratio_plot, 40)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        pg.setConfigOptions(antialias=True, background='#2e3440', foreground='#ffffff')

    def process_audio(self):
        try:
            raw_data = self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
            buffer = np.frombuffer(raw_data, dtype=np.float32)
            filtered, self.bp_zi = lfilter(self.bp_b, self.bp_a, buffer, zi=self.bp_zi)
            windowed = filtered * WINDOW
            spectrum = np.fft.rfft(windowed)
            magnitude_linear = np.abs(spectrum) / BUFFER_SIZE
            filtered = np.maximum(magnitude_linear - self.noise_floor, 0)
            return 20 * np.log10(filtered + 1e-10)
        except Exception as e:
            print(f"Audio error: {e}")
            return np.zeros(len(self.frequencies))

    def process_frame(self):
        magnitude = self.process_audio()
        self.update_plots(magnitude)
        self.detect_peaks(magnitude)
        self.classify_frequencies(magnitude)

    def classify_frequencies(self, magnitude):
        peaks, props = find_peaks(
            magnitude,
            height=PEAK_THRESHOLD,
            prominence=(0.5, None),  # Reduced minimum prominence
            width=(3, 60),  # Wider low-frequency acceptance
            distance=10
        )

        for idx in peaks:
            freq = self.frequencies[idx]
            for name, target in SOLFEGGIO_FREQS:
                # Enhanced 174Hz detection
                if name == '174Hz':
                    tolerance = target * (NOTE_TOLERANCE * 1.5 / 100)  # 3% tolerance
                    prominence_boost = 1.6
                else:
                    tolerance = target * (NOTE_TOLERANCE / 100)  # 2% tolerance
                    prominence_boost = 1.3

                if abs(freq - target) <= tolerance:
                    self.current_counts[name] += props["prominences"][peaks.tolist().index(idx)] * prominence_boost
                    break

    def update_plots(self, magnitude):
        self.spectrum_curve.setData(self.frequencies, magnitude)
        levels = magnitude[self.target_indices]
        for i, bar in enumerate(self.bars):
            level = levels[i] if i < len(levels) else MIN_DB
            bar.setOpts(height=level)

    def detect_peaks(self, magnitude):
        current = set()
        peaks, _ = find_peaks(magnitude, height=PEAK_THRESHOLD)
        for peak in peaks:
            freq = self.frequencies[peak]
            for name, target in TARGET_NOTES.items():
                tolerance = target * (NOTE_TOLERANCE / 100)
                if abs(freq - target) <= tolerance:
                    current.add(name)
        self.note_buffer.append(current)
        persistent = set().union(*self.note_buffer)
        text = "Active: " + ", ".join(sorted(persistent)) if persistent else "No strong signals"
        self.detection_label.setText(text)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            current_time = datetime.now()
            elapsed = (current_time - (self.last_enter_time or self.start_time)).total_seconds()

            total = sum(self.current_counts.values())
            percentages = [(c / total) * 100 if total > 0 else 0 for c in self.current_counts.values()]

            # Enhanced 174Hz visualization boost
            scaled = []
            for p, (name, _) in zip(percentages, SOLFEGGIO_FREQS):
                if name == '174Hz':
                    scaled.append(p * 1.4 + np.power(p, 1.4))  # Extra boost
                else:
                    scaled.append(p * 1.25 + np.power(p, 1.3))

            self.ratio_bars.setOpts(height=scaled)
            self.current_counts = {name: 0 for name in TARGET_NOTES}
            self.last_enter_time = current_time
            print(f"📈 174Hz-optimized capture ({elapsed:.1f}s)")
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    analyzer = RealTimeAnalyzer()
    analyzer.show()
    sys.exit(app.exec_())