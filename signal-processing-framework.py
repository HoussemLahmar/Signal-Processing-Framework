import numpy as np
from collections import deque
from typing import List, Optional, Tuple

class SignalProcessor:
    """
    A real-time signal processing framework implementing common DSP operations.
    Supports filtering, FFT analysis, and convolution with optimized algorithms
    for embedded system applications.
    """
    
    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self.signal_buffer = deque(maxlen=buffer_size)
        self.sample_rate = 44100  # Default sample rate
        
    def update_sample_rate(self, new_rate: int):
        """Update the sample rate used for frequency calculations."""
        self.sample_rate = new_rate

    class FIRFilter:
        """Finite Impulse Response Filter implementation."""
        
        def __init__(self, coefficients: List[float]):
            self.coefficients = np.array(coefficients)
            self.buffer = deque([0] * len(coefficients), maxlen=len(coefficients))
            
        def process_sample(self, sample: float) -> float:
            """Process a single sample through the FIR filter."""
            self.buffer.appendleft(sample)
            return np.dot(self.coefficients, self.buffer)
            
        @staticmethod
        def design_lowpass(cutoff_freq: float, num_taps: int, sample_rate: float) -> 'FIRFilter':
            """Design a lowpass FIR filter."""
            nyquist = sample_rate / 2
            normalized_cutoff = cutoff_freq / nyquist
            
            # Using Hamming window for coefficient calculation
            h = np.hamming(num_taps)
            n = np.arange(num_taps)
            sinc = np.sinc(2 * normalized_cutoff * (n - (num_taps - 1) / 2))
            coefficients = h * sinc
            coefficients = coefficients / np.sum(coefficients)
            
            return SignalProcessor.FIRFilter(coefficients)

    class IIRFilter:
        """Infinite Impulse Response Filter implementation."""
        
        def __init__(self, b_coeffs: List[float], a_coeffs: List[float]):
            self.b_coeffs = np.array(b_coeffs)
            self.a_coeffs = np.array(a_coeffs)
            self.x_buffer = deque([0] * len(b_coeffs), maxlen=len(b_coeffs))
            self.y_buffer = deque([0] * (len(a_coeffs)-1), maxlen=len(a_coeffs)-1)
            
        def process_sample(self, sample: float) -> float:
            """Process a single sample through the IIR filter."""
            self.x_buffer.appendleft(sample)
            
            # Compute the filtered value
            y = np.dot(self.b_coeffs, self.x_buffer)
            y -= np.dot(self.a_coeffs[1:], self.y_buffer)
            y /= self.a_coeffs[0]
            
            self.y_buffer.appendleft(y)
            return y

    class FFTAnalyzer:
        """Real-time FFT analysis with windowing."""
        
        def __init__(self, window_size: int, hop_size: int):
            self.window_size = window_size
            self.hop_size = hop_size
            self.buffer = deque(maxlen=window_size)
            self.window = np.hanning(window_size)
            
        def process_buffer(self, data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
            """Process a buffer of samples and return magnitude spectrum."""
            # Add new samples to buffer
            self.buffer.extend(data)
            
            if len(self.buffer) < self.window_size:
                return np.array([]), np.array([])
            
            # Apply window function and compute FFT
            windowed_data = np.array(self.buffer) * self.window
            spectrum = np.fft.rfft(windowed_data)
            frequencies = np.fft.rfftfreq(self.window_size)
            
            # Compute magnitude spectrum in dB
            magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
            
            return frequencies, magnitude_db

    def apply_convolution(self, signal: List[float], kernel: List[float]) -> List[float]:
        """Apply convolution between signal and kernel."""
        return list(np.convolve(signal, kernel, mode='same'))

    def process_block(self, input_block: List[float]) -> List[float]:
        """Process a block of input samples through the entire DSP chain."""
        output_block = []
        
        for sample in input_block:
            # Add to buffer
            self.signal_buffer.append(sample)
            
            # Process through chain of operations
            processed_sample = sample
            
            # Add your processing chain here
            # Example: processed_sample = self.some_filter.process_sample(processed_sample)
            
            output_block.append(processed_sample)
            
        return output_block

# Example usage and test functions
def test_lowpass_filter():
    """Test the lowpass FIR filter implementation."""
    # Create a test signal with two frequencies
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 100 * t)
    
    # Design and apply lowpass filter
    lowpass = SignalProcessor.FIRFilter.design_lowpass(
        cutoff_freq=20,
        num_taps=101,
        sample_rate=1000
    )
    
    # Process signal
    filtered_signal = [lowpass.process_sample(s) for s in signal]
    return filtered_signal

def test_fft_analyzer():
    """Test the FFT analyzer implementation."""
    # Create a test signal
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
    
    # Create analyzer
    analyzer = SignalProcessor.FFTAnalyzer(window_size=1024, hop_size=512)
    
    # Process signal
    freqs, magnitudes = analyzer.process_buffer(signal)
    return freqs, magnitudes

if __name__ == "__main__":
    # Example usage
    processor = SignalProcessor(buffer_size=1024)
    filtered_signal = test_lowpass_filter()
    freqs, magnitudes = test_fft_analyzer()
