# Signal Processing Framework

A real-time signal processing framework implementing common DSP (Digital Signal Processing) operations in Python. This framework is optimized for embedded system applications and provides implementations for FIR/IIR filtering, FFT analysis, and convolution.


## Features

- **Real-time Processing**: Efficient sample-by-sample processing with circular buffers
- **FIR Filtering**: 
  - Custom coefficient design
  - Built-in lowpass filter design using Hamming window
  - Optimized for real-time applications
- **IIR Filtering**:
  - Support for custom filter coefficients
  - Efficient implementation using circular buffers
- **FFT Analysis**:
  - Real-time spectrum analysis
  - Windowing support (Hanning window)
  - Magnitude spectrum computation in dB
- **Convolution**: Efficient convolution implementation using NumPy

## Installation

```bash
pip install numpy
```

## Usage Examples

### Basic Signal Processing

```python
from signal_processor import SignalProcessor

# Initialize the processor
processor = SignalProcessor(buffer_size=1024)

# Process a block of samples
input_block = [0.5, 0.6, 0.7, 0.8]
output_block = processor.process_block(input_block)
```

### FIR Lowpass Filter

```python
# Design and apply a lowpass filter
lowpass = SignalProcessor.FIRFilter.design_lowpass(
    cutoff_freq=20,
    num_taps=101,
    sample_rate=1000
)

# Process samples
filtered_sample = lowpass.process_sample(input_sample)
```

### FFT Analysis

```python
# Create FFT analyzer
analyzer = SignalProcessor.FFTAnalyzer(window_size=1024, hop_size=512)

# Process signal and get spectrum
frequencies, magnitudes = analyzer.process_buffer(signal_data)
```

## API Reference

### SignalProcessor

Main class providing the signal processing framework.

#### Methods:
- `update_sample_rate(new_rate: int)`: Update the sample rate
- `process_block(input_block: List[float]) -> List[float]`: Process a block of samples
- `apply_convolution(signal: List[float], kernel: List[float]) -> List[float]`: Apply convolution

### FIRFilter

Finite Impulse Response Filter implementation.

#### Methods:
- `process_sample(sample: float) -> float`: Process a single sample
- `design_lowpass(cutoff_freq: float, num_taps: int, sample_rate: float) -> FIRFilter`: Design a lowpass filter

### IIRFilter

Infinite Impulse Response Filter implementation.

#### Methods:
- `process_sample(sample: float) -> float`: Process a single sample through the IIR filter

### FFTAnalyzer

Real-time FFT analysis with windowing.

#### Methods:
- `process_buffer(data: List[float]) -> Tuple[np.ndarray, np.ndarray]`: Process signal buffer and return frequency spectrum

## Requirements

- Python 3.7+
- NumPy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

