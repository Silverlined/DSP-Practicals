import sounddevice as sd
import soundfile as sf
import numpy as np
import math
from scipy.signal import butter, lfilter
from scipy.io import wavfile
from matplotlib import pyplot


def playAudio(file_path):
    data, fs = sf.read(file_path)
    sd.play(data, samplerate=fs)
    status = sd.wait()


def recordAudio(file_name, sample_rate, audio_samples):
    audio_samples = audio_samples.astype(np.int16)
    wavfile.write(file_name, sample_rate, audio_samples)


def plotAudioSignal(time, time_series, dft_frequencies, dft_y):
    pyplot.figure(figsize=(19, 12))
    pyplot.subplot(211)
    pyplot.title("Audio time series")
    pyplot.ylabel("Amplitude")
    pyplot.xlabel("Time (s)")
    pyplot.plot(time, time_series)
    plot2 = pyplot.subplot(212)
    pyplot.title("DFT of %s" % "Laurel-Yanny")
    pyplot.ylabel("")
    pyplot.xlabel("Frequency [Hz]")
    # plot2.yaxis.set_major_formatter(formatter)
    pyplot.plot(dft_frequencies, abs(dft_y.real))
    pyplot.show()


def plotSpectogram(time_series, sample_rate):
    pyplot.title("Spectrogram of %s" % "Laurel-Yanny")
    pyplot.ylabel("Frequency [Hz]")
    pyplot.xlabel("Time [s]")
    pyplot.specgram(time_series, NFFT=512, Fs=sample_rate, mode="psd")
    pyplot.show()


def plotSection(start, end, y, filtered_y1, filtered_y2):
    # Plot short section of time series
    xlim = (start, end)
    section_range = np.arange(xlim[0], xlim[1] + 1)
    pyplot.plot(section_range, y[section_range], "k")
    pyplot.plot(section_range, filtered_y1[section_range], "b")
    pyplot.plot(section_range, filtered_y2[section_range], "r")
    pyplot.ylim([1.5 * y[section_range].min(), 1.5 * y[section_range].max()])
    pyplot.xlim(xlim)
    pyplot.legend(["y", "< 1320 Hz", "> 440 Hz"])
    pyplot.title("Fourier filters")
    pyplot.grid()
    pyplot.show()


def computeDFT(time_series, num_samples, sample_spacing):
    dft_y = np.fft.rfft(time_series)
    dft_frequencies = np.fft.rfftfreq(num_samples, d=sample_spacing)
    return dft_frequencies, dft_y


def applyFourierFilters(time_series, num_samples, sample_rate, sample_spacing):
    y = time_series
    dft_frequencies, dft_y = computeDFT(time_series, num_samples, sample_spacing)

    # Find DFT indices corresponding to frequency ranges to zero out
    flt1 = dft_frequencies < 1320
    flt2 = dft_frequencies > 440

    dft_y_copy = dft_y.copy()
    dft_y_copy[flt1] = 0
    attenuated_y = np.fft.irfft(dft_y_copy)
    filtered_y1 = y - attenuated_y

    dft_y_copy = dft_y.copy()
    dft_y_copy[flt2] = 0
    attenuated_y = np.fft.irfft(dft_y_copy)
    filtered_y2 = y - attenuated_y

    recordAudio("filtered_1", sample_rate, filtered_y1)
    recordAudio("filtered_2", sample_rate, filtered_y2)

    # plotSpectogram(filtered_y1 sample_rate)
    plotSection(8150, 8250, y, filtered_y1, filtered_y2)
    playAudio("filtered_1.wav")


def fir_low_pass(time_series, fs, fc, filter_len, outputType):
    h = np.sinc(2 * fc / fs * (np.arange(filter_len) - (filter_len - 1) / 2))  # Compute sinc filter.

    h *= np.hamming(filter_len)  # Apply window.

    h /= np.sum(h)  # Normalize to get unity gain.

    filtered_signal = np.convolve(time_series, h).astype(outputType)
    return filtered_signal, h


def fir_high_pass(time_series, fs, fc, filter_len, outputType):
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    signal, h = fir_low_pass(time_series, fs, fc, filter_len, outputType)
    h = -h
    h[(filter_len - 1) // 2] += 1

    filtered_signal = np.convolve(time_series, h).astype(outputType)
    return filtered_signal, h


def applyTimeDomainFiltering(time_series, sample_rate):
    filtered_1, h1 = fir_low_pass(time_series, sample_rate, fc=1320, filter_len=311, outputType=np.int16)
    filtered_2, h2 = fir_high_pass(time_series, sample_rate, fc=441, filter_len=311, outputType=np.int16)

    recordAudio("filtered_1.wav", sample_rate, filtered_1)
    recordAudio("filtered_2.wav", sample_rate, filtered_2)

    # plotSpectogram(filtered_2, sample_rate)
    plotSection(8150, 8250, time_series, filtered_1, filtered_2)
    playAudio("filtered_1.wav")


def applyButterworthFiltering(time_series, sample_rate):
    N = 8  # Filter Order
    nyq = 0.5 * sample_rate
    lowcut = 1320
    fc_low = lowcut / nyq  # normalized cut-off frequency
    highcut = 440
    fc_high = highcut / nyq  # normalized cut-off frequency

    lowpass_b, lowpass_a = butter(N, fc_low, btype="lowpass")
    highpass_b, highpass_a = butter(N, fc_high, btype="highpass")

    filtered_1 = lfilter(lowpass_b, lowpass_a, time_series)
    filtered_2 = lfilter(highpass_b, highpass_a, time_series)

    recordAudio("filtered_1.wav", sample_rate, filtered_1)
    recordAudio("filtered_2.wav", sample_rate, filtered_2)

    # plotSpectogram(filtered_2, sample_rate)
    plotSection(8150, 8250, time_series, filtered_1, filtered_2)
    playAudio("filtered_2.wav")


def main():
    file_path = "laurel_yanny.wav"

    frame_rate, pcm_values = wavfile.read(file_path)
    # playAudio(file_path)

    num_samples = len(pcm_values)  # Number of sample points
    T = 1.0 / frame_rate  # Sample spacing/period
    time = np.linspace(start=0, stop=num_samples / frame_rate, num=len(pcm_values))

    dft_frequencies, dft_y = computeDFT(pcm_values, num_samples, T)

    # plotAudioSignal(time, pcm_values, dft_frequencies, dft_y)
    # plotSpectogram(pcm_values, frame_rate)
    # applyFourierFilters(pcm_values, num_samples, frame_rate, T)
    # applyTimeDomainFiltering(pcm_values, frame_rate)
    # applyButterworthFiltering(pcm_values, frame_rate)


if __name__ == "__main__":
    main()
