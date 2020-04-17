import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import math
import pylab
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


def getFormat(x, pos):
    "The two args are the value and tick position"
    if x >= 1e6:
        return "%1.1fM" % (x / 1e6)
    else:
        return "%dK" % (x / 1000)


formatter = FuncFormatter(getFormat)


def graphSignal(time, pcm_values, dft_frequencies, dft_y):
    pylab.figure(figsize=(19, 12))
    pylab.subplot(211)
    pylab.title("Signal wave")
    pylab.ylabel("Amplitude")
    pylab.xlabel("Time (s)")
    pylab.plot(time, pcm_values)
    plot2 = pylab.subplot(212)
    pylab.title("DFT of %s" % "Laurel-Yanny")
    pylab.ylabel("")
    pylab.xlabel("Frequency [Hz]")
    # plot2.yaxis.set_major_formatter(formatter)
    pylab.plot(dft_frequencies, abs(dft_y.real))
    pylab.show()


def getSpectogram(pcm_values):
    signalBins = []
    i = 0
    while i < len(pcm_values):
        signal = pcm_values[i : (i + 499)]
        signalBins.append(np.fft.rfft(signal))
        i += 500
    print(len(signalBins))
    return signalBins


def main():
    file_path = "laurel_yanny.wav"

    frame_rate, pcm_values = wavfile.read(file_path)

    N = len(pcm_values)  # Number of sample points
    T = 1.0 / frame_rate  # Sample spacing/period
    time = np.linspace(start=0, stop=N / frame_rate, num=len(pcm_values))

    dft_y = np.fft.rfft(pcm_values)
    dft_frequencies = np.fft.rfftfreq(N, d=T)

    pylab.title("Spectrogram of %s" % "Laurel-Yanny")
    pylab.ylabel("Frequency [Hz]")
    pylab.xlabel("Time (s)")
    pylab.specgram(pcm_values, NFFT=512, Fs=frame_rate, mode="psd")

    graphSignal(time, pcm_values, dft_frequencies, dft_y)


if __name__ == "__main__":
    main()
