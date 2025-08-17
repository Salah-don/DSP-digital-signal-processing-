import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft , fftfreq # Charger le fichier audio
from scipy.signal import butter , lfilter
data, samplerate = sf.read("DSP/khkh.wav")
print("Taille du signal :", len(data))
print("Fréquence d'échantillonnage :", samplerate)

# Afficher le signal audio
# Créer un axe temps (en secondes)
temps = np.arange(len(data)) / samplerate

plt.plot(temps[:500], data[:500])  # Afficher les 1000 premiers échantillons
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.title("Signal audio")
plt.show()


# Nombre d'échantillons
N = len(data)

# Transformer en fréquences
yf = fft(data)                  # Transformée de Fourier
xf = fftfreq(N, 1 / samplerate) # Axe des fréquences

plt.plot(xf[:N//2], np.abs(yf[:N//2]))  # On garde la moitié utile
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude")
plt.title("Spectre du signal")
plt.show()

# Afficher le spectre en dB
plt.plot(xf[:N//2], 20 * np.log10(np.abs(yf[:N//2])))  # On garde la moitié utile
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.title("Spectre du signal (dB)")
plt.show()
# appliquer un filtre passe-bas
from scipy.signal import butter, lfilter

# Création d’un filtre passe-bas
def butter_filter(cutoff, fs, btype, order=5):
    nyq = 0.5 * fs                 # Fréquence de Nyquist
    normal_cutoff = cutoff / nyq   # Normalisation
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

def apply_filter(data, cutoff, fs, btype="low", order=5):
    b, a = butter_filter(cutoff, fs, btype, order)
    y = lfilter(b, a, data)
    return y

# === Paramètres ===
cutoff = 1000  # fréquence de coupure (Hz)

# Filtrage passe-bas et passe-haut
lowpassed = apply_filter(data, cutoff=cutoff, fs=samplerate, btype="low")
highpassed = apply_filter(data, cutoff=cutoff, fs=samplerate, btype="high")

# === Plot ===
t = np.arange(len(data)) / samplerate

plt.figure(figsize=(12, 8))

# Signal original (zoom sur 1000 premiers points)
plt.subplot(3, 1, 1)
plt.plot(t[:1000], data[:1000])
plt.title("Signal original (zoom)")
plt.xlabel("Temps (s)"); plt.ylabel("Amplitude")

# Passe-bas
plt.subplot(3, 1, 2)
plt.plot(t[:1000], lowpassed[:1000], color='g')
plt.title(f"Signal filtré passe-bas (cutoff = {cutoff} Hz)")
plt.xlabel("Temps (s)"); plt.ylabel("Amplitude")

# Passe-haut
plt.subplot(3, 1, 3)
plt.plot(t[:1000], highpassed[:1000], color='r')
plt.title(f"Signal filtré passe-haut (cutoff = {cutoff} Hz)")
plt.xlabel("Temps (s)"); plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()