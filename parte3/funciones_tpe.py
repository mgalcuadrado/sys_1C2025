import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from scipy.signal import firwin, lfilter, spectrogram
import librosa
import soundfile as sf

FIGSIZE_X = 10
FIGSIZE_Y = 6


def Acelerar_audio_con_decimacion_y_ventaneo(input_wav, output_wav, factor, numtaps: int = 101):
    """
    Acelera un archivo WAV mono aplicando filtro antialiasing antes de la decimación.
    """
    # Leer archivo
    fs, data = wavfile.read(input_wav)
    
    # Normalización para que la información del audio sea de tipo float32
    data = data.astype(np.float32)

    # Filtro antialiasing (FIR con ventana Hamming)
    cutoff = 0.5 / factor
    fir_coeff = firwin(numtaps, cutoff, window='hamming')
    filtered = lfilter(fir_coeff, 1.0, data)

    # Compensar retardo del filtro
    delay = (numtaps - 1) // 2
    filtered = np.roll(filtered, -delay)
    filtered[-delay:] = 0.0

    # Decimación (acelerar): se guarda en el nuevo vector cada factor muestras
    accelerated = filtered[::factor]

    # Re-normalizar a int16 para guardar
    accelerated = np.clip(accelerated, -32768, 32767).astype(np.int16)

    #  Guardar con misma frecuencia de muestreo
    wavfile.write(output_wav, fs, accelerated)


def Acelerar_audio_con_librosa(input_wav, output_wav, factor):
    y, sr = librosa.load(input_wav, sr=None) 
    y_fast = librosa.effects.time_stretch(y, rate= factor) # Estirar el tiempo: factor < 1 acelera (0.5 = 2x más rápido)
    sf.write(output_wav, y_fast, sr) # Guardar con pitch original


# =================================== FUNCIONES AUXILIARES PARTE 2 ======================================
# graficar_espectrograma_banda_angosta grafica el espectrograma de banda angosta de una señal audio_data con una fs sample_rate, considerando la porción de la señal desde start_time_s [en segundos] y end_time_s[en segundos], y grafica desde las frecuencias start_freq a end_freq. Si start_time_s y end_time_s son None se grafica la señal audio_data completa. Lo mismo pasa con start_freq y end_freq. Además, se pide el título del gráfico y el nombre de la imagen. Adicionalmente se ofrece un arreglo func, donde func[0] debe ser el nombre de la función a llamar y en el resto de los elementos del vector se pueden incluir otros datos necesarios para realizar la operación deseada.
def Graficar_espectrograma_banda_angosta(audio_data, sample_rate, start_time_s, end_time_s, start_freq, end_freq, title, image_name, func):
    samples_per_segment = 2048 #longitud de la ventana considerada para cada fragmento de la señal
    sample_overlap = samples_per_segment / 2 # la mitad de las muestras se superponen entre una ventana y la siguiente para evitar efectos de borde
    sample = audio_data
    if start_time_s is not None and end_time_s is not None:
        start_time = int(start_time_s * sample_rate)
        end_time = int(end_time_s * sample_rate)
        sample = audio_data[start_time:end_time]   
    frecuencias, tiempos, intensidades = spectrogram(sample, sample_rate, window='hamming', nperseg=samples_per_segment, noverlap=sample_overlap)
    intensidades_dB = 10 * np.log10(intensidades + 1e-10)
    # Gráfico del espectrograma de banda angosta completo 
    plt.figure(figsize=(FIGSIZE_X, FIGSIZE_Y))
    plt.pcolormesh(tiempos, frecuencias, intensidades_dB, shading='gouraud', cmap='viridis')
    if func is not None:
        func(frecuencias, tiempos, intensidades) 
    plt.ylabel('Frecuencia [Hz]')
    plt.xlabel('Tiempo [s]')
    plt.title(title)
    if start_freq is not None and end_freq is not None:
        plt.ylim(start_freq, end_freq)
    plt.colorbar(label='Densidad espectral [dB]')
    plt.tight_layout()
    plt.savefig(image_name)
    plt.show()

def Desacelerar_audio_con_expansion_y_pasabajos(input_wav, output_wav, factor, numtaps: int = 101):
    """
    Desacelera un archivo WAV mono utilizando expansión por inserción de ceros
    y un filtro pasa-bajos para interpolación.
    """
    # Leer archivo
    fs, data = wavfile.read(input_wav)

    # Normalización para trabajar con float32
    data = data.astype(np.float32)

    # Expansión: insertar (factor - 1) ceros entre muestras
    expanded = np.zeros(int(len(data) * factor), dtype=np.float32)
    expanded[::factor] = data

    # Filtro pasa-bajos para interpolar (ventana Hamming)
    cutoff = 0.5 / factor  # Frecuencia normalizada (Nyquist = 1.0)
    fir_coeff = firwin(numtaps, cutoff, window='hamming')
    filtered = lfilter(fir_coeff, 1.0, expanded)

    # Compensar retardo del filtro
    delay = (numtaps - 1) // 2
    filtered = np.roll(filtered, -delay)
    filtered[-delay:] = 0.0

    # Re-normalizar a int16 para guardar
    filtered = np.clip(filtered, -32768, 32767).astype(np.int16)

    # Guardar con la misma frecuencia de muestreo
    wavfile.write(output_wav, fs, filtered)

