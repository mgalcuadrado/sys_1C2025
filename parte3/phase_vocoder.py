import librosa
import numpy as np
import soundfile as sf
from scipy.signal import get_window
from espectrograma import obtener_espectrograma, graficar_espectrograma 

#Defino la funcion phase vocoder

def phase_vocoder(matriz, factor, hop_length):  #factor: velocidad del audio, puede ser x0.5, x2, x3, etc.
    n_frec, n_frames = matriz.shape  #devuelve el numero de filas y columnas
    paso_temp = np.arange(0, n_frames-1, factor) 
    nuevo_arreglo = np.zeros((n_frec, len(paso_temp)), dtype=complex) #crea memoria para la nueva matriz espectral compleja (dtype=complex)
    fase = np.angle(matriz[:, 0]) #Se obtiene la fases para el prier frame
    omega = 2* np.pi * np.arange(n_frec) / hop_length # vector w_k = 2.pi.k/N
    nueva_fase = 0
    for i, t in enumerate(paso_temp):
        t_int= int(np.floor(t))
        t_frac = t - t_int
        mag = (1-t_frac) * np.abs(matriz[:, t_int]) + t_frac * np.abs(matriz[:, t_int + 1])  #Interpolacion para el vector de magnitudes de la neva columna de frames 
    #Se ajusta la fase para el nuevo arreglo
        dif_fases = np.angle(matriz[:, t_int + 1]) - np.angle(matriz[:, t_int])
        dif_fases = np.mod(dif_fases - omega * hop_length + np.pi, 2 * np.pi) - np.pi  #ajusta el valor en el intervalo de -pi a pi
        dif_fases += omega * hop_length                                                #Obtiene la fase real
        nueva_fase += dif_fases
        nuevo_arreglo[:, i] = mag * np.exp(1j * nueva_fase)
    return nuevo_arreglo 

# Cargar la señal de voz
audio, fs = librosa.load('Picasso_hombre_3s.wav', sr=None) #Mantiene la frecuencia de muestreo del archivo
print(fs) #fs=16k

win_duration = 0.128  # en milisegundos - Ventana larga → buena resolución frecuencial (banda ansgosta)
muestras_por_ventana = int(win_duration * fs)
salto_por_ventana = int(muestras_por_ventana * 0.25)  # Solapamiento

#Matriz donde cada columna representa el espectro en cierto instante
espectro_original = librosa.stft(audio, n_fft= muestras_por_ventana, hop_length= salto_por_ventana)

espectro_modificado = phase_vocoder(espectro_original, 0.5, salto_por_ventana)
audio_modificado = librosa.istft(espectro_modificado, hop_length= salto_por_ventana)

sf.write('picasso_rapido_modificado.wav',audio_modificado, fs)


overlap = int(muestras_por_ventana * 0.50)
ventana = get_window('hann', muestras_por_ventana)

frecuencias, tiempos, intensidades_dB = obtener_espectrograma(audio, fs, ventana, muestras_por_ventana, overlap)
graficar_espectrograma(tiempos, frecuencias, intensidades_dB, "Espectrograma de Picasso rapido original", 'espectrograma_rapido_original.png', 8000)


frecuencias, tiempos, intensidades_dB = obtener_espectrograma(audio_modificado, fs, ventana, muestras_por_ventana, overlap)
graficar_espectrograma(tiempos, frecuencias, intensidades_dB, "Espectrograma de Picasso rapido modificado", 'espectrograma_rapido_modificado.png', 8000)


