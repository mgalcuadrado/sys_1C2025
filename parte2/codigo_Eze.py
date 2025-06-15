import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
# ========================================== FUNCIONES AUXILIARES ==================================================== #


def obtener_espectrograma(audio, fs, ventana, muestras_por_ventana, muestras_superpuestas_por_ventana):
   frecuencias, tiempos, intensidades = signal.spectrogram(audio, fs, window=ventana, nperseg=muestras_por_ventana,
               noverlap=muestras_superpuestas_por_ventana, mode='magnitude')


   intensidades_dB = 10 * np.log10(intensidades + 1e-10)  # Añadimos epsilon para evitar log(0)


   return frecuencias, tiempos, intensidades_dB


def graficar_espectrograma(tiempos, frecuencias, intensidades_dB, titulo, y_lim):
   plt.figure(figsize=(10, 6))
   plt.pcolormesh(tiempos, frecuencias, intensidades_dB, shading='gouraud', cmap='viridis')
   plt.title(titulo)
   plt.ylabel("Frecuencia [Hz]")
   plt.xlabel("Tiempo [s]")
   plt.colorbar(label='Intensidad [dB]')
   plt.ylim(0, y_lim)  # límite útil para formantes
   plt.tight_layout()
   plt.show()



fs, audio = wavfile.read("Picasso_hombre_5s.wav")

# Seteo de parámetros
win_duration = 0.005  # en milisegundos - Ventana corta → buena resolución temporal (banda ancha)
muestras_por_ventana = int(win_duration * fs)
muestras_superpuestas_por_ventana = int(muestras_por_ventana * 0.5)  # 50% de superposición
ventana = signal.get_window('hann', muestras_por_ventana)


frecuencias, tiempos, intensidades_dB = obtener_espectrograma(audio, fs, ventana, muestras_por_ventana, muestras_superpuestas_por_ventana)
graficar_espectrograma(tiempos, frecuencias, intensidades_dB, "Espectrograma de Banda Ancha - 'Picasso'", 4000)
