"""
Music Analyzer

This program allows the user to analyze an audio file and generate a series of plots that provide insights into the audio file's characteristics. The plots are saved as an image file.

Author: Seisa Likotsi
Date: May 2024
"""

import tkinter as tk
from tkinter import filedialog
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os

def load_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        analyze_file(file_path)

def analyze_file(file_path):
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    try:
        # Create a figure with multiple subplots and a dark background
        fig, ax = plt.subplots(5, 2, figsize=(25, 14), facecolor='#2E2E2E')
        plt.style.use('dark_background')  # Use Matplotlib's dark background style
        
        # Set the title of the figure to the name of the audio file
        audio_file_name = os.path.basename(file_path)
        fig.suptitle(audio_file_name, fontsize=16, color='white')

        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=ax[0, 0])
        ax[0, 0].set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')
        ax[0, 0].title.set_color('white')  # Set title color
        ax[0, 0].xaxis.label.set_color('white')  # Set x-axis label color
        ax[0, 0].yaxis.label.set_color('white')  # Set y-axis label color

        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[0, 1], cmap='magma')
        ax[0, 1].set(title='Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax[0, 1].title.set_color('white')  # Set title color
        ax[0, 1].xaxis.label.set_color('white')  # Set x-axis label color
        ax[0, 1].yaxis.label.set_color('white')  # Set y-axis label color
        fig.colorbar(img, ax=ax[0, 1], format='%+2.0f dB')

        # Calculate and display tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        ax[1, 0].text(0.5, 0.5, f'Tempo: {tempo:.2f} BPM', horizontalalignment='center', verticalalignment='center', fontsize=15, color='white')
        ax[1, 0].set(title='Tempo', xticks=[], yticks=[])
        ax[1, 0].title.set_color('white')  # Set title color

        # Plot MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax[1, 1], cmap='magma')
        ax[1, 1].set(title='MFCC', xlabel='Time (s)', ylabel='MFCC Coefficients')
        ax[1, 1].title.set_color('white')  # Set title color
        ax[1, 1].xaxis.label.set_color('white')  # Set x-axis label color
        ax[1, 1].yaxis.label.set_color('white')  # Set y-axis label color
        fig.colorbar(img, ax=ax[1, 1])

        # Plot chroma feature
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax[2, 0], cmap='magma')
        ax[2, 0].set(title='Chroma Feature', xlabel='Time (s)')
        ax[2, 0].title.set_color('white')  # Set title color
        ax[2, 0].xaxis.label.set_color('white')  # Set x-axis label color
        ax[2, 0].yaxis.label.set_color('white')  # Set y-axis label color
        fig.colorbar(img, ax=ax[2, 0])

        # Plot zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        ax[2, 1].plot(zcr[0], color='cyan')
        ax[2, 1].set(title='Zero Crossing Rate', xlabel='Frames', ylabel='Rate')
        ax[2, 1].title.set_color('white')  # Set title color
        ax[2, 1].xaxis.label.set_color('white')  # Set x-axis label color
        ax[2, 1].yaxis.label.set_color('white')  # Set y-axis label color

        # Noise and distortion detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        ax[3, 0].plot(times, onset_env, label='Onset strength', color='white')
        ax[3, 0].set(title='Noise and Distortion Detection', xlabel='Time (s)', ylabel='Onset Strength')
        ax[3, 0].title.set_color('white')
        ax[3, 0].xaxis.label.set_color('white')
        ax[3, 0].yaxis.label.set_color('white')

        # Frequency balance
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        bass = np.mean(librosa.amplitude_to_db(S[(freqs > 0) & (freqs < 150)], ref=np.max))
        mid = np.mean(librosa.amplitude_to_db(S[(freqs >= 150) & (freqs < 2000)], ref=np.max))
        treble = np.mean(librosa.amplitude_to_db(S[freqs >= 2000], ref=np.max))
        bars = ax[3, 1].bar(['Bass', 'Midrange', 'Treble'], [bass, mid, treble], color=[           'blue', 'green', 'red'])

        # Add a legend for each bar
        for bar, label in zip(bars, ['Bass', 'Midrange', 'Treble']):
            ax[3, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, label,                  ha='center', va='bottom', color='white')

            ax[3, 1].set(title='Frequency Balance', ylabel='Amplitude (dB)')
            ax[3, 1].title.set_color('white')
            ax[3, 1].yaxis.label.set_color('white')
                
        # Timbre analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        ax[4, 0].plot(librosa.times_like(spectral_centroid), spectral_centroid[0], color='white')
        ax[4, 0].set(title='Spectral Centroid (Timbre)', xlabel='Time (s)', ylabel='Hz')
        ax[4, 0].title.set_color('white')
        ax[4, 0].xaxis.label.set_color('white')
        ax[4, 0].yaxis.label.set_color('white')
        

        # Display the plots in the GUI
        for a in ax.flat:
            a.label_outer()
            a.set_facecolor('#2E2E2E')  # Set subplot background color

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # Remove the 'Time (s)' label from all plots except the last one in each column
        for i in range(4):
            ax[i, 0].set_xlabel('')
            ax[i, 1].set_xlabel('')
        
        # Add labels and colorbar to the Zero Crossing Rate plot
        ax[2, 1].set_ylabel('Rate')
        ax[2, 1].set_xlabel('Frames')
        ax[2, 1].legend(['Zero Crossing Rate'])
        
        # Save the figure as an image file
        output_file = os.path.splitext(audio_file_name)[0] + ".png"
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        
        print("Analysis complete. The plot has been saved as 'music_analysis_plot.png'.")
        
        close_app()
        
    except Exception as e:
        print(f"Error analyzing file: {e}")

# Function to close the application
def close_app():
    root.quit()
    root.destroy()

# Set up the GUI
root = tk.Tk()
root.title('Music Analyzer')
root.geometry('1200x1000')
root.configure(bg='#2E2E2E')  # Set background color

# Configure button styles
button_style = {'bg': '#444444', 'fg': 'white', 'activebackground': '#555555', 'activeforeground': 'white'}

load_button = tk.Button(root, text='Load Music File', command=load_file, **button_style)
load_button.pack(pady=10)

exit_button = tk.Button(root, text='Exit', command=close_app, **button_style)
exit_button.pack(pady=10)

root.mainloop()
