# Music Analyzer

## Author
Seisa Likotsi

## Description
Music Analyzer is a Python program that allows users to analyze an audio file and generate a series of plots that provide insights into the audio file's characteristics. The plots are saved as an image file.
This program has been tested only on Ubuntu.

## Features
- Load an audio file for analysis.
- Generate a series of plots including:
    - Waveform
    - Spectrogram
    - Tempo
    - Mel-frequency cepstral coefficients (MFCC)
    - Chroma feature
    - Zero crossing rate
    - Noise and distortion detection
    - Frequency balance
    - Spectral centroid (Timbre)
- Save the generated plots as an image file.

## Requirements
- Python 3.6 or later
- Libraries: tkinter, librosa, matplotlib, numpy, numba

## Usage
1. Run the program.
2. Click on the 'Load Music File' button to select an audio file for analysis,
   or click the 'Exit' button to exit the program.
3. The program will analyze the audio file and generate a series of plots.
4. The plots will be saved as an image file with the same name as the audio file (but with a .png extension).
5. A message will be printed to the console when the analysis is complete and the plot has been saved.
6. The program will then exit.


