from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sqlite3
import numpy as np
import mne
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
import csv
from io import StringIO
import base64
import io

app = Flask(__name__)
CORS(app)
dash_app = Dash(__name__, server=app, url_base_pathname='/')

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('eeg_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS eeg_signals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, time REAL, channel TEXT, voltage REAL)''')
    conn.commit()
    conn.close()

# Simulate EEG data
def simulate_eeg():
    fs = 256  # Sampling frequency (Hz)
    t = np.linspace(0, 10, 10 * fs)  # 10 seconds
    channels = ['Fz', 'Cz', 'Pz']  # Example EEG channels
    data = np.zeros((len(channels), len(t)))
    
    # Generate synthetic EEG with band-specific signals
    for i, _ in enumerate(channels):
        # Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
        data[i] = (0.5 * np.sin(2 * np.pi * 2 * t) +  # Delta
                   0.3 * np.sin(2 * np.pi * 6 * t) +  # Theta
                   0.4 * np.sin(2 * np.pi * 10 * t) + # Alpha
                   0.2 * np.sin(2 * np.pi * 20 * t) + # Beta
                   0.1 * np.sin(2 * np.pi * 40 * t))  # Gamma
        data[i] += np.random.normal(0, 0.05, len(t))   # Noise
    
    return t.tolist(), channels, data.tolist()

# Analyze EEG data with MNE
def analyze_eeg(time, channels, data):
    fs = 256
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(np.array(data), info)
    
    # Compute PSD
    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=0.5, fmax=100, n_fft=1024)
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-100 Hz)': (30, 100)
    }
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psd[:, idx], axis=1).tolist()
    
    return time, channels, data, band_powers

# Store EEG data in database
def store_eeg_data(time, channels, data):
    conn = sqlite3.connect('eeg_data.db')
    c = conn.cursor()
    c.execute('DELETE FROM eeg_signals')  # Clear previous data
    for i, t in enumerate(time):
        for j, ch in enumerate(channels):
            c.execute('INSERT INTO eeg_signals (time, channel, voltage) VALUES (?, ?, ?)', 
                      (t, ch, data[j][i]))
    conn.commit()
    conn.close()

@app.route('/upload_eeg', methods=['POST'])
def upload_eeg():
    try:
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be CSV'}), 400
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(content))
        header = next(csv_reader)  # Assume header: time, ch1, ch2, ...
        time = []
        channels = header[1:]  # Exclude time column
        data = [[] for _ in channels]
        for row in csv_reader:
            time.append(float(row[0]))
            for i, value in enumerate(row[1:]):
                data[i].append(float(value))
        store_eeg_data(time, channels, data)
        time, channels, data, band_powers = analyze_eeg(time, channels, data)
        return jsonify({'time': time, 'channels': channels, 'data': data, 'band_powers': band_powers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_eeg', methods=['GET'])
def get_eeg():
    try:
        time, channels, data = simulate_eeg()
        store_eeg_data(time, channels, data)
        time, channels, data, band_powers = analyze_eeg(time, channels, data)
        return jsonify({'time': time, 'channels': channels, 'data': data, 'band_powers': band_powers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dash layout
dash_app.layout = html.Div(className='container mx-auto p-4', children=[
    html.H1('BCI EEG Data Visualizer', className='text-2xl font-bold mb-4'),
    html.Div(className='mb-4', children=[
        html.Label('Upload EEG Data (CSV: time, ch1, ch2, ...)', className='block mb-1'),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV', className='bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600'),
            accept='.csv'
        ),
        html.Button('Load Sample EEG', id='load-sample', className='ml-2 bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600')
    ]),
    html.Div(id='output-data-upload'),
    dcc.Graph(id='eeg-signal'),
    dcc.Graph(id='band-power')
])

# Dash callbacks
@dash_app.callback(
    [Output('eeg-signal', 'figure'), Output('band-power', 'figure')],
    [Input('upload-data', 'contents'), Input('load-sample', 'n_clicks')],
    prevent_initial_call=True
)
def update_graphs(contents, n_clicks):
    time, channels, data, band_powers = [], [], [], {}
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        csv_reader = csv.reader(StringIO(decoded))
        header = next(csv_reader)
        time = []
        channels = header[1:]
        data = [[] for _ in channels]
        for row in csv_reader:
            time.append(float(row[0]))
            for i, value in enumerate(row[1:]):
                data[i].append(float(value))
        time, channels, data, band_powers = analyze_eeg(time, channels, data)
    else:
        time, channels, data = simulate_eeg()
        time, channels, data, band_powers = analyze_eeg(time, channels, data)
    
    # EEG signal plot
    eeg_figure = {
        'data': [go.Scatter(x=time, y=data[i], name=ch) for i, ch in enumerate(channels)],
        'layout': {
            'title': 'Raw EEG Signal',
            'xaxis': {'title': 'Time (s)'},
            'yaxis': {'title': 'Voltage (µV)'}
        }
    }
    
    # Band power plot
    band_figure = {
        'data': [
            go.Bar(
                x=channels,
                y=band_powers[band],
                name=band
            ) for band in band_powers
        ],
        'layout': {
            'title': 'Power Spectral Density by Frequency Band',
            'xaxis': {'title': 'Channel'},
            'yaxis': {'title': 'Power (µV²/Hz)'},
            'barmode': 'group'
        }
    }
    
    return eeg_figure, band_figure

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)