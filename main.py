import requests
from metar import Metar
import asyncio
import edge_tts
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

def fetch_metar(icao):
    """Fetches the current METAR from the NOAA public API"""
    icao = icao.upper()
    url = f"https://tgftp.nws.noaa.gov/data/observations/metar/stations/{icao}.TXT"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            if len(lines) >= 2:
                return lines[1]
        print(f"Error: Could not find weather data for airport: {icao}")
        return None
    except Exception as e:
        print(f"Connection error: {e}")
        return None

def build_atis_text(icao, metar_string):
    """Parses the METAR and builds the basic ATIS text transcript"""
    try:
        obs = Metar.Metar(metar_string)
        atis_text = f"Attention all aircraft, this is Information Alpha for {icao.upper()} airport. "
        atis_text += f"Weather observations. {obs.string()}"
        return atis_text
    except Exception as e:
        print(f"Parsing error: Could not decode METAR data. {e}")
        return None

def apply_radio_effect(input_filename, output_filename):
    """Applies a cockpit radio/headset filter using bandpass and light noise"""
    try:
        print("Applying cockpit radio effect...")
        # Read the generated clean speech file
        data, fs = sf.read(input_filename)
        
        # If stereo, convert to mono for processing
        if len(data.shape) > 1:
            data = data[:, 0]
            
        # Butter bandpass filter: limits frequencies between 300Hz and 3000Hz (Radio standard)
        lowcut = 300.0
        highcut = 3000.0
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(6, [low, high], btype='band')
        filtered_data = lfilter(b, a, data)
        
        # Add a very tiny static background noise (aviation hiss)
        noise = np.random.normal(0, 0.003, filtered_data.shape)
        radio_audio = filtered_data + noise
        
        # Boost volume slightly after filtering
        radio_audio = radio_audio * 1.5
        
        # Save the final file as WAV (safer without external dependencies)
        sf.write(output_filename, radio_audio, fs)
        print(f"[Success] Radio ATIS saved as: {output_filename}")
    except Exception as e:
        print(f"Radio effect error: {e}")

async def text_to_speech_ai(text, clean_file="clean.wav"):
    """Converts text to WAV using a high-quality native American female AI voice"""
    try:
        print("\nGenerating ATIS audio file with Native Female AI Voice...")
        # Using Ava for a crisp, 100% native American accent
        voice = "en-US-AvaNeural"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(clean_file)
    except Exception as e:
        print(f"Audio generation error: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    airport = input("Enter airport ICAO code (e.g., KLAX, LEMD, SPJC): ")
    metar_data = fetch_metar(airport)
    
    if metar_data:
        print(f"\n[Raw METAR]: {metar_data}")
        atis_transcript = build_atis_text(airport, metar_data)
        
        if atis_transcript:
            print(f"\n[Generated ATIS Transcript]:\n{atis_transcript}")
            
            # 1. Generate the clean native voice first
            asyncio.run(text_to_speech_ai(atis_transcript, "clean.wav"))
            
            # 2. Apply the cockpit filter and output the final radio file
            apply_radio_effect("clean.wav", "atis_radio.wav")