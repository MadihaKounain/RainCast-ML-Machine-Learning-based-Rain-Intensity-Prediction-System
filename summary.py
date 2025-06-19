# utils/summary.py
def generate_weather_summary(intensity):
    if intensity < 1:
        return "☀️ Clear weather expected."
    elif intensity < 3:
        return "🌦️ Light rain expected."
    elif intensity < 6:
        return "🌧️ Moderate rain expected."
    else:
        return "⛈️ Heavy rain. Take precautions!"
