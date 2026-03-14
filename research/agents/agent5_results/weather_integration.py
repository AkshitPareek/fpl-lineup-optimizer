
# WEATHER FEATURE INTEGRATION
# Add these features to your model training

# Weather data structure (example)
weather_features = {
    'temperature': 15.0,  # Celsius
    'precipitation': 0.0,  # mm
    'wind_speed': 10.0,   # km/h
    'humidity': 70.0,     # %
    'is_rainy': 0,        # 0/1
    'is_windy': 0,        # 0/1 (wind > 20 km/h)
    'is_hot': 0,          # 0/1 (temp > 25C)
    'is_cold': 0,         # 0/1 (temp < 5C)
}

# Weather API integration (requires API key)
def get_weather_data(stadium, match_date, api_key):
    """
    Fetch weather data for match location.
    
    APIs to consider:
    - OpenWeatherMap (free tier available)
    - WeatherAPI
    - Visual Crossing
    """
    import requests
    
    # Example with OpenWeatherMap
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': stadium,
        'appid': api_key,
        'units': 'metric'
    }
    
    # response = requests.get(url, params=params)
    # data = response.json()
    
    # Return formatted features
    return {
        'temperature': 15.0,
        'wind_speed': 10.0,
        'precipitation': 0.0,
        'humidity': 70.0
    }

# Historical weather impact analysis
def analyze_weather_impact(player_data, weather_data):
    """
    Analyze how weather affects specific players.
    
    Some players perform better/worse in:
    - Rain (slippery conditions)
    - Wind (affects long shots/crosses)
    - Heat (fatigue)
    - Cold (muscle stiffness)
    """
    pass
