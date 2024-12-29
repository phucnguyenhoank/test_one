#%%
from dotenv import load_dotenv
import os


# %%
load_dotenv('.env', override=True)


# %%
api_key = os.getenv("WEATHER_API_KEY")

#%%
import requests

# %%
lat = 10.845485244046833
lon = 106.79671737706057
url = f"https://api.openweathermap.org/data/2.5/forecast?units=metric&cnt=1&lat={lat}&lon={lon}&appid={api_key}"

# Use the get function from the requests library to store the response from the API
response = requests.get(url, params={"appid": api_key})
data = response.json()


# %%
feels_like = data['list'][0]['main']['feels_like']
city = data['city']['name']
print(f"The temperature currently feels like {feels_like}°C in {city}.")


# %%
