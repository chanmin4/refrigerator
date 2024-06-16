import requests
import pymysql
import time
from datetime import datetime, timedelta

# MySQL 연결 설정
connection = pymysql.connect(
    host='smart-fridge.cn8m88cosddm.us-east-1.rds.amazonaws.com',
    user='chanmin4',
    password='location1957',
    db='smart-fridge',
    charset='utf8mb4',
    port=3307,
    cursorclass=pymysql.cursors.DictCursor
)

def save_power(power, cost_saving, timestamp):
    with connection.cursor() as cursor:
        sql = "INSERT INTO power_usage (timestamp, power, cost_saving) VALUES (%s, %s, %s);"
        cursor.execute(sql, (timestamp, power, cost_saving))
        connection.commit()
    print(f"Power {power} and cost saving {cost_saving} updated to MySQL.")

def fetch_power_usage(device_id, access_token):
    url = f"https://api.smartthings.com/v1/devices/{device_id}/status"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        power_usage = data['components']['main']['powerMeter']['power']['value']
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cost_saving = calculate_cost_saving(power_usage, current_time)
        save_power(power_usage, cost_saving, current_time)
        print(f"Success at {current_time}!")
    else:
        print("Failed to fetch data:", response.status_code)

def fetch_and_send_power_usage_periodically(device_id, access_token):
    while True:
        fetch_power_usage(device_id, access_token)
        time.sleep(60)

def calculate_cost_saving(power, current_time):
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM total_volume ORDER BY ABS(TIMESTAMPDIFF(SECOND, timestamp, %s)) LIMIT 1;", (current_time,))
        closest_volume_entry = cursor.fetchone()

    total_volume = closest_volume_entry['total_volume']
    storage_percentage = (total_volume / 146.0) * 100

    if 0 <= storage_percentage <= 200:
        reduced_power = power * 0.01
    elif 201 <= storage_percentage <= 400:
        reduced_power = power * 0.02
    else:
        reduced_power = power * 0.03

    if reduced_power <= 200:
        reduced_cost = 910 + (reduced_power * 0.0933)
    elif 200 < reduced_power <= 400:
        reduced_cost = 1600 + (reduced_power * 0.1879)
    else:
        reduced_cost = 7300 + (reduced_power * 0.2806)

    if power <= 200:
        original_cost = 910 + (power * 0.0933)
    elif 200 < power <= 400:
        original_cost = 1600 + (power * 0.1879)
    else:
        original_cost = 7300 + (power * 0.2806)

    cost_saving = (original_cost - reduced_cost) * 525.6
    return cost_saving

# Example usage
device_id = '15326ac6-b0b3-4875-801c-9273fa848676'
access_token = '927f5961-87fe-4cb0-89f4-675520460baf'
fetch_and_send_power_usage_periodically(device_id, access_token)
