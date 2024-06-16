import requests
import pymysql
import time
from datetime import datetime

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

def save_power(power, cost_saving, timestamp, total_volume):
    with connection.cursor() as cursor:
        sql = "INSERT INTO power_usage (timestamp, power, cost_saving) VALUES (%s, %s, %s);"
        cursor.execute(sql, (timestamp, power, cost_saving))
        connection.commit()
    print(f"Power {power} and cost saving {cost_saving} updated to MySQL. Current total_volume is: {total_volume}")

def fetch_power_usage(device_id, access_token, total_volume):
    url = f"https://api.smartthings.com/v1/devices/{device_id}/status"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        power_usage = data['components']['main']['powerMeter']['power']['value']  # Wh 단위
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cost_saving = calculate_cost_saving(power_usage, total_volume)
        save_power(power_usage, cost_saving, current_time, total_volume)
        print(f"Success at {current_time}!")
    else:
        print("Failed to fetch data:", response.status_code)

def fetch_and_send_power_usage_periodically(device_id, access_token):
    while True:
        with connection.cursor() as cursor:
            cursor.execute("SELECT total_volume FROM total_volume ORDER BY timestamp DESC LIMIT 1;")
            closest_volume_entry = cursor.fetchone()
        total_volume = closest_volume_entry['total_volume']
        fetch_power_usage(device_id, access_token, total_volume)
        time.sleep(60)

def calculate_cost_saving(power, total_volume):
    storage_percentage = (total_volume / 146.0) * 100

    # Wh 단위를 kWh 단위로 변환
    power_kwh = power / 1000.0

    # 전력량 감소율 계산
    if storage_percentage > 60:
        reduced_percentage = storage_percentage - 60
        reduced_power_kwh = power_kwh - (reduced_percentage * 0.005)
    else:
        reduced_power_kwh = power_kwh - (storage_percentage * 0.005)

    reduced_power_kwh = max(0, reduced_power_kwh)  # 전력 사용량이 음수가 되지 않도록 조정

    # KEPCO 요금 체계에 따른 원래 비용 및 감소된 비용 계산
    if reduced_power_kwh <= 200:
        reduced_cost = 910 + (reduced_power_kwh * 93.3)
    elif 200 < reduced_power_kwh <= 400:
        reduced_cost = 1600 + (reduced_power_kwh * 187.9)
    else:
        reduced_cost = 7300 + (reduced_power_kwh * 280.6)

    if power_kwh <= 200:
        original_cost = 910 + (power_kwh * 93.3)
    elif 200 < power_kwh <= 400:
        original_cost = 1600 + (power_kwh * 187.9)
    else:
        original_cost = 7300 + (power_kwh * 280.6)

    # 비용 절감액 계산 (한 달 기준으로 계산)
    cost_saving = (original_cost - reduced_cost) * 30
    return round(cost_saving, 1)  # 소수점 1자리까지 반올림

# Example usage
device_id = '15326ac6-b0b3-4875-801c-9273fa848676'
access_token = '927f5961-87fe-4cb0-89f4-675520460baf'
fetch_and_send_power_usage_periodically(device_id, access_token)
