import requests
import json
import time
import pymysql
import random
from datetime import datetime,timezone, timedelta
# MySQL 연결 설정
connection = pymysql.connect(
    host='smart-fridge.cn8m88cosddm.us-east-1.rds.amazonaws.com',
    user='chanmin4',
    password='location1957',
    db='smart-fridge',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
def save_power(power, timestamp):
    with connection.cursor() as cursor:
        sql = "INSERT INTO power_usage (timestamp, power) VALUES (%s, %s);"
        cursor.execute(sql, (timestamp, power))
        connection.commit()
    print("power updated to MySQL.")

def fetch_power_usage(device_id, access_token):
    #smartthings id
    #여기 디바이스아이디조정
    url = f"https://api.smartthings.com/v1/devices/8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1/status"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        #power_usage = data['components']['main']['powerMeter']['power']['value']
        #send data
        #power_usage=random.random(30,50)
        #current_time = time.time()
        #current_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')           
        #save_power(power_usage,current_time_str)
        
        #print(f"Success at {current_time}!")
    else:
        print("Failed to fetch data:", response.status_code)
def fetch_and_send_power_usage_periodically(device_id, access_token):
    while True:
        power_usage=random.randrange(30,50)
        current_time = time.time()
        current_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')           
        save_power(power_usage,current_time_str)
        
        print(f"Success at {current_time_str}  {power_usage}!")
        #fetch_power_usage(device_id, access_token)
        time.sleep(5)  # 1800초는 30분을 의미합니다.
# Example usage
#smartthing id&token
#여기부분수정
#insert token between devices/token/status
device_id = '8aa1c32f-1cec-48fe-8d4d-7add3a0f96c1'
access_token = '927f5961-87fe-4cb0-89f4-675520460baf'
fetch_and_send_power_usage_periodically(device_id, access_token)
