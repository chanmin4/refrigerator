import time
import adafruit_dht
import board
import pymysql
import time
from datetime import datetime,timezone, timedelta
connection = pymysql.connect(
    host='smart-fridge.cn8m88cosddm.us-east-1.rds.amazonaws.com',
    user='chanmin4',
    password='location1957',
    db='smart-fridge',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
def save_temp_humid(temp,humid, timestamp):
    with connection.cursor() as cursor:
        sql = "INSERT INTO temp_humid (timestamp, humidity,temperature) VALUES (%s, %s,%s)"
        cursor.execute(sql, (timestamp, humid,temp))
        connection.commit()
    print("humid,temp updated to MySQL.")

dht_device=adafruit_dht.DHT22(board.D4)
while(True):
    try:
        temperature = dht_device.temperature
        humidity=dht_device.humidity
        print(f"Temp: {temperature:.1f} C Humidity:{humidity:.1f}%")
        current_time = time.time()
        current_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
            
        save_temp_humid(temperature,humidity,current_time_str)
    except RuntimeError as error:
        print(error.args[0])
        
    time.sleep(30.0)
    
    
    
    
