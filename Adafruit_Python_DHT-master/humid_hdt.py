import time
#import adafruit_dht
#import board
import pymysql
import time
from datetime import datetime,timezone, timedelta
import random
connection = pymysql.connect(
    host='smart-fridge.cn8m88cosddm.us-east-1.rds.amazonaws.com',
    user='chanmin4',
    password='location1957',
    db='smart-fridge',
    charset='utf8mb4',
    port=3307,
    cursorclass=pymysql.cursors.DictCursor
)
def save_temp_humid(temp,humid, timestamp):
    with connection.cursor() as cursor:
        #1개값들어있어야함
        sql = """
        UPDATE temp_humid
        SET timestamp = %s, humidity = %s, temperature = %s
        WHERE id = 1
        """
        cursor.execute(sql, (timestamp, humid,temp))
        connection.commit()
    print("humid,temp updated to MySQL.")

#dht_device=adafruit_dht.DHT22(board.D4)
while(True):
    try:
        #temperature = dht_device.temperature
        #humidity=dht_device.humidity
        #print(f"Temp: {temperature:.1f} C Humidity:{humidity:.1f}%")
        current_time = time.time()
        current_time_str = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')
            
        #save_temp_humid(temperature,humidity,current_time_str)
        temp_humid=random.randrange(58,62)
        temp_temp=random.randrange(8,10)
        save_temp_humid(temp_temp,temp_humid,current_time_str)
        print(temp_temp,temp_humid)
    except RuntimeError as error:
        print(error.args[0])
        
    time.sleep(60.0)
    
    
    
    
