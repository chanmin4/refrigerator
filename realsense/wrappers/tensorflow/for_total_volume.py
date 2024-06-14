import os
import sys
import threading
import time
import json
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from scipy.spatial import distance
import requests
import uuid
import base64
import pymysql
from datetime import datetime, timezone, timedelta

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

def fetch_total_volume():
    with connection.cursor() as cursor:
        sql = "SELECT total_volume FROM total_volume ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(sql)
        result = cursor.fetchone()
        if result:
            return result['total_volume']
        return None

def save_total_volume(total_volume, timestamp):
    with connection.cursor() as cursor:
        sql = """
        INSERT INTO total_volume (timestamp, total_volume)
        VALUES (%s, %s)
        """
        cursor.execute(sql, (timestamp, total_volume))
        connection.commit()
    print("New total_volume inserted to MySQL.")

def periodic_task():
    while True:
        total_volume = fetch_total_volume()
        if total_volume is not None:
            current_time = time.time()
            timestamp_col = datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')                
                   
            save_total_volume(total_volume, timestamp_col)
        time.sleep(60)

if __name__ == "__main__":
    thread = threading.Thread(target=periodic_task)
    thread.daemon = True
    thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script terminated by user.")
        connection.close()
