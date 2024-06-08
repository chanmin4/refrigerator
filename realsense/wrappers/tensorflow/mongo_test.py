from pymongo import MongoClient

# MongoDB Atlas 연결 문자열
uri = "mongodb+srv://chanmin404:location1957!@refridge.g1vskrx.mongodb.net/test?retryWrites=true&w=majority"

# MongoDB 클라이언트 생성
client = MongoClient(uri)

# 연결 테스트
try:
    client.server_info()  # 서버 정보 요청
    print("Connected to MongoDB Atlas successfully")
except Exception as e:
    print("Could not connect to MongoDB Atlas:", e)
