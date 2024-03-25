import requests

# GET 요청 보내기
response = requests.get('http://google.com')

# 응답 확인
print('Status Code:', response.status_code)
print('Response Body:', response.text)
