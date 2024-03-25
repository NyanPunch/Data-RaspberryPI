import requests

# POST 요청에 사용할 데이터
data = {'key1': 'value1', 'key2': 'value2'}

# POST 요청 보내기
response = requests.post('http://google.com', data=data)

# 응답 확인
print('Status Code:', response.status_code)
print('Response Body:', response.text)
