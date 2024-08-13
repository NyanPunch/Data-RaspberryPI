from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# 데이터베이스 경로 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 데이터베이스 모델 정의
class Detection(db.Model):
    __tablename__ = 'detections'
    timestamp = db.Column(db.Text, primary_key=True)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    distance = db.Column(db.Float, nullable=False)

# 라우트 설정
@app.route('/')
def index():
    detections = Detection.query.all()  # 모든 데이터를 조회
    return render_template('index.html', detections=detections)

# 데이터베이스 생성
def create_tables():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    create_tables()  # 애플리케이션 실행 전에 테이블 생성
    app.run(debug=True)
