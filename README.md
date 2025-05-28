---
title: 놈팽쓰(MemoryTag) 테스트 앱
emoji: 🎭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
---

# 놈팽쓰 (NomPang-S) - AI 페르소나 생성 시스템

151개 변수를 활용한 한국형 AI 페르소나 생성 및 대화 시스템

## 🚀 주요 기능

- **151개 변수 성격 시스템**: 온기, 능력, 외향성, 유머 등 세밀한 성격 설정
- **8가지 성격 유형**: 열정적 엔터테이너, 차가운 완벽주의자, 따뜻한 상담사 등
- **멀티 API 지원**: Gemini와 OpenAI GPT-4o/GPT-4o-mini 동시 지원
- **매력적 결함 시스템**: 완벽하지 않기에 더 매력적인 캐릭터
- **유머 매트릭스**: 3차원 유머 좌표계로 개성 있는 대화

## ⚙️ API 키 설정 (필수)

대화 기능을 사용하려면 **API 키를 반드시 설정해야 합니다.**

### 로컬 개발 환경

1. `.env` 파일 생성:
```bash
# .env 파일에 다음 내용 추가
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

2. API 키 발급 방법:
   - **Gemini API**: https://makersuite.google.com/app/apikey
   - **OpenAI API**: https://platform.openai.com/api-keys

### Hugging Face Spaces 환경

1. **Spaces Settings** → **Variables and secrets** 이동
2. 다음 환경변수 추가:
   ```
   GEMINI_API_KEY = your_gemini_api_key_here
   OPENAI_API_KEY = your_openai_api_key_here
   ```
3. **Restart Space** 클릭하여 적용

## 🔧 API 연결 테스트

```bash
python debug_api.py
```

## 💡 사용법

1. **이미지 업로드**: 페르소나를 만들고 싶은 사물의 사진 업로드
2. **API 설정**: 상단의 API 설정에서 Gemini 또는 OpenAI 선택 및 키 입력
3. **페르소나 생성**: 151개 변수로 구성된 고유한 성격 자동 생성
4. **대화하기**: 생성된 페르소나와 자연스러운 대화 시작

## 🎭 성격 유형

- **열정적 엔터테이너**: 온기↑ + 유머↑ + 외향성↑
- **차가운 완벽주의자**: 능력↑ + 온기↓ + 외향성↓  
- **따뜻한 상담사**: 온기↑ + 공감↑ + 유머↓
- **위트 넘치는 지식인**: 능력↑ + 유머↑ + 온기↓
- **수줍은 몽상가**: 외향성↓ + 창의성↑ + 온기=
- **카리스마틱 리더**: 능력↑ + 외향성↑ + 온기=
- **장난꾸러기 친구**: 유머↑ + 외향성↑ + 능력↓
- **신비로운 현자**: 창의성↑ + 외향성↓ + 능력↑

## 🐛 문제 해결

**"뭔가 문제가 생긴 것 같아" 메시지가 반복될 때:**
- API 키가 설정되지 않은 상태입니다
- 위의 API 키 설정 방법을 따라 설정하세요
- `python debug_api.py`로 연결 상태를 확인하세요

## 📦 의존성

```
gradio>=4.0.0
google-generativeai>=0.3.0
openai==1.54.3
pillow>=9.0.0
matplotlib>=3.5.0
python-dotenv>=0.19.0
```

## 🤝 기여

이슈 리포트와 풀 리퀘스트를 환영합니다!

---

**Made with ❤️ for Korean AI Persona Generation**

## 주요 기능

1. **영혼 깨우기**: 
   - 사물 이미지를 분석하여 물리적 특성 추출
   - 프론트엔드용 간단한 페르소나와 백엔드용 상세 페르소나 생성
   - 151개 성격 변수 생성 (백엔드 시스템)

2. **대화하기**:
   - 생성된 페르소나와 자연스러운 대화
   - 성격에 맞는 응답 생성
   - 대화 내역 저장

3. **페르소나 관리**:
   - 생성된 페르소나 저장 및 로드
   - 페르소나 목록 관리

## 설치 방법

1. 저장소를 클론합니다:
   ```bash
   git clone [저장소 URL]
   cd nompang_test
   ```

2. 필요한 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

3. `.env` 파일을 생성하고 Gemini API 키를 설정합니다:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## 실행 방법

앱을 실행하려면 다음 명령어를 사용합니다:

```bash
python app.py
```

웹 브라우저에서 `http://localhost:7860`으로 접속하여 앱을 사용할 수 있습니다.

## 사용 방법

1. **영혼 깨우기 탭**:
   - 사물 이미지를 업로드하거나 이름을 입력합니다.
   - "영혼 깨우기" 버튼을 클릭하여 페르소나를 생성합니다.
   - 프론트엔드 뷰와 백엔드 상세 정보를 탭으로 전환하여 확인할 수 있습니다.
   - "페르소나 저장" 버튼을 클릭하여 생성된 페르소나를 저장합니다.

2. **대화하기 탭**:
   - "새 대화 시작" 버튼을 클릭하여 현재 페르소나와 대화를 시작합니다.
   - 메시지를 입력하고 "전송" 버튼을 클릭하여 대화합니다.
   - "대화 초기화" 버튼을 클릭하여 대화 내역을 초기화할 수 있습니다.

3. **페르소나 관리 탭**:
   - "페르소나 목록 새로고침" 버튼을 클릭하여 저장된 페르소나 목록을 갱신합니다.
   - 목록에서 페르소나를 선택하고 "선택한 페르소나 불러오기" 버튼을 클릭하여 불러옵니다.
   - 불러온 페르소나의 정보를 확인하고 대화하기 탭으로 이동하여 대화할 수 있습니다.

## 시스템 구조

- **app.py**: 메인 Gradio 애플리케이션
- **modules/persona_generator.py**: 페르소나 생성 및 대화 처리
- **modules/data_manager.py**: 데이터 저장 및 로드
- **data/personas/**: 저장된 페르소나 데이터
- **data/conversations/**: 대화 내역 데이터

## 참고 사항

- 이 앱은 Gemini API를 사용하여 페르소나를 생성하고 대화합니다.
- API 키가 설정되지 않으면 기본 페르소나로 제한된 기능을 사용할 수 있습니다.
- 이미지 분석 결과는 Gemini API의 이미지 처리 기능을 사용합니다.

## 라이선스

MIT License

## 업데이트 정보

- Gradio SDK 버전을 5.31.0으로 업데이트했습니다.
- 탭 선택 로직이 업데이트되었습니다.
- ID 관련 오류를 수정했습니다. 