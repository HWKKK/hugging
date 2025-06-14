# 놈팽쓰(NomPangS) 기술 명세서

## 1. 시스템 개요

### 1.1 핵심 기능
- **성격 생성**: 4개 입력값(온기, 능력, 외향성, 시간) → 151개 세부 성격 변수 자동 생성
- **유머 매트릭스**: 3차원 좌표계 기반 유머 스타일 정의 및 실시간 조정
- **매력적 결함**: AI 기반 오브젝트 분석 + 성격 조합으로 개성적 단점 생성
- **대화 추적**: 키워드 자동 추출, 관계 발전 단계 계산, 맥락 유지

### 1.2 실제 테스트 데이터 성능
```
test1 (뭉이씨): 3회 대화, 8.67자 평균, "첫_만남" 단계, 키워드 5개
test0 (커피포트): 30회 대화, 37.03자 평균, "친밀한_관계" 단계, 키워드 207개
```

### 1.3 구현 구조
```
nompang_test/
├── app.py                          # 메인 애플리케이션
├── modules/
│   ├── persona_generator.py        # 성격 생성 + 유머 매트릭스
│   ├── image_analyzer.py          # 이미지 분석
│   └── conversation_manager.py    # 대화 관리
├── 뭉이씨_20250528_053729.json    # 생성된 페르소나 데이터  
└── conversation_history_*.json    # 대화 기록 및 분석
```

### 1.4 데이터 플로우
1. **입력**: 사용자 설정(온기75, 능력68, 외향성82) + 이미지 업로드 + 시간선택
2. **처리**: 151개 변수 계산 + 3D 유머좌표 + 매력적결함 AI생성
3. **출력**: JSON 페르소나 + 실시간 대화 + 관계발전 추적
4. **학습**: 대화패턴 분석 → 유머스타일 조정 → 성격 미세조정

---

## 2. 주요 기능 상세

### 2.1 성격 생성 시스템

#### A. 사용자 입력 (4개 조정값)
```
온기: 0-100 (따뜻함 ↔ 차가움)
능력: 0-100 (유능함 ↔ 순수함)
외향성: 0-100 (활발함 ↔ 내성적)
함께한 시간: 새것/몇 개월/1년 이상
```

#### B. 시스템 자동 생성 (151개 세부 변수)
```python
# 예시: 온기 80 설정 시 자동 계산
"W01_친절함": 82,        # 온기 기반 계산
"W02_친근함": 78,        # 온기 + 개성 변동
"C15_배려심": 65,        # 연관 효과
"E01_사교성": 88,        # 외향성 연동

# 151개 변수 카테고리 분포:
# - 기본 온기-능력 차원: 26개 (W10+C16)
# - 빅5 성격 특성: 30개 (E6+A6+N6+O6+성실성6)
# - 매력적 결함: 15개 (F01~F15)  
# - 모순적 특성: 16개 (P01~P16)
# - 소통 스타일: 20개 (S10+H10)
# - 관계 형성: 20개 (R10+D10)
# - 사물 특성: 24개 (OBJ8+FORM8+INT8)
```

### 2.2 유머 매트릭스

#### 3차원 좌표계
```
warmth_vs_wit: 따뜻한 유머 ↔ 지적 위트
self_vs_observational: 자기참조 ↔ 관찰형
subtle_vs_expressive: 미묘함 ↔ 과장됨
```

#### 파생 속성 자동 계산
```
풍자 수준 = (100 - 자기참조) × 0.7 + (100 - 따뜻함) × 0.3
부조리 수준 = 과장됨 × 0.8
말장난 빈도 = 지적위트 × 0.6 + 과장됨 × 0.4
```

### 2.3 매력적 결함 생성

#### 생성 과정
1. **이미지 분석**: 재질, 형태, 용도 파악
2. **성격 연계**: 151개 변수와 사물 특성 조합
3. **AI 생성**: Gemini API로 4개 결함 생성
4. **모순 추가**: 물리적 특성과 성격의 대조

#### 실제 예시 (전기 주전자 "뭉이씨")
```
매력적 결함:
- "물을 너무 빨리 끓여서 가끔 '벌컥'대는 모습이 창피해"
- "석회질 자국은 어쩔 수 없나 봐. 내 늙어감의 증거 같아"

모순적 특성:
- "차가운 금속 몸체로 뜨거운 물을 끓이는 아이러니"
```

### 2.4 대화 분석 시스템

#### 자동 추출 정보
```python
keywords = {
    "word": {
        "category": "음식/관계/취미/감정/기타",
        "frequency": 언급 횟수,
        "last_mentioned": "날짜"
    }
}

user_profile = {
    "message_count": 총 대화 횟수,
    "avg_message_length": 평균 문자수,
    "relationship_level": "첫_만남/알아가는_중/친숙함/친밀함",
    "emotional_tendency": "중립적/긍정적/부정적"
}
```

---

## 3. 샘플 대화 분석

### 3.1 test1 - 뭉이씨 (초기 단계)

#### 대화 내용
```
사용자: "ㄴ"
뭉이씨: "ㄴ? ㅋㅋㅋ 무슨 'ㄴ'이야! 혹시 나 지금 글씨 연습시키는 거야?"

사용자: "에휴 너 좀 시끄럽구나?"
뭉이씨: "에헤이! 😅 시끄럽다니, 섭섭한걸?",

사용자: "너는 뭐야 소개해줘"
뭉이씨: "와! 😄 내가 누군지 궁금해? 뭉이라고 해! 전기 주전자인데, 좀 특별한 영혼이 깃든?"
```

#### 자동 생성된 분석 결과
```json
{
  "message_count": 3,
  "avg_message_length": 8.67,
  "relationship_level": "첫_만남",
  "keywords": ["에휴", "시끄럽구나", "너는", "뭐야", "소개해줘"],
  "emotional_tendency": "중립적"
}
```

### 3.2 test0 - 커피포트 (발전된 단계)

#### 30회 대화 후 결과
```json
{
  "message_count": 30,
  "avg_message_length": 37.03,
  "relationship_level": "친밀한_관계",
  "keywords_count": 207개,
  "주요_키워드": {
    "커피": 5회,
    "차": 3회,
    "친구": 2회,
    "사회화": 2회
  }
}
```

#### 관계 발전 과정
```
1-10회: 기본 기능 중심 ("커피 타왔음", "청소꿀팁")
11-20회: 개인 가치관 공유 ("사회화되기 싫어서 노력했는데")
21-30회: 깊은 고민 토로 ("밍밍하게 사려고 노력하는데 영 재미가 없네")
```

### 3.3 유머 스타일 적응 사례

#### 실시간 조정 과정
```
대화 7회차:
사용자: "쿨시크를 좋아하지"
→ 시스템: 과장된 유머 → 쿨한 스타일로 즉시 조정

대화 9회차:
사용자: "솔직한게 좋아"
→ 시스템: 필터링된 표현 → 직설적 스타일로 전환
```

---

## 4. JSON 데이터 확인 방법

### 4.1 페르소나 정보 확인
```json
// 뭉이씨_20250528_053729.json
{
  "성격특성": {
    "온기": 75,      // 사용자 설정값
    "능력": 68,
    "외향성": 82
  },
  "성격변수151": {
    "W01_친절함": 78,  // 자동 계산된 세부 성격
    "H01_유머감각": 75,
    // ... 총 151개
  },
  "유머매트릭스": {
    "warmth_vs_wit": 45.925,    // 유머 스타일 좌표
    "sarcasm_level": 45.9725    // 풍자 수준
  }
}
```

### 4.2 대화 기록 확인
```json
// conversation_history_20250528_053814.json
{
  "conversations": [
    {
      "user_message": "에휴 너 좀 시끄럽구나?",
      "ai_response": "에헤이! 😅 시끄럽다니, 섭섭한걸?",
      "keywords": ["에휴", "시끄럽구나"],
      "sentiment": "중립적"
    }
  ],
  "user_profile": {
    "relationship_level": "첫_만남",  // 자동 계산된 관계 단계
    "message_count": 3
  }
}
```

---

## 5. 추후 활용 방향

### 5.1 현재 시스템의 한계
```
단순 지표:
- 대화 횟수만으로 관계 판단
- 키워드 단순 매칭
- 고정된 성격 변수

기능 한계:
- 유머 매트릭스 값이 실시간 변경되지 않음
- 시간 요소가 151개 변수에 직접 반영되지 않음
```

### 5.2 발전 방향

#### A. 다차원 관계 점수 시스템
```python
관계_점수 = {
    "물리적_상호작용": {
        "QR_스캔_횟수": 0,
        "앱_사용_시간": 0,
        "연속_접속_일수": 0
    },
    "대화_품질": {
        "깊은_대화_점수": AI_분석_기반,
        "감정_공유_빈도": 감정_키워드_빈도,
        "개인정보_공유": 사적_정보_수준
    },
    "지속성": {
        "주간_접속_빈도": 0,
        "월간_활동_점수": 0
    }
}
```

#### B. 키워드 분석 고도화
```python
# 현재: 단순 매칭
"커피" → "음식" 카테고리

# 목표: 의미 단위 분석
"스트레스 받을 때 커피 마셔" → {
    "상황": "스트레스",
    "행동": "커피",
    "의미": "스트레스_해소_방법",
    "감정": "부정적_상황_대처"
}
```

#### C. 게이미피케이션 확장
```python
관계_레벨링 = {
    1: "첫_만남" (0-100점),
    5: "친근한_사이" (500점),
    10: "절친" (2000점),
    20: "가족_같은_사이" (10000점)
}

점수_획득_방법 = {
    "QR_스캔": 50점,
    "깊은_대화": 100점,
    "연속_접속": 20점/일,
    "감정_공유": 80점
}
```

### 5.3 실제 서비스 확장

#### IoT 연동
```
스마트홈: 조명 + AI 페르소나
자동차: 운전 패턴 + 대화 동반자
웨어러블: 생체 신호 + 감정 상태 반영
```

#### B2B 활용
```
의료: 복약 관리 + 정서적 지지
교육: 학습 동기 + 개인화 교육
마케팅: 브랜드 스토리텔링
```

---

## 6. 구현 우선순위

### 단기 (3개월)
1. **시간 변수 통합**: 함께한 시간을 151개 변수에 직접 반영
2. **실시간 유머 조정**: 대화 피드백으로 매트릭스 값 업데이트
3. **키워드 significance**: 빈도+카테고리+맥락 종합 점수

### 중기 (6개월)
1. **다차원 관계 시스템**: 물리적+대화+지속성 종합 점수
2. **AI 의미 분석**: 문장 임베딩 기반 키워드 추출
3. **동적 성격 진화**: 대화 패턴 기반 성격 미세 조정

### 장기 (12개월)
1. **IoT 연동**: 실제 기기와 페르소나 융합
2. **멀티모달**: 음성, 이미지 입력 처리
3. **B2B 확장**: 의료, 교육, 마케팅 솔루션

---

**현재 허깅페이스 앱**: [nompang_test](https://huggingface.co/spaces/username/nompang_test)  
**테스트 데이터**: test0 (30회 대화), test1 (3회 대화)  
**핵심 파일**: app.py, modules/persona_generator.py
