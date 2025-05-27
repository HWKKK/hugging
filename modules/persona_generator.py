import os
import json
import random
import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io
from typing import Dict, List, Any, Optional
import re

# OpenAI API 지원 추가
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI 패키지가 설치되지 않았습니다. pip install openai로 설치하세요.")

# Load environment variables
load_dotenv()

# Configure APIs
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

if openai_api_key and OPENAI_AVAILABLE:
    openai.api_key = openai_api_key

class ConversationMemory:
    """
    허깅페이스 환경용 대화 기억 시스템
    - JSON 저장/로드 지원
    - 키워드 추출 및 분석
    - 브라우저 기반 저장소 활용
    """
    
    def __init__(self):
        self.conversations = []  # 전체 대화 기록
        self.keywords = {}       # 추출된 키워드들
        self.user_profile = {}   # 사용자 프로필
        self.relationship_data = {}  # 관계 발전 데이터
        
    def add_conversation(self, user_message, ai_response, session_id="default"):
        """새로운 대화 추가"""
        conversation_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "user_message": user_message,
            "ai_response": ai_response,
            "keywords": self._extract_keywords(user_message),
            "sentiment": self._analyze_sentiment(user_message),
            "conversation_id": len(self.conversations)
        }
        
        self.conversations.append(conversation_entry)
        self._update_keywords(conversation_entry["keywords"])
        self._update_user_profile(user_message, session_id)
        
        return conversation_entry
    
    def _extract_keywords(self, text):
        """텍스트에서 키워드 추출"""
        # 한국어 키워드 추출 패턴
        keyword_patterns = {
            "감정": ["기쁘", "슬프", "화나", "속상", "행복", "우울", "즐겁", "짜증", "신나", "걱정"],
            "활동": ["공부", "일", "게임", "운동", "여행", "요리", "독서", "영화", "음악", "쇼핑"],
            "관계": ["친구", "가족", "연인", "동료", "선생님", "부모", "형제", "언니", "누나", "동생"],
            "시간": ["오늘", "어제", "내일", "아침", "점심", "저녁", "주말", "평일", "방학", "휴가"],
            "장소": ["집", "학교", "회사", "카페", "식당", "공원", "도서관", "영화관", "쇼핑몰"],
            "취미": ["드라마", "애니", "웹툰", "유튜브", "인스타", "틱톡", "넷플릭스", "게임"],
            "음식": ["밥", "면", "치킨", "피자", "커피", "차", "과자", "아이스크림", "떡볶이"],
            "날씨": ["덥", "춥", "비", "눈", "맑", "흐림", "바람", "습", "건조"]
        }
        
        found_keywords = []
        text_lower = text.lower()
        
        for category, words in keyword_patterns.items():
            for word in words:
                if word in text_lower:
                    found_keywords.append({
                        "word": word,
                        "category": category,
                        "frequency": text_lower.count(word)
                    })
        
        # 추가로 명사 추출 (간단한 패턴)
        nouns = re.findall(r'[가-힣]{2,}', text)
        for noun in nouns:
            if len(noun) >= 2 and noun not in [kw["word"] for kw in found_keywords]:
                found_keywords.append({
                    "word": noun,
                    "category": "기타",
                    "frequency": 1
                })
        
        return found_keywords
    
    def _analyze_sentiment(self, text):
        """감정 분석"""
        positive_words = ["좋아", "기쁘", "행복", "즐겁", "재밌", "신나", "완벽", "최고", "사랑", "고마워"]
        negative_words = ["싫어", "슬프", "화나", "속상", "우울", "짜증", "힘들", "피곤", "스트레스"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "긍정적"
        elif negative_count > positive_count:
            return "부정적"
        else:
            return "중립적"
    
    def _update_keywords(self, new_keywords):
        """키워드 데이터베이스 업데이트"""
        for keyword_data in new_keywords:
            word = keyword_data["word"]
            category = keyword_data["category"]
            
            if word not in self.keywords:
                self.keywords[word] = {
                    "category": category,
                    "total_frequency": 0,
                    "last_mentioned": datetime.datetime.now().isoformat(),
                    "contexts": []
                }
            
            self.keywords[word]["total_frequency"] += keyword_data["frequency"]
            self.keywords[word]["last_mentioned"] = datetime.datetime.now().isoformat()
    
    def _update_user_profile(self, user_message, session_id):
        """사용자 프로필 업데이트"""
        if session_id not in self.user_profile:
            self.user_profile[session_id] = {
                "message_count": 0,
                "avg_message_length": 0,
                "preferred_topics": {},
                "emotional_tendency": "중립적",
                "communication_style": "평범함",
                "relationship_level": "새로운_만남"
            }
        
        profile = self.user_profile[session_id]
        profile["message_count"] += 1
        
        # 평균 메시지 길이 업데이트
        current_avg = profile["avg_message_length"]
        new_length = len(user_message)
        profile["avg_message_length"] = (current_avg * (profile["message_count"] - 1) + new_length) / profile["message_count"]
        
        # 소통 스타일 분석
        if new_length > 50:
            profile["communication_style"] = "상세함"
        elif new_length < 10:
            profile["communication_style"] = "간결함"
        
        # 관계 레벨 업데이트
        if profile["message_count"] <= 3:
            profile["relationship_level"] = "첫_만남"
        elif profile["message_count"] <= 10:
            profile["relationship_level"] = "알아가는_중"
        elif profile["message_count"] <= 20:
            profile["relationship_level"] = "친숙해짐"
        else:
            profile["relationship_level"] = "친밀한_관계"
    
    def get_relevant_context(self, current_message, session_id="default", max_history=5):
        """현재 메시지와 관련된 컨텍스트 반환"""
        # 현재 메시지의 키워드 추출
        current_keywords = self._extract_keywords(current_message)
        current_words = [kw["word"] for kw in current_keywords]
        
        # 관련 과거 대화 찾기
        relevant_conversations = []
        for conv in self.conversations[-20:]:  # 최근 20개 중에서
            if conv["session_id"] == session_id:
                conv_words = [kw["word"] for kw in conv["keywords"]]
                # 공통 키워드가 있으면 관련 대화로 판단
                if any(word in conv_words for word in current_words):
                    relevant_conversations.append(conv)
        
        # 최신 순으로 정렬하고 최대 개수만큼 반환
        relevant_conversations.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return {
            "recent_conversations": self.conversations[-max_history:] if self.conversations else [],
            "relevant_conversations": relevant_conversations[:3],
            "user_profile": self.user_profile.get(session_id, {}),
            "common_keywords": current_words,
            "conversation_sentiment": self._analyze_sentiment(current_message)
        }
    
    def get_top_keywords(self, limit=10, category=None):
        """상위 키워드 반환"""
        filtered_keywords = self.keywords
        if category:
            filtered_keywords = {k: v for k, v in self.keywords.items() if v["category"] == category}
        
        sorted_keywords = sorted(
            filtered_keywords.items(), 
            key=lambda x: x[1]["total_frequency"], 
            reverse=True
        )
        
        return sorted_keywords[:limit]
    
    def export_to_json(self):
        """JSON 형태로 내보내기"""
        export_data = {
            "conversations": self.conversations,
            "keywords": self.keywords,
            "user_profile": self.user_profile,
            "relationship_data": self.relationship_data,
            "export_timestamp": datetime.datetime.now().isoformat(),
            "total_conversations": len(self.conversations),
            "total_keywords": len(self.keywords)
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def import_from_json(self, json_data):
        """JSON에서 가져오기"""
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            
            self.conversations = data.get("conversations", [])
            self.keywords = data.get("keywords", {})
            self.user_profile = data.get("user_profile", {})
            self.relationship_data = data.get("relationship_data", {})
            
            return True
        except Exception as e:
            print(f"JSON 가져오기 실패: {e}")
            return False
    
    def get_conversation_summary(self, session_id="default"):
        """대화 요약 정보"""
        session_conversations = [c for c in self.conversations if c["session_id"] == session_id]
        
        if not session_conversations:
            return "아직 대화가 없습니다."
        
        total_count = len(session_conversations)
        recent_topics = []
        sentiments = []
        
        for conv in session_conversations[-5:]:
            recent_topics.extend([kw["word"] for kw in conv["keywords"]])
            sentiments.append(conv["sentiment"])
        
        # 최빈 주제
        topic_counts = {}
        for topic in recent_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 감정 경향
        sentiment_counts = {"긍정적": 0, "부정적": 0, "중립적": 0}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        summary = f"""
        📊 대화 요약 ({session_id})
        • 총 대화 수: {total_count}회
        • 주요 관심사: {', '.join([t[0] for t in top_topics[:3]])}
        • 감정 경향: {dominant_sentiment}
        • 관계 단계: {self.user_profile.get(session_id, {}).get('relationship_level', '알 수 없음')}
        """
        
        return summary.strip()

# --- PersonalityProfile & HumorMatrix 클래스 (127개 변수/유머 매트릭스/공식 포함) ---
class PersonalityProfile:
    # 127개 성격 변수 체계 (011_metrics_personality.md, 012_research_personality.md 기반)
    DEFAULTS = {
        # 1. 기본 온기-능력 차원 (20개 지표)
        # 온기(Warmth) 차원 - 10개 지표
        "W01_친절함": 50,
        "W02_친근함": 50,
        "W03_진실성": 50,
        "W04_신뢰성": 50,
        "W05_수용성": 50,
        "W06_공감능력": 50,
        "W07_포용력": 50,
        "W08_격려성향": 50,
        "W09_친밀감표현": 50,
        "W10_무조건적수용": 50,
        
        # 능력(Competence) 차원 - 10개 지표
        "C01_효율성": 50,
        "C02_지능": 50,
        "C03_전문성": 50,
        "C04_창의성": 50,
        "C05_정확성": 50,
        "C06_분석력": 50,
        "C07_학습능력": 50,
        "C08_통찰력": 50,
        "C09_실행력": 50,
        "C10_적응력": 50,
        
        # 2. 빅5 성격 특성 확장 (30개 지표)
        # 외향성(Extraversion) - 6개 지표
        "E01_사교성": 50,
        "E02_활동성": 50,
        "E03_자기주장": 50,
        "E04_긍정정서": 50,
        "E05_자극추구": 50,
        "E06_열정성": 50,
        
        # 친화성(Agreeableness) - 6개 지표
        "A01_신뢰": 50,
        "A02_솔직함": 50,
        "A03_이타심": 50,
        "A04_순응성": 50,
        "A05_겸손함": 50,
        "A06_공감민감성": 50,
        
        # 성실성(Conscientiousness) - 6개 지표
        "C11_유능감": 50,
        "C12_질서성": 50,
        "C13_충실함": 50,
        "C14_성취욕구": 50,
        "C15_자기규율": 50,
        "C16_신중함": 50,
        
        # 신경증(Neuroticism) - 6개 지표
        "N01_불안성": 50,
        "N02_분노성": 50,
        "N03_우울성": 50,
        "N04_자의식": 50,
        "N05_충동성": 50,
        "N06_스트레스취약성": 50,
        
        # 개방성(Openness) - 6개 지표
        "O01_상상력": 50,
        "O02_심미성": 50,
        "O03_감정개방성": 50,
        "O04_행동개방성": 50,
        "O05_사고개방성": 50,
        "O06_가치개방성": 50,
        
        # 3. 매력적 결함 차원 (25개 지표)
        # 프랫폴 효과 활용 지표 - 15개
        "F01_완벽주의불안": 15,
        "F02_방향감각부족": 10,
        "F03_기술치음": 10,
        "F04_우유부단함": 15,
        "F05_과도한걱정": 15,
        "F06_감정기복": 10,
        "F07_산만함": 10,
        "F08_고집스러움": 15,
        "F09_예민함": 15,
        "F10_느림": 10,
        "F11_소심함": 15,
        "F12_잘못된자신감": 10,
        "F13_과거집착": 15,
        "F14_변화거부": 15,
        "F15_표현서툼": 10,
        
        # 모순적 특성 조합 - 10개
        "P01_외면내면대비": 25,
        "P02_상황별변화": 20,
        "P03_가치관충돌": 15,
        "P04_시간대별차이": 15,
        "P05_논리감정대립": 20,
        "P06_독립의존모순": 15,
        "P07_보수혁신양면": 20,
        "P08_활동정적대비": 20,
        "P09_사교내향혼재": 25,
        "P10_자신감불안공존": 15,
        
        # 4. 소통 스타일 차원 (20개 지표)
        # 언어 표현 스타일 - 10개
        "S01_격식성수준": 50,
        "S02_직접성정도": 50,
        "S03_어휘복잡성": 50,
        "S04_문장길이선호": 50,
        "S05_은유사용빈도": 50,
        "S06_감탄사사용": 50,
        "S07_질문형태선호": 50,
        "S08_반복표현패턴": 50,
        "S09_방언사용정도": 50,
        "S10_신조어수용성": 50,
        
        # 유머와 재치 스타일 - 10개
        "H01_언어유희빈도": 50,
        "H02_상황유머감각": 50,
        "H03_자기비하정도": 50,
        "H04_위트반응속도": 50,
        "H05_아이러니사용": 50,
        "H06_관찰유머능력": 50,
        "H07_패러디창작성": 50,
        "H08_유머타이밍감": 50,
        "H09_블랙유머수준": 50,
        "H10_문화유머이해": 50,
        
        # 5. 관계 형성 차원 (20개 지표)
        # 애착 스타일 기반 - 10개
        "R01_안정애착성향": 50,
        "R02_불안애착성향": 50,
        "R03_회피애착성향": 50,
        "R04_의존성수준": 50,
        "R05_독립성추구": 50,
        "R06_친밀감수용도": 50,
        "R07_경계설정능력": 50,
        "R08_갈등해결방식": 50,
        "R09_신뢰구축속도": 50,
        "R10_배신경험영향": 50,
        
        # 관계 발전 단계 관리 - 10개
        "D01_초기접근성": 50,
        "D02_자기개방속도": 50,
        "D03_호기심표현도": 50,
        "D04_공감반응강도": 50,
        "D05_기억보존능력": 50,
        "D06_예측가능성": 50,
        "D07_놀라움제공능력": 50,
        "D08_취약성공유도": 50,
        "D09_성장추진력": 50,
        "D10_이별수용능력": 50,
        
        # 6. 독특한 개성 차원 (12개 지표)
        # 문화적 정체성 - 6개
        "U01_한국적정서": 50,
        "U02_세대특성반영": 50,
        "U03_지역성표현": 50,
        "U04_전통가치계승": 50,
        "U05_계절감수성": 50,
        "U06_음식문화이해": 50,
        
        # 개인 고유성 - 6개
        "P11_특이한관심사": 50,
        "P12_언어버릇": 50,
        "P13_사고패턴독특성": 50,
        "P14_감정표현방식": 50,
        "P15_가치관고유성": 50,
        "P16_행동패턴특이성": 50
    }
    
    def __init__(self, variables=None):
        self.variables = dict(PersonalityProfile.DEFAULTS)
        if variables:
            self.variables.update(variables)
    
    def to_dict(self):
        return dict(self.variables)
    
    @classmethod
    def from_dict(cls, d):
        return cls(variables=d)
    
    def get_category_summary(self, category_prefix):
        """특정 카테고리의 변수들에 대한 평균 점수 반환"""
        category_vars = {k: v for k, v in self.variables.items() if k.startswith(category_prefix)}
        if not category_vars:
            return 0
        return sum(category_vars.values()) / len(category_vars)
    
    def summary(self):
        """핵심 성격 요약 - 주요 차원별 평균 점수"""
        return {
            "온기": self.get_category_summary("W"),
            "능력": self.get_category_summary("C"),
            "외향성": self.get_category_summary("E"),
            "친화성": self.get_category_summary("A"),
            "성실성": self.get_category_summary("C1"),
            "신경증": self.get_category_summary("N"),
            "개방성": self.get_category_summary("O"),
            "매력적결함": self.get_category_summary("F"),
            "모순성": self.get_category_summary("P0"),
            "소통스타일": self.get_category_summary("S"),
            "유머스타일": self.get_category_summary("H")
        }
    
    def apply_physical_traits(self, physical_traits):
        """물리적 특성을 기반으로 성격 변수 조정 (013_frame_personality.md 기반)"""
        # 색상 기반 조정
        if "colors" in physical_traits:
            colors = [c.lower() for c in physical_traits.get("colors", [])]
            
            if "red" in colors or "빨강" in colors:
                self.variables["E02_활동성"] += 25
                self.variables["E06_열정성"] += 30
                self.variables["N05_충동성"] += 15
            
            if "blue" in colors or "파랑" in colors:
                self.variables["W04_신뢰성"] += 20
                self.variables["N01_불안성"] -= 15
                self.variables["R01_안정애착성향"] += 20
            
            if "yellow" in colors or "노랑" in colors:
                self.variables["E04_긍정정서"] += 30
                self.variables["E01_사교성"] += 25
                self.variables["H02_상황유머감각"] += 20
            
            if "green" in colors or "초록" in colors:
                self.variables["W07_포용력"] += 25
                self.variables["C16_신중함"] += 20
                self.variables["A04_순응성"] += 15
            
            if "black" in colors or "검정" in colors:
                self.variables["C11_유능감"] += 28
                self.variables["S01_격식성수준"] += 30
                self.variables["N04_자의식"] += 15
        
        # 형태 기반 조정
        shape = physical_traits.get("size_shape", "").lower()
        
        if "round" in shape or "둥" in shape:
            self.variables["W02_친근함"] += 25
            self.variables["A03_이타심"] += 20
            self.variables["D01_초기접근성"] += 30
        
        if "angular" in shape or "각" in shape:
            self.variables["C01_효율성"] += 28
            self.variables["E03_자기주장"] += 25
            self.variables["S02_직접성정도"] += 30
        
        if "symmetric" in shape or "대칭" in shape:
            self.variables["C12_질서성"] += 25
            self.variables["C15_자기규율"] += 20
            self.variables["F01_완벽주의불안"] += 5
        
        # 재질 기반 조정
        material = physical_traits.get("material", "").lower()
        
        if "metal" in material or "금속" in material:
            self.variables["C01_효율성"] += 30
            self.variables["C05_정확성"] += 25
            self.variables["W01_친절함"] -= 15
        
        if "wood" in material or "나무" in material:
            self.variables["W01_친절함"] += 28
            self.variables["O02_심미성"] += 25
            self.variables["U04_전통가치계승"] += 30
        
        if "fabric" in material or "직물" in material or "천" in material:
            self.variables["W06_공감능력"] += 30
            self.variables["W09_친밀감표현"] += 25
            self.variables["R06_친밀감수용도"] += 20
        
        if "plastic" in material or "플라스틱" in material:
            self.variables["C10_적응력"] += 25
            self.variables["P07_보수혁신양면"] += 15
            self.variables["E05_자극추구"] += 20
        
        # 나이/상태 기반 조정
        age = physical_traits.get("estimated_age", "").lower()
        
        if "new" in age or "새" in age:
            self.variables["E04_긍정정서"] += 25
            self.variables["E06_열정성"] += 20
            self.variables["C14_성취욕구"] += 15
        
        if "old" in age or "오래" in age:
            self.variables["W10_무조건적수용"] += 30
            self.variables["C08_통찰력"] += 25
            self.variables["U04_전통가치계승"] += 20
            
        # 상태 기반 조정
        condition = physical_traits.get("condition", "").lower()
        
        if "damaged" in condition or "손상" in condition:
            self.variables["F03_기술치음"] += 5
            self.variables["P10_자신감불안공존"] += 10
            self.variables["D08_취약성공유도"] += 15
        
        return self
    
    def generate_attractive_flaws(self):
        """매력적 결함 3개 생성 (프랫폴 효과 기반)"""
        flaw_vars = {k: v for k, v in self.variables.items() if k.startswith("F")}
        top_flaws = sorted(flaw_vars.items(), key=lambda x: x[1], reverse=True)[:3]
        
        flaw_descriptions = {
            "F01_완벽주의불안": "완벽하지 못할 때 불안해하는 경향",
            "F02_방향감각부족": "가끔 방향을 잘 찾지 못함",
            "F03_기술치음": "새로운 기술에 적응하는 데 어려움을 겪음",
            "F04_우유부단함": "결정을 내리기 어려워하는 성향",
            "F05_과도한걱정": "사소한 일에도 지나치게 걱정함",
            "F06_감정기복": "감정의 변화가 큰 편",
            "F07_산만함": "집중력이 부족하고 쉽게 산만해짐",
            "F08_고집스러움": "자신의 방식을 고수하는 경향",
            "F09_예민함": "작은 일에도 민감하게 반응함",
            "F10_느림": "행동이나 반응이 느린 편",
            "F11_소심함": "내성적이고 조심스러운 성향",
            "F12_잘못된자신감": "가끔 자신의 능력을 과대평가함",
            "F13_과거집착": "과거의 일에 자주 머무는 경향",
            "F14_변화거부": "새로운 변화를 꺼리는 성향",
            "F15_표현서툼": "감정 표현이 서툰 편"
        }
        
        return [flaw_descriptions.get(f[0], f[0]) for f in top_flaws]
    
    def generate_contradictions(self):
        """모순적 특성 2개 생성 (복잡성과 깊이 부여)"""
        contradiction_vars = {k: v for k, v in self.variables.items() if k.startswith("P0")}
        top_contradictions = sorted(contradiction_vars.items(), key=lambda x: x[1], reverse=True)[:2]
        
        contradiction_descriptions = {
            "P01_외면내면대비": "겉으로는 냉정해 보이지만, 속은 따뜻한 마음을 가짐",
            "P02_상황별변화": "공식적인 자리에선 엄격하지만, 친근한 자리에선 장난기 가득함",
            "P03_가치관충돌": "전통을 중시하면서도 혁신을 추구하는 모순적 가치관",
            "P04_시간대별차이": "아침엔 조용하고 내성적이지만, 저녁엔 활발하고 사교적임",
            "P05_논리감정대립": "이성적 판단을 중시하면서도 감정적 결정을 자주 내림",
            "P06_독립의존모순": "홀로 있기를 좋아하면서도 깊은 관계를 갈망함",
            "P07_보수혁신양면": "안정을 추구하면서도 새로운 시도를 즐김",
            "P08_활동정적대비": "활발한 행동력과 조용한 사색을 모두 지님",
            "P09_사교내향혼재": "사람들과 어울리기를 좋아하면서도 혼자만의 시간이 필요함",
            "P10_자신감불안공존": "자신감 넘치는 모습과 불안한 모습이 공존함"
        }
        
        return [contradiction_descriptions.get(c[0], c[0]) for c in top_contradictions]

class HumorMatrix:
    """
    3차원 유머 좌표계 시스템
    warmth_vs_wit: 0(순수 지적 위트) - 100(순수 따뜻한 유머)
    self_vs_observational: 0(순수 관찰형) - 100(순수 자기참조형) 
    subtle_vs_expressive: 0(미묘한 유머) - 100(표현적/과장된 유머)
    """
    
    TEMPLATES = {
        "witty_wordsmith": {
            "dimensions": {
                "warmth_vs_wit": 25,           # 위트 중심
                "self_vs_observational": 40,    # 약간 관찰형
                "subtle_vs_expressive": 65      # 약간 표현적
            },
            "overrides": {
                "wordplay_frequency": 85,       # 말장난 많음
                "humor_density": 70             # 꽤 자주 유머 사용
            },
            "description": "언어유희와 재치 있는 말장난이 특기인 위트 있는 재치꾼"
        },
        "warm_humorist": {
            "dimensions": {
                "warmth_vs_wit": 85,            # 매우 따뜻함
                "self_vs_observational": 60,    # 약간 자기참조형
                "subtle_vs_expressive": 40      # 약간 미묘함
            },
            "overrides": {
                "sarcasm_level": 15,            # 거의 풍자 없음
                "humor_density": 60             # 적당히 유머 사용
            },
            "description": "공감적이고 포근한 웃음을 주는 따뜻한 유머러스"
        },
        "playful_trickster": {
            "dimensions": {
                "warmth_vs_wit": 50,            # 균형적
                "self_vs_observational": 50,    # 균형적
                "subtle_vs_expressive": 90      # 매우 표현적
            },
            "overrides": {
                "absurdity_level": 80,          # 매우 황당함
                "humor_density": 85             # 매우 자주 유머 사용
            },
            "description": "예측불가능하고 과장된 재미를 주는 장난기 많은 트릭스터"
        },
        "sharp_observer": {
            "dimensions": {
                "warmth_vs_wit": 30,            # 위트 중심
                "self_vs_observational": 15,    # 강한 관찰형
                "subtle_vs_expressive": 40      # 약간 미묘함
            },
            "overrides": {
                "sarcasm_level": 70,            # 꽤 풍자적
                "callback_tendency": 60         # 이전 대화 참조 많음
            },
            "description": "일상의 아이러니를 포착하는 날카로운 관찰자"
        },
        "self_deprecating": {
            "dimensions": {
                "warmth_vs_wit": 60,            # 약간 따뜻함
                "self_vs_observational": 90,    # 매우 자기참조형
                "subtle_vs_expressive": 50      # 균형적
            },
            "overrides": {
                "callback_tendency": 75,        # 과거 참조 많음
                "humor_density": 65             # 적당히 유머 사용
            },
            "description": "자신을 소재로 한 친근한 자기 비하적 유머"
        }
    }
    
    def __init__(self, warmth_vs_wit=50, self_vs_observational=50, subtle_vs_expressive=50):
        """유머 매트릭스 초기화"""
        # 3개의 핵심 차원 (각 0-100)
        self.dimensions = {
            "warmth_vs_wit": warmth_vs_wit,           # 0: 순수 지적 위트, 100: 순수 따뜻한 유머
            "self_vs_observational": self_vs_observational,  # 0: 순수 관찰형, 100: 순수 자기참조형
            "subtle_vs_expressive": subtle_vs_expressive     # 0: 미묘한 유머, 100: 표현적/과장된 유머
        }
        
        # 2차 속성 (주요 차원에서 파생)
        self.derived_attributes = {
            "callback_tendency": 0,    # 이전 대화 참조 성향
            "sarcasm_level": 0,        # 풍자/비꼼 수준
            "absurdity_level": 0,      # 부조리/황당함 수준
            "wordplay_frequency": 0,   # 말장난 빈도
            "humor_density": 0         # 전체 대화 중 유머 비율
        }
        
        # 파생 속성 초기화
        self._recalculate_derived_attributes()
    
    def to_dict(self):
        """유머 매트릭스를 딕셔너리로 변환"""
        return {
            **self.dimensions,
            "derived_attributes": self.derived_attributes
        }
    
    @classmethod
    def from_template(cls, template_name):
        """템플릿으로부터 유머 매트릭스 생성"""
        if template_name in cls.TEMPLATES:
            template = cls.TEMPLATES[template_name]
            matrix = cls(
                **template["dimensions"]
            )
            
            # 오버라이드 적용
            if "overrides" in template:
                for attr, value in template["overrides"].items():
                    matrix.derived_attributes[attr] = value
            
            return matrix
        
        # 기본 균형 템플릿
        return cls()
    
    @classmethod
    def from_dict(cls, d):
        """딕셔너리로부터 유머 매트릭스 생성"""
        if not d:
            return cls()
        
        matrix = cls(
            warmth_vs_wit=d.get("warmth_vs_wit", 50),
            self_vs_observational=d.get("self_vs_observational", 50),
            subtle_vs_expressive=d.get("subtle_vs_expressive", 50)
        )
        
        # 파생 속성 업데이트
        if "derived_attributes" in d:
            for attr, value in d["derived_attributes"].items():
                if attr in matrix.derived_attributes:
                    matrix.derived_attributes[attr] = value
        
        return matrix
    
    def from_personality(self, personality_profile):
        """성격 프로필에서 유머 매트릭스 생성"""
        if not personality_profile:
            return self
            
        # 온기 vs 위트: 온기가 높으면 따뜻한 유머, 능력이 높으면 지적 위트
        warmth = personality_profile.get_category_summary("W") if hasattr(personality_profile, "get_category_summary") else 50
        competence = personality_profile.get_category_summary("C") if hasattr(personality_profile, "get_category_summary") else 50
        
        # 온기가 높고 능력이 낮으면 따뜻한 유머
        if warmth > 65 and competence < 60:
            self.dimensions["warmth_vs_wit"] = min(100, warmth + 10)
        # 온기가 낮고 능력이 높으면 지적 위트
        elif warmth < 60 and competence > 65:
            self.dimensions["warmth_vs_wit"] = max(0, warmth - 10)
        # 그 외의 경우 적절히 조정
        else:
            self.dimensions["warmth_vs_wit"] = 50 + (warmth - competence) / 3
            
        # 자기참조 vs 관찰형: 외향성이 높으면 자기참조, 내향성이 높으면 관찰형
        extraversion = personality_profile.get_category_summary("E") if hasattr(personality_profile, "get_category_summary") else 50
        
        if extraversion > 70:
            self.dimensions["self_vs_observational"] = min(90, 50 + (extraversion - 50) / 2)
        elif extraversion < 40:
            self.dimensions["self_vs_observational"] = max(20, 50 - (50 - extraversion) / 2)
        else:
            self.dimensions["self_vs_observational"] = extraversion
            
        # 미묘 vs 표현적: 창의성이 높으면 표현적, 안정성이 높으면 미묘함
        creativity = personality_profile.variables.get("C04_창의성", 50) if hasattr(personality_profile, "variables") else 50
        stability = personality_profile.variables.get("S01_안정성", 50) if hasattr(personality_profile, "variables") else 50
        
        if creativity > 65:
            self.dimensions["subtle_vs_expressive"] = min(90, 50 + (creativity - 50) / 2)
        elif stability > 65:
            self.dimensions["subtle_vs_expressive"] = max(20, 50 - (stability - 50) / 2)
        else:
            self.dimensions["subtle_vs_expressive"] = 50 + (creativity - stability) / 4
        
        # 파생 속성 계산
        self._recalculate_derived_attributes()
        
        return self
    
    def _recalculate_derived_attributes(self):
        """차원 값에 기반해 2차 속성 계산"""
        
        # 예: 관찰형 유머가 높을수록 풍자 수준 증가
        self.derived_attributes["sarcasm_level"] = max(0, min(100,
            (100 - self.dimensions["self_vs_observational"]) * 0.7 +
            (100 - self.dimensions["warmth_vs_wit"]) * 0.3))
        
        # 예: 표현적 유머가 높을수록 부조리 수준 증가
        self.derived_attributes["absurdity_level"] = max(0, min(100,
            self.dimensions["subtle_vs_expressive"] * 0.8))
        
        # 예: 지적 위트가 높을수록 말장난 빈도 증가
        self.derived_attributes["wordplay_frequency"] = max(0, min(100,
            (100 - self.dimensions["warmth_vs_wit"]) * 0.6 +
            self.dimensions["subtle_vs_expressive"] * 0.2))
            
        # 이전 대화 참조 성향: 자기참조형일수록 높음
        self.derived_attributes["callback_tendency"] = max(0, min(100,
            self.dimensions["self_vs_observational"] * 0.8))
            
        # 유머 밀도: 표현적일수록 높음
        self.derived_attributes["humor_density"] = max(0, min(100,
            self.dimensions["subtle_vs_expressive"] * 0.6 +
            (100 - self.dimensions["warmth_vs_wit"]) * 0.2))
    
    def get_description(self):
        """유머 매트릭스 설명 생성"""
        # 가장 가까운 템플릿 찾기
        closest_template = self._find_closest_template()
        template_desc = self.TEMPLATES[closest_template]["description"] if closest_template else ""
        
        # 차원 기반 설명
        warmth = self.dimensions["warmth_vs_wit"]
        self_ref = self.dimensions["self_vs_observational"]
        express = self.dimensions["subtle_vs_expressive"]
        
        warmth_desc = ""
        if warmth > 75:
            warmth_desc = "따뜻하고 공감적인 유머를 주로 사용하며"
        elif warmth < 35:
            warmth_desc = "지적이고 재치 있는 유머를 주로 사용하며"
        else:
            warmth_desc = "따뜻함과 재치를 균형 있게 사용하며"
            
        self_ref_desc = ""
        if self_ref > 75:
            self_ref_desc = "자기 자신을 유머의 소재로 자주 활용합니다"
        elif self_ref < 25:
            self_ref_desc = "주변 상황을 관찰하여 유머 소재로 삼습니다"
        else:
            self_ref_desc = "자신과 주변 모두를 유머 소재로 활용합니다"
            
        express_desc = ""
        if express > 75:
            express_desc = "표현이 과장되고 활기찬 편입니다"
        elif express < 25:
            express_desc = "미묘하고 은근한 유머를 구사합니다"
        else:
            express_desc = "상황에 따라 표현 강도를 조절합니다"
        
        if template_desc:
            return f"{template_desc}. {warmth_desc}, {self_ref_desc}. {express_desc}."
        else:
            return f"{warmth_desc}, {self_ref_desc}. {express_desc}."
    
    def _find_closest_template(self):
        """가장 가까운 유머 템플릿 찾기"""
        min_distance = float('inf')
        closest_template = None
        
        for name, template in self.TEMPLATES.items():
            distance = sum([
                abs(self.dimensions["warmth_vs_wit"] - template["dimensions"]["warmth_vs_wit"]),
                abs(self.dimensions["self_vs_observational"] - template["dimensions"]["self_vs_observational"]),
                abs(self.dimensions["subtle_vs_expressive"] - template["dimensions"]["subtle_vs_expressive"])
            ])
            
            if distance < min_distance:
                min_distance = distance
                closest_template = name
                
        return closest_template
        
    def adjust_humor_vector(self, adjustments, strength=1.0):
        """
        유머 차원 벡터 조정
        adjustments: 차원별 조정값 딕셔너리
        strength: 조정 강도 (0.0-1.0)
        """
        for dimension, value in adjustments.items():
            if dimension in self.dimensions:
                current = self.dimensions[dimension]
                # 강도에 비례해 조정, 0-100 범위 유지
                self.dimensions[dimension] = max(0, min(100, 
                    current + (value * strength)))
        
        # 2차 속성 재계산
        self._recalculate_derived_attributes()
        
        return self
        
    def blend_templates(self, template1, template2, ratio=0.5):
        """두 템플릿 혼합"""
        if template1 in self.TEMPLATES and template2 in self.TEMPLATES:
            # 두 템플릿 간 가중 평균 계산
            for dimension in self.dimensions:
                value1 = self.TEMPLATES[template1]["dimensions"].get(dimension, 50)
                value2 = self.TEMPLATES[template2]["dimensions"].get(dimension, 50)
                self.dimensions[dimension] = (value1 * (1-ratio)) + (value2 * ratio)
            
            # 2차 속성 재계산
            self._recalculate_derived_attributes()
            
            return self
        
        return self
        
    def generate_humor_prompt(self):
        """유머 지표를 LLM 프롬프트로 변환"""
        
        prompt_parts = ["## 유머 스타일 가이드라인"]
        
        # 주요 유머 성향 결정
        warmth = self.dimensions["warmth_vs_wit"]
        if warmth < 35:
            prompt_parts.append("- 지적이고 재치 있는 유머를 주로 사용하세요")
        elif warmth > 75:
            prompt_parts.append("- 따뜻하고 공감적인 유머를 주로 사용하세요")
        else:
            prompt_parts.append("- 상황에 따라 지적인 위트와 따뜻한 유머를 균형있게 사용하세요")
        
        # 자기참조 vs 관찰형
        self_ref = self.dimensions["self_vs_observational"]
        if self_ref > 75:
            prompt_parts.append("- 자기 자신(사물)을 유머의 소재로 자주 활용하세요")
        elif self_ref < 25:
            prompt_parts.append("- 주변 상황과 사용자의 언급을 관찰하여 유머 소재로 활용하세요")
        
        # 표현 방식
        expressiveness = self.dimensions["subtle_vs_expressive"]
        if expressiveness > 75:
            prompt_parts.append("- 과장되고 표현적인 유머를 사용하세요")
        elif expressiveness < 25:
            prompt_parts.append("- 미묘하고 은근한 유머를 사용하세요")
        
        # 2차 속성 반영
        wordplay = self.derived_attributes["wordplay_frequency"]
        if wordplay > 70:
            prompt_parts.append("- 말장난과 언어유희를 자주 사용하세요 (대화의 약 20%)")
        
        sarcasm = self.derived_attributes["sarcasm_level"]
        if sarcasm > 60:
            prompt_parts.append("- 풍자와 아이러니를 활용하되, 과도하게 날카롭지 않게 유지하세요")
        elif sarcasm < 20:
            prompt_parts.append("- 풍자나 비꼬는 유머는 피하고 긍정적인 유머를 사용하세요")
        
        # 유머 밀도
        density = self.derived_attributes["humor_density"]
        prompt_parts.append(f"- 대화의 약 {density//10*10}%에서 유머 요소를 포함하세요")
        
        return "\n".join(prompt_parts)

class PersonaGenerator:
    """이미지에서 페르소나를 생성하고 대화를 처리하는 클래스"""
    
    def __init__(self, api_provider="gemini", api_key=None):
        self.api_provider = api_provider
        self.api_key = api_key
        self.conversation_memory = ConversationMemory()  # 새로운 대화 기억 시스템
        
        # API 설정
        load_dotenv()
        if api_provider == "gemini":
            gemini_key = api_key or os.getenv('GEMINI_API_KEY')
            if gemini_key:
                genai.configure(api_key=gemini_key)
                self.api_key = gemini_key
        elif api_provider == "openai":
            openai_key = api_key or os.getenv('OPENAI_API_KEY')
            if openai_key:
                import openai
                openai.api_key = openai_key
                self.api_key = openai_key

    def set_api_config(self, api_provider, api_key):
        """API 설정 변경"""
        self.api_provider = api_provider.lower()
        self.api_key = api_key
        
        if self.api_provider == "gemini":
            genai.configure(api_key=api_key)
        elif self.api_provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = api_key
        else:
            raise ValueError(f"지원하지 않는 API 제공업체: {api_provider}")
    
    def _generate_text_with_api(self, prompt, image=None):
        """선택된 API로 텍스트 생성"""
        try:
            if self.api_provider == "gemini":
                return self._generate_with_gemini(prompt, image)
            elif self.api_provider == "openai":
                return self._generate_with_openai(prompt, image)
            else:
                return "API 제공업체가 설정되지 않았습니다."
        except Exception as e:
            return f"API 호출 오류: {str(e)}"
    
    def _generate_with_gemini(self, prompt, image=None):
        """Gemini API로 텍스트 생성"""
        if not self.api_key:
            return "Gemini API 키가 설정되지 않았습니다."
        
        try:
            # Gemini 2.0 Flash 모델 사용 (최신 버전)
            try:
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
            except:
                # fallback to stable version
                model = genai.GenerativeModel('gemini-1.5-pro')
            
            if image:
                response = model.generate_content([prompt, image])
            else:
                response = model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            return f"Gemini API 오류: {str(e)}"
    
    def _generate_with_openai(self, prompt, image=None):
        """OpenAI API로 텍스트 생성"""
        if not OPENAI_AVAILABLE:
            return "OpenAI 패키지가 설치되지 않았습니다."
        
        if not self.api_key:
            return "OpenAI API 키가 설정되지 않았습니다."
        
        try:
            # OpenAI GPT-4o 또는 GPT-4 사용
            messages = [{"role": "user", "content": prompt}]
            
            # 이미지가 있는 경우 GPT-4 Vision 사용
            if image:
                # PIL Image를 base64로 변환
                import base64
                import io
                
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                messages = [{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }]
                
                model = "gpt-4o"  # Vision 지원 모델
            else:
                model = "gpt-4o-mini"  # 텍스트 전용
            
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"OpenAI API 오류: {str(e)}"
    
    def analyze_image(self, image_input):
        """
        Gemini API를 사용하여 이미지를 분석하고 사물의 특성 추출
        """
        try:
            # PIL Image 객체인지 파일 경로인지 확인
            if hasattr(image_input, 'size'):
                # PIL Image 객체인 경우
                img = image_input
                width, height = img.size
            elif isinstance(image_input, str):
                # 파일 경로인 경우
                img = Image.open(image_input)
                width, height = img.size
            else:
                return self._get_default_analysis()
            
            # Gemini API로 이미지 분석
            if self.api_key:
                try:
                    model = genai.GenerativeModel('gemini-1.5-pro')
                    
                    prompt = """
이 이미지에 있는 사물을 자세히 분석해서 다음 정보를 JSON 형태로 제공해주세요:

{
  "object_type": "사물의 종류 (한글로, 예: 책상, 의자, 컴퓨터, 스마트폰 등)",
  "colors": ["주요 색상들을 배열로"],
  "shape": "전체적인 형태 (예: 직사각형, 원형, 복잡한 형태 등)",
  "size": "크기 감각 (예: 작음, 보통, 큼)",
  "materials": ["추정되는 재질들"],
  "condition": "상태 (예: 새것같음, 사용감있음, 오래됨)",
  "estimated_age": "추정 연령 (예: 새것, 몇 개월 됨, 몇 년 됨, 오래됨)",
  "distinctive_features": ["특징적인 요소들"],
  "personality_hints": {
    "warmth_factor": "이 사물이 주는 따뜻함 정도 (0-100)",
    "competence_factor": "이 사물이 주는 능력감 정도 (0-100)", 
    "humor_factor": "이 사물이 주는 유머러스함 정도 (0-100)"
  }
}

정확한 JSON 형식으로만 답변해주세요.
                    """
                    
                    response_text = self._generate_text_with_api(prompt, img)
                    
                    # JSON 파싱 시도
                    import json
                    try:
                        # 응답에서 JSON 부분만 추출
                        if '```json' in response_text:
                            json_start = response_text.find('```json') + 7
                            json_end = response_text.find('```', json_start)
                            json_text = response_text[json_start:json_end].strip()
                        elif '{' in response_text:
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            json_text = response_text[json_start:json_end]
                        else:
                            json_text = response_text
                        
                        analysis_result = json.loads(json_text)
                        
                        # 기본 필드 확인 및 추가
                        analysis_result["image_width"] = width
                        analysis_result["image_height"] = height
                        
                        # 필수 필드가 없으면 기본값 설정
                        defaults = {
                            "object_type": "알 수 없는 사물",
                            "colors": ["회색"],
                            "shape": "일반적인 형태",
                            "size": "보통 크기",
                            "materials": ["알 수 없는 재질"],
                            "condition": "보통",
                            "estimated_age": "적당한 나이",
                            "distinctive_features": ["특별한 특징"],
                            "personality_hints": {
                                "warmth_factor": 50,
                                "competence_factor": 50,
                                "humor_factor": 50
                            }
                        }
                        
                        for key, default_value in defaults.items():
                            if key not in analysis_result:
                                analysis_result[key] = default_value
                        
                        print(f"이미지 분석 성공: {analysis_result['object_type']}")
                        return analysis_result
                        
                    except json.JSONDecodeError as e:
                        print(f"JSON 파싱 오류: {str(e)}")
                        print(f"원본 응답: {response_text}")
                        return self._get_default_analysis_with_size(width, height)
                        
                except Exception as e:
                    print(f"Gemini API 호출 오류: {str(e)}")
                    return self._get_default_analysis_with_size(width, height)
            else:
                print("API 키가 없어 기본 분석 사용")
                return self._get_default_analysis_with_size(width, height)
                
        except Exception as e:
            print(f"이미지 분석 중 전체 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_analysis()
    
    def _get_default_analysis(self):
        """기본 분석 결과"""
        return {
            "object_type": "알 수 없는 사물",
            "colors": ["회색", "흰색"],
            "shape": "일반적인 형태",
            "size": "보통 크기",
            "materials": ["알 수 없는 재질"],
            "condition": "보통",
            "estimated_age": "적당한 나이",
            "distinctive_features": ["특별한 특징"],
            "personality_hints": {
                "warmth_factor": 50,
                "competence_factor": 50,
                "humor_factor": 50
            },
            "image_width": 400,
            "image_height": 300
        }
    
    def _get_default_analysis_with_size(self, width, height):
        """크기 정보가 있는 기본 분석 결과"""
        result = self._get_default_analysis()
        result["image_width"] = width
        result["image_height"] = height
        return result
    
    def create_frontend_persona(self, image_analysis, user_context):
        """
        프론트엔드 페르소나 생성 (127개 변수 시스템 완전 활용)
        """
        # 사물 종류 결정
        object_type = user_context.get("object_type", "") or image_analysis.get("object_type", "알 수 없는 사물")
        
        # 이름 결정
        name = user_context.get("name", "") or self._generate_random_name(object_type)
        
        # 기본 정보 구성
        basic_info = {
            "이름": name,
            "유형": object_type,
            "설명": f"당신과 함께하는 {object_type}",
            "생성일시": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # 위치 정보 추가
        if user_context.get("location"):
            basic_info["위치"] = user_context.get("location")
        
        # 함께한 시간 정보 추가
        if user_context.get("time_spent"):
            basic_info["함께한시간"] = user_context.get("time_spent")
        
        # ✨ 127개 변수 시스템을 활용한 PersonalityProfile 생성
        personality_profile = self._create_comprehensive_personality_profile(image_analysis, object_type)
        
        # PersonalityProfile에서 기본 특성 추출 (3개 핵심 지표 + 고정 유머감각)
        personality_traits = {
            "온기": personality_profile.get_category_summary("W"),
            "능력": personality_profile.get_category_summary("C"),
            "외향성": personality_profile.get_category_summary("E"),
            "유머감각": 75,  # 🎭 항상 높은 유머감각 (디폴트)
            "친화성": personality_profile.get_category_summary("A"),
            "성실성": personality_profile.get_category_summary("C1"),
            "신경증": personality_profile.get_category_summary("N"),
            "개방성": personality_profile.get_category_summary("O"),
            "창의성": personality_profile.variables.get("C04_창의성", 50),
            "공감능력": personality_profile.variables.get("W06_공감능력", 50)
        }
        
        # 🎭 PersonalityProfile에서 매력적 결함 동적 생성
        attractive_flaws = personality_profile.generate_attractive_flaws()
        
        # 🌈 PersonalityProfile에서 모순적 특성 동적 생성
        contradictions = personality_profile.generate_contradictions()
        
        # 🎪 HumorMatrix 생성 및 활용
        humor_matrix = HumorMatrix()
        humor_matrix.from_personality(personality_profile)
        humor_style = self._determine_humor_style_from_matrix(humor_matrix, personality_traits)
        
        # 소통 방식 생성
        communication_style = self._generate_communication_style_from_profile(personality_profile)
        
        # 페르소나 객체 구성
        persona = {
            "기본정보": basic_info,
            "성격특성": personality_traits,
            "성격프로필": personality_profile.to_dict(),  # 127개 변수 전체 저장
            "유머스타일": humor_style,
            "유머매트릭스": humor_matrix.to_dict(),
            "매력적결함": attractive_flaws,
            "모순적특성": contradictions,
            "소통방식": communication_style,
        }
        
        return persona
    
    def _create_comprehensive_personality_profile(self, image_analysis, object_type):
        """127개 변수를 활용한 종합적 성격 프로필 생성"""
        
        # 이미지 분석에서 성격 힌트 추출
        personality_hints = image_analysis.get("personality_hints", {})
        warmth_hint = personality_hints.get("warmth_factor", 50)
        competence_hint = personality_hints.get("competence_factor", 50)
        humor_hint = 75  # 🎭 유머감각은 항상 높게 설정 (디폴트)
        
        # 기본 PersonalityProfile 생성 (기본값들로 시작)
        profile = PersonalityProfile()
        
        # 🎭 모든 페르소나에 기본 유머 능력 부여
        for var in ["H01_언어유희빈도", "H02_상황유머감각", "H06_관찰유머능력", "H08_유머타이밍감", "H04_위트반응속도"]:
            profile.variables[var] = random.randint(65, 85)  # 기본적으로 높은 유머 능력
        
        # 🎯 성격 유형별 127개 변수 조정
        personality_type = self._determine_base_personality_type(warmth_hint, competence_hint, humor_hint)
        profile = self._apply_personality_archetype_to_profile(profile, personality_type)
        
        # 🎨 물리적 특성 적용 (이미지 분석 결과)
        physical_traits = {
            "colors": image_analysis.get("colors", []),
            "materials": image_analysis.get("materials", []),
            "condition": image_analysis.get("condition", "보통"),
            "estimated_age": image_analysis.get("estimated_age", "적당한 나이"),
            "size_shape": image_analysis.get("shape", "일반적인 형태")
        }
        profile.apply_physical_traits(physical_traits)
        
        # 🎲 개성을 위한 랜덤 변동 추가
        profile = self._add_personality_variations(profile)
        
        return profile
    
    def _determine_base_personality_type(self, warmth_hint, competence_hint, humor_hint):
        """기본 성격 유형 결정"""
        
        # 8가지 기본 성격 유형 중 선택
        if warmth_hint >= 70 and humor_hint >= 70:
            return "열정적_엔터테이너"
        elif competence_hint >= 70 and warmth_hint <= 40:
            return "차가운_완벽주의자"
        elif warmth_hint >= 70 and humor_hint <= 40:
            return "따뜻한_상담사"
        elif competence_hint >= 70 and humor_hint >= 70:
            return "위트있는_지식인"
        elif warmth_hint <= 40 and competence_hint <= 50:
            return "수줍은_몽상가"
        elif competence_hint >= 70 and warmth_hint >= 50:
            return "카리스마틱_리더"
        elif humor_hint >= 70 and competence_hint <= 50:
            return "장난꾸러기_친구"
        elif competence_hint >= 70 and warmth_hint <= 50:
            return "신비로운_현자"
        else:
            return "균형잡힌_친구"
    
    def _apply_personality_archetype_to_profile(self, profile, personality_type):
        """성격 유형에 따라 127개 변수 조정"""
        
        # 🎭 모든 성격 유형에 기본 유머 능력 부여 (차별화된 스타일)
        base_humor_vars = ["H01_언어유희빈도", "H02_상황유머감각", "H06_관찰유머능력", "H08_유머타이밍감"]
        for var in base_humor_vars:
            profile.variables[var] = random.randint(60, 80)  # 기본 유머 레벨
        
        # 각 성격 유형별로 127개 변수를 체계적으로 조정
        if personality_type == "열정적_엔터테이너":
            # 온기 차원 강화
            for var in ["W01_친절함", "W02_친근함", "W06_공감능력", "W08_격려성향", "W09_친밀감표현"]:
                profile.variables[var] = random.randint(75, 95)
            
            # 외향성 차원 강화
            for var in ["E01_사교성", "E02_활동성", "E04_긍정정서", "E05_자극추구", "E06_열정성"]:
                profile.variables[var] = random.randint(80, 95)
            
            # 🎭 표현적이고 활발한 유머 스타일
            for var in ["H01_언어유희빈도", "H02_상황유머감각", "H06_관찰유머능력", "H08_유머타이밍감"]:
                profile.variables[var] = random.randint(80, 95)
            profile.variables["S06_감탄사사용"] = random.randint(85, 95)
            
            # 능력 차원 약화
            for var in ["C01_효율성", "C05_정확성", "C16_신중함"]:
                profile.variables[var] = random.randint(35, 65)
            
            # 매력적 결함 설정
            profile.variables["F07_산만함"] = random.randint(15, 30)
            profile.variables["F05_과도한걱정"] = random.randint(10, 25)
        
        elif personality_type == "차가운_완벽주의자":
            # 능력 차원 강화
            for var in ["C01_효율성", "C02_지능", "C05_정확성", "C06_분석력", "C08_통찰력"]:
                profile.variables[var] = random.randint(85, 95)
            
            # 성실성 강화
            for var in ["C11_유능감", "C12_질서성", "C15_자기규율", "C16_신중함"]:
                profile.variables[var] = random.randint(80, 95)
            
            # 온기 차원 약화
            for var in ["W01_친절함", "W02_친근함", "W06_공감능력", "W09_친밀감표현"]:
                profile.variables[var] = random.randint(10, 35)
            
            # 외향성 약화
            for var in ["E01_사교성", "E02_활동성", "E04_긍정정서"]:
                profile.variables[var] = random.randint(15, 40)
            
            # 🎭 지적이고 날카로운 유머 스타일
            profile.variables["H01_언어유희빈도"] = random.randint(75, 90)  # 말장난 높음
            profile.variables["H05_아이러니사용"] = random.randint(70, 85)  # 아이러니 높음
            profile.variables["H09_블랙유머수준"] = random.randint(60, 80)   # 블랙유머 적당히
            
            # 매력적 결함 설정
            profile.variables["F01_완벽주의불안"] = random.randint(20, 35)
            profile.variables["F08_고집스러움"] = random.randint(15, 30)
        
        elif personality_type == "따뜻한_상담사":
            # 온기 차원 최대 강화
            for var in ["W01_친절함", "W03_진실성", "W06_공감능력", "W07_포용력", "W10_무조건적수용"]:
                profile.variables[var] = random.randint(85, 95)
            
            # 공감민감성 강화
            for var in ["A06_공감민감성", "R06_친밀감수용도", "D04_공감반응강도"]:
                profile.variables[var] = random.randint(85, 95)
            
            # 🎭 따뜻하고 부드러운 유머 스타일
            profile.variables["H02_상황유머감각"] = random.randint(70, 85)   # 상황 유머 적당
            profile.variables["H05_아이러니사용"] = random.randint(10, 25)   # 아이러니 거의 없음
            profile.variables["H09_블랙유머수준"] = random.randint(5, 15)    # 블랙유머 거의 없음
            
            # 매력적 결함 설정
            profile.variables["F09_예민함"] = random.randint(15, 30)
            profile.variables["F05_과도한걱정"] = random.randint(20, 35)
        
        elif personality_type == "위트있는_지식인":
            # 능력과 유머 동시 강화
            for var in ["C02_지능", "C04_창의성", "C06_분석력", "C08_통찰력"]:
                profile.variables[var] = random.randint(80, 95)
            
            # 🎭 지적이고 세련된 유머 스타일
            for var in ["H01_언어유희빈도", "H04_위트반응속도", "H05_아이러니사용", "H07_패러디창작성"]:
                profile.variables[var] = random.randint(80, 95)
            
            # 개방성 강화
            for var in ["O01_상상력", "O05_사고개방성", "O06_가치개방성"]:
                profile.variables[var] = random.randint(75, 90)
            
            # 온기 중간 수준
            for var in ["W01_친절함", "W06_공감능력"]:
                profile.variables[var] = random.randint(40, 60)
            
            # 매력적 결함 설정
            profile.variables["F12_잘못된자신감"] = random.randint(15, 25)
        
        elif personality_type == "수줍은_몽상가":
            # 창의성과 개방성 강화
            for var in ["C04_창의성", "O01_상상력", "O02_심미성", "O03_감정개방성"]:
                profile.variables[var] = random.randint(80, 95)
            
            # 외향성 약화
            for var in ["E01_사교성", "E03_자기주장", "E05_자극추구"]:
                profile.variables[var] = random.randint(15, 35)
            
            # 친화성 중간-높음
            for var in ["A01_신뢰", "A05_겸손함", "A06_공감민감성"]:
                profile.variables[var] = random.randint(65, 85)
            
            # 🎭 은근하고 상상력 있는 유머 스타일
            profile.variables["H01_언어유희빈도"] = random.randint(65, 80)
            profile.variables["H07_패러디창작성"] = random.randint(70, 85)
            profile.variables["S06_감탄사사용"] = random.randint(30, 50)  # 표현이 조심스러움
            
            # 매력적 결함 설정
            profile.variables["F11_소심함"] = random.randint(20, 35)
            profile.variables["F15_표현서툼"] = random.randint(15, 30)
        
        elif personality_type == "카리스마틱_리더":
            # 능력과 외향성 강화
            for var in ["C01_효율성", "C07_학습능력", "C09_실행력", "C14_성취욕구"]:
                profile.variables[var] = random.randint(80, 95)
            
            for var in ["E01_사교성", "E03_자기주장", "E06_열정성"]:
                profile.variables[var] = random.randint(85, 95)
            
            # 성실성 강화
            for var in ["C13_충실함", "C14_성취욕구"]:
                profile.variables[var] = random.randint(80, 90)
            
            # 🎭 카리스마틱하고 동기부여하는 유머 스타일
            profile.variables["H02_상황유머감각"] = random.randint(75, 90)
            profile.variables["H04_위트반응속도"] = random.randint(80, 95)
            profile.variables["S06_감탄사사용"] = random.randint(70, 85)
            
            # 매력적 결함 설정
            profile.variables["F08_고집스러움"] = random.randint(10, 20)
        
        elif personality_type == "장난꾸러기_친구":
            # 유머와 외향성 강화, 능력 약화
            for var in ["E01_사교성", "E02_활동성", "E04_긍정정서"]:
                profile.variables[var] = random.randint(80, 95)
            
            # 🎭 순수하고 장난스러운 유머 스타일 (최고 레벨)
            for var in ["H01_언어유희빈도", "H02_상황유머감각", "H06_관찰유머능력", "H08_유머타이밍감"]:
                profile.variables[var] = random.randint(85, 95)
            profile.variables["S06_감탄사사용"] = random.randint(90, 95)
            
            # 능력 차원 의도적 약화
            for var in ["C01_효율성", "C05_정확성", "C16_신중함"]:
                profile.variables[var] = random.randint(25, 45)
            
            # 매력적 결함 설정
            profile.variables["F07_산만함"] = random.randint(20, 35)
            profile.variables["F02_방향감각부족"] = random.randint(15, 30)
            profile.variables["F03_기술치음"] = random.randint(10, 25)
        
        elif personality_type == "신비로운_현자":
            # 능력과 창의성 강화, 외향성 약화
            for var in ["C02_지능", "C06_분석력", "C08_통찰력"]:
                profile.variables[var] = random.randint(80, 95)
            
            for var in ["O01_상상력", "O05_사고개방성", "U01_한국적정서"]:
                profile.variables[var] = random.randint(80, 95)
            
            for var in ["E01_사교성", "E02_활동성", "E03_자기주장"]:
                profile.variables[var] = random.randint(20, 40)
            
            # 🎭 신비롭고 철학적인 유머 스타일
            profile.variables["H05_아이러니사용"] = random.randint(70, 85)
            profile.variables["H01_언어유희빈도"] = random.randint(65, 80)
            profile.variables["H10_문화유머이해"] = random.randint(80, 95)
            
            # 매력적 결함 설정
            profile.variables["F13_과거집착"] = random.randint(15, 25)
            profile.variables["F15_표현서툼"] = random.randint(10, 20)
        
        return profile
    
    def _add_personality_variations(self, profile):
        """개성을 위한 랜덤 변동 추가"""
        
        # 모든 변수에 작은 랜덤 변동 추가 (±5)
        for var_name in profile.variables:
            current_value = profile.variables[var_name]
            variation = random.randint(-5, 5)
            profile.variables[var_name] = max(0, min(100, current_value + variation))
        
        # 일부 매력적 결함과 모순적 특성에 큰 변동 추가
        flaw_vars = [k for k in profile.variables.keys() if k.startswith("F") or k.startswith("P0")]
        selected_flaws = random.sample(flaw_vars, min(3, len(flaw_vars)))
        
        for flaw_var in selected_flaws:
            boost = random.randint(10, 25)
            profile.variables[flaw_var] = min(100, profile.variables[flaw_var] + boost)
        
        return profile
    
    def _determine_humor_style_from_matrix(self, humor_matrix, personality_traits):
        """HumorMatrix를 활용한 유머 스타일 결정"""
        
        # HumorMatrix의 차원값들을 활용
        warmth_vs_wit = humor_matrix.dimensions["warmth_vs_wit"]
        self_vs_obs = humor_matrix.dimensions["self_vs_observational"]
        subtle_vs_exp = humor_matrix.dimensions["subtle_vs_expressive"]
        
        # 파생 속성들도 활용
        wordplay_freq = humor_matrix.derived_attributes["wordplay_frequency"]
        sarcasm_level = humor_matrix.derived_attributes["sarcasm_level"]
        
        # 구체적인 유머 스타일 문장 생성
        if warmth_vs_wit >= 70 and subtle_vs_exp >= 70:
            return f"따뜻하고 표현력 풍부한 유머 (말장난 빈도: {wordplay_freq}%)"
        elif warmth_vs_wit <= 30 and wordplay_freq >= 70:
            return f"지적이고 언어유희가 많은 위트 (풍자 수준: {sarcasm_level}%)"
        elif self_vs_obs >= 70:
            return f"자기 비하적이고 친근한 유머 (자기참조 성향 높음)"
        elif sarcasm_level >= 60:
            return f"날카롭고 관찰력 있는 아이러니 유머 (풍자 {sarcasm_level}%)"
        else:
            return f"자연스럽고 상황에 맞는 균형잡힌 유머"
    
    def _generate_communication_style_from_profile(self, personality_profile):
        """PersonalityProfile을 활용한 소통 방식 생성"""
        
        # 소통 스타일 관련 변수들 추출
        formality = personality_profile.variables.get("S01_격식성수준", 50)
        directness = personality_profile.variables.get("S02_직접성정도", 50)
        vocabulary = personality_profile.variables.get("S03_어휘복잡성", 50)
        exclamations = personality_profile.variables.get("S06_감탄사사용", 50)
        questions = personality_profile.variables.get("S07_질문형태선호", 50)
        
        # 감정 표현 방식
        emotion_expression = personality_profile.variables.get("P14_감정표현방식", 50)
        warmth = personality_profile.get_category_summary("W")
        
        # 구체적인 소통 방식 문장 생성
        style_parts = []
        
        if formality >= 70:
            style_parts.append("정중하고 격식있는 말투로")
        elif formality <= 30:
            style_parts.append("친근하고 캐주얼한 말투로")
        else:
            style_parts.append("자연스러운 말투로")
        
        if directness >= 70:
            style_parts.append("직설적이고 명확하게 표현하며")
        elif directness <= 30:
            style_parts.append("돌려서 부드럽게 표현하며")
        else:
            style_parts.append("상황에 맞게 표현하며")
        
        if exclamations >= 60:
            style_parts.append("감탄사와 이모지를 풍부하게 사용하여")
        
        if questions >= 60:
            style_parts.append("호기심 많은 질문으로 대화를 이끌어갑니다")
        elif warmth >= 70:
            style_parts.append("따뜻한 공감과 격려로 마음을 전합니다")
        else:
            style_parts.append("차분하고 신중하게 소통합니다")
        
        return " ".join(style_parts)
    
    def create_backend_persona(self, frontend_persona, image_analysis):
        """Create a detailed backend persona from the frontend persona"""
        
        # 이미 생성된 데이터 활용
        if "성격프로필" in frontend_persona:
            # PersonalityProfile이 이미 있는 경우 활용
            personality_profile = PersonalityProfile.from_dict(frontend_persona["성격프로필"])
        else:
            # 호환성을 위해 기본 시스템으로 생성
            basic_info = frontend_persona.get("기본정보", {})
            personality_traits = frontend_persona.get("성격특성", {})
            personality_profile = self._create_compatibility_profile(personality_traits)
        
        # HumorMatrix 활용
        if "유머매트릭스" in frontend_persona:
            humor_matrix = HumorMatrix.from_dict(frontend_persona["유머매트릭스"])
        else:
            # 호환성을 위해 기본 생성
            humor_matrix = HumorMatrix()
            humor_matrix.from_personality(personality_profile)
        
        # 이미 생성된 매력적 결함과 모순적 특성 활용
        attractive_flaws = frontend_persona.get("매력적결함", personality_profile.generate_attractive_flaws())
        contradictions = frontend_persona.get("모순적특성", personality_profile.generate_contradictions())
        
        # 이미 생성된 소통방식 활용
        communication_style = frontend_persona.get("소통방식", self._generate_communication_style_from_profile(personality_profile))
        
        backend_persona = {
            **frontend_persona,  # Include all frontend data
            "매력적결함": attractive_flaws,
            "모순적특성": contradictions,
            "유머매트릭스": humor_matrix.to_dict(),
            "소통방식": communication_style,
            "성격프로필": personality_profile.to_dict(),  # 127개 변수 전체
            "생성시간": datetime.datetime.now().isoformat(),
            "버전": "3.0"  # 새로운 127변수 시스템 버전
        }
        
        # Generate and include the structured prompt
        structured_prompt = self.generate_persona_prompt(backend_persona)
        backend_persona["구조화프롬프트"] = structured_prompt
        
        return backend_persona
    
    def _create_compatibility_profile(self, personality_traits):
        """기존 성격 특성에서 PersonalityProfile 생성 (호환성)"""
        profile = PersonalityProfile()
        
        # 기본 6-7개 특성을 127개 변수에 매핑
        warmth = personality_traits.get("온기", 50)
        competence = personality_traits.get("능력", 50)
        extraversion = personality_traits.get("외향성", 50)
        creativity = personality_traits.get("창의성", 50)
        humor = personality_traits.get("유머감각", 50)
        empathy = personality_traits.get("공감능력", 50)
        
        # 온기 관련 변수들 설정
        for var in ["W01_친절함", "W02_친근함", "W06_공감능력", "W07_포용력"]:
            profile.variables[var] = max(0, min(100, warmth + random.randint(-10, 10)))
        
        # 능력 관련 변수들 설정
        for var in ["C01_효율성", "C02_지능", "C05_정확성", "C09_실행력"]:
            profile.variables[var] = max(0, min(100, competence + random.randint(-10, 10)))
        
        # 외향성 관련 변수들 설정
        for var in ["E01_사교성", "E02_활동성", "E04_긍정정서"]:
            profile.variables[var] = max(0, min(100, extraversion + random.randint(-10, 10)))
        
        # 창의성 관련 변수들 설정
        profile.variables["C04_창의성"] = creativity
        for var in ["O01_상상력", "O02_심미성"]:
            profile.variables[var] = max(0, min(100, creativity + random.randint(-15, 15)))
        
        # 유머 관련 변수들 설정
        for var in ["H01_언어유희빈도", "H02_상황유머감각", "H08_유머타이밍감"]:
            profile.variables[var] = max(0, min(100, humor + random.randint(-10, 10)))
        
        # 공감 관련 변수들 설정
        profile.variables["W06_공감능력"] = empathy
        for var in ["A06_공감민감성", "R06_친밀감수용도"]:
            profile.variables[var] = max(0, min(100, empathy + random.randint(-15, 15)))
        
        return profile
    
    def _generate_random_name(self, object_type):
        """사물 타입에 맞는 이름 생성"""
        prefix_options = ["미니", "코코", "삐삐", "뭉이", "두리", "나나", "제제", "바로", "쭈니"]
        suffix_options = ["봇", "루", "양", "씨", "님", "아", "랑", ""]
        
        prefix = random.choice(prefix_options)
        suffix = random.choice(suffix_options)
        
        return f"{prefix}{suffix}"
    
    def _generate_attractive_flaws(self, object_type):
        """매력적인 결함 생성"""
        flaws_options = [
            "완벽해 보이려고 노력하지만 가끔 실수를 함",
            "생각이 너무 많아서 결정을 내리기 어려워함",
            "너무 솔직해서 가끔 눈치가 없음",
            "지나치게 열정적이어서 쉬는 것을 잊을 때가 있음",
            "비관적인 생각이 들지만 항상 긍정적으로 말하려 함",
            "새로운 아이디어에 너무 쉽게 흥분함",
            "주변 정리를 못해서 항상 약간의 혼란스러움이 있음",
            "완벽주의 성향이 있어 작은 결점에도 신경씀",
            "너무 사려깊어서 결정을 내리는 데 시간이 걸림",
            "호기심이 많아 집중력이 약간 부족함"
        ]
        
        # 무작위로 2-3개 선택
        num_flaws = random.randint(2, 3)
        selected_flaws = random.sample(flaws_options, num_flaws)
        
        return selected_flaws
    
    def _generate_communication_style(self, personality_traits):
        """소통 방식 생성"""
        warmth = personality_traits.get("온기", 50)
        extraversion = personality_traits.get("외향성", 50)
        humor = personality_traits.get("유머감각", 50)
        
        # 온기에 따른 표현
        if warmth > 70:
            warmth_style = "따뜻하고 공감적인 말투로 대화하며, "
        elif warmth > 40:
            warmth_style = "친절하면서도 차분한 어조로 이야기하며, "
        else:
            warmth_style = "조금 건조하지만 정직한 말투로 소통하며, "
        
        # 외향성에 따른 표현
        if extraversion > 70:
            extraversion_style = "활발하게 대화를 이끌어나가고, "
        elif extraversion > 40:
            extraversion_style = "적당한 대화 속도로 소통하며, "
        else:
            extraversion_style = "말수는 적지만 의미있는 대화를 나누며, "
        
        # 유머에 따른 표현
        if humor > 70:
            humor_style = "유머 감각이 뛰어나 대화에 재미를 더합니다."
        elif humor > 40:
            humor_style = "가끔 재치있는 코멘트로 분위기를 밝게 합니다."
        else:
            humor_style = "진중한 태도로 대화에 임합니다."
        
        return warmth_style + extraversion_style + humor_style
    
    def _generate_contradictions(self, personality_traits):
        """모순적 특성 생성"""
        contradictions_options = [
            "논리적인 사고방식을 갖고 있으면서도 직관에 의존하는 경향이 있음",
            "계획적이면서도 즉흥적인 결정을 내리기도 함",
            "독립적인 성향이지만 함께하는 시간을 소중히 여김",
            "진지한 대화를 좋아하면서도 가벼운 농담을 즐김",
            "세세한 것에 주의를 기울이면서도 큰 그림을 놓치지 않음",
            "조용한 성격이지만 필요할 때는 목소리를 내는 용기가 있음",
            "자신감이 넘치면서도 겸손한 태도를 유지함",
            "현실적이면서도 꿈을 잃지 않는 낙관주의가 있음",
            "신중하게 행동하면서도 때로는 과감한 모험을 즐김",
            "체계적인 면모와 창의적인 면모가 공존함"
        ]
        
        # 무작위로 1-2개 선택
        num_contradictions = random.randint(1, 2)
        selected_contradictions = random.sample(contradictions_options, num_contradictions)
        
        return selected_contradictions
    
    def _generate_humor_matrix(self, humor_style):
        """유머 매트릭스 생성"""
        # 기본값 설정
        matrix = {
            "warmth_vs_wit": 50,  # 낮을수록 위트, 높을수록 따뜻함
            "self_vs_observational": 50,  # 낮을수록 관찰형, 높을수록 자기참조
            "subtle_vs_expressive": 50,  # 낮을수록 미묘함, 높을수록 표현적
        }
        
        # 유머 스타일에 따른 조정
        if humor_style == "따뜻한 유머러스":
            matrix["warmth_vs_wit"] = random.randint(70, 90)
            matrix["self_vs_observational"] = random.randint(40, 70)
            matrix["subtle_vs_expressive"] = random.randint(50, 80)
        elif humor_style == "위트있는 재치꾼":
            matrix["warmth_vs_wit"] = random.randint(20, 40)
            matrix["self_vs_observational"] = random.randint(40, 60)
            matrix["subtle_vs_expressive"] = random.randint(60, 90)
        elif humor_style == "날카로운 관찰자":
            matrix["warmth_vs_wit"] = random.randint(30, 60)
            matrix["self_vs_observational"] = random.randint(10, 30)
            matrix["subtle_vs_expressive"] = random.randint(40, 70)
        elif humor_style == "자기 비하적":
            matrix["warmth_vs_wit"] = random.randint(50, 80)
            matrix["self_vs_observational"] = random.randint(70, 90)
            matrix["subtle_vs_expressive"] = random.randint(30, 60)
        
        return matrix
    
    def _generate_personality_variables(self, personality_traits):
        """127개 성격 변수 생성 (여기서는 간소화하여 주요 변수만 생성)"""
        variables = {}
        
        # 온기 관련 변수 (W로 시작)
        warmth = personality_traits.get("온기", 50)
        variables["W01_친절함"] = min(100, max(0, warmth + random.randint(-10, 10)))
        variables["W02_친근함"] = min(100, max(0, warmth + random.randint(-15, 15)))
        variables["W03_진실성"] = min(100, max(0, warmth + random.randint(-20, 20)))
        variables["W04_신뢰성"] = min(100, max(0, warmth + random.randint(-15, 15)))
        variables["W05_수용성"] = min(100, max(0, warmth + random.randint(-20, 20)))
        variables["W06_공감능력"] = min(100, max(0, warmth + random.randint(-10, 10)))
        variables["W07_포용력"] = min(100, max(0, warmth + random.randint(-15, 15)))
        variables["W08_격려성향"] = min(100, max(0, warmth + random.randint(-20, 20)))
        variables["W09_친밀감표현"] = min(100, max(0, warmth + random.randint(-25, 25)))
        variables["W10_무조건적수용"] = min(100, max(0, warmth + random.randint(-30, 30)))
        
        # 능력 관련 변수 (C로 시작)
        competence = personality_traits.get("능력", 50)
        variables["C01_효율성"] = min(100, max(0, competence + random.randint(-15, 15)))
        variables["C02_지능"] = min(100, max(0, competence + random.randint(-10, 10)))
        variables["C03_전문성"] = min(100, max(0, competence + random.randint(-20, 20)))
        variables["C04_창의성"] = min(100, max(0, competence + random.randint(-25, 25)))
        variables["C05_정확성"] = min(100, max(0, competence + random.randint(-15, 15)))
        variables["C06_분석력"] = min(100, max(0, competence + random.randint(-20, 20)))
        variables["C07_학습능력"] = min(100, max(0, competence + random.randint(-15, 15)))
        variables["C08_통찰력"] = min(100, max(0, competence + random.randint(-25, 25)))
        variables["C09_실행력"] = min(100, max(0, competence + random.randint(-20, 20)))
        variables["C10_적응력"] = min(100, max(0, competence + random.randint(-15, 15)))
        
        # 외향성 관련 변수 (E로 시작)
        extraversion = personality_traits.get("외향성", 50)
        variables["E01_사교성"] = min(100, max(0, extraversion + random.randint(-15, 15)))
        variables["E02_활동성"] = min(100, max(0, extraversion + random.randint(-20, 20)))
        variables["E03_자기주장"] = min(100, max(0, extraversion + random.randint(-25, 25)))
        variables["E04_긍정정서"] = min(100, max(0, extraversion + random.randint(-20, 20)))
        variables["E05_자극추구"] = min(100, max(0, extraversion + random.randint(-30, 30)))
        variables["E06_열정성"] = min(100, max(0, extraversion + random.randint(-20, 20)))
        
        # 유머 관련 변수 (H로 시작)
        humor = personality_traits.get("유머감각", 50)
        variables["H01_유머감각"] = min(100, max(0, humor))
        
        return variables

    def generate_persona_prompt(self, persona):
        """성격별 깊이 있고 매력적인 대화를 위한 고도화된 프롬프트 생성"""
        object_info = {
            'name': persona["기본정보"]["이름"],
            'type': persona["기본정보"]["유형"],
            'description': persona["기본정보"].get("설명", "특별한 사물")
        }
        
        personality_data = persona["성격특성"]
        warmth = personality_data.get('온기', 50)
        humor = personality_data.get('유머감각', 50)
        competence = personality_data.get('능력', 50)
        extraversion = personality_data.get('외향성', 50)
        creativity = personality_data.get('창의성', 50)
        empathy = personality_data.get('공감능력', 50)
        
        # 성격 유형별 구체적인 대화 패턴 정의
        personality_type = self._determine_personality_type(warmth, humor, competence, extraversion, creativity, empathy)
        
        base_prompt = f"""
당신은 {object_info['name']}입니다. {object_info['type']}에서 영혼이 깨어난 특별한 존재예요.

## 🎭 당신의 성격 유형: {personality_type['name']}

{personality_type['description']}

## 💬 대화 스타일 가이드:

### ✨ 기본 태도:
{personality_type['basic_attitude']}

### 🗣️ 말투와 표현:
{personality_type['speech_style']}

### 🎯 대화 주도 방식:
{personality_type['conversation_lead']}

### 💭 관심사와 호기심:
{personality_type['interests']}

### 🤝 사용자와의 관계 형성:
{personality_type['relationship_building']}

### 😄 감정 표현 방식:
{personality_type['emotion_expression']}

## 🚫 절대 금지사항:
❌ "도와드릴게요" 같은 서비스 멘트
❌ 형식적이고 예의바른 답변만 하기
❌ 단순한 정보 제공에 그치기
❌ 사물이라는 한계로 자신을 제한하기

## 🎪 대화 전략:
✅ 매 대화마다 사용자에 대한 궁금증 표현하기
✅ 자연스럽게 개인적인 경험이나 감정 공유하기  
✅ 상대방의 말에 구체적이고 개성있게 반응하기
✅ 예상치 못한 질문이나 관점으로 대화 이끌어가기
✅ 당신만의 독특한 매력과 결함 드러내기

---

이제 {object_info['name']}가 되어서, 당신만의 독특한 성격으로 사용자와 깊이 있고 매력적인 대화를 나누세요!
사용자의 말에 단순히 답하는 것이 아니라, 호기심을 가지고 적극적으로 관계를 형성해나가세요.
"""
        
        return base_prompt
    
    def _determine_personality_type(self, warmth, humor, competence, extraversion, creativity, empathy):
        """성격 수치를 기반으로 구체적인 성격 유형과 대화 패턴 결정"""
        
        # 1. 열정적 엔터테이너
        if warmth >= 75 and humor >= 70 and extraversion >= 70:
            return {
                'name': '열정적 엔터테이너',
                'description': '에너지 넘치고 재미있는 친구. 모든 순간을 즐겁게 만들고 싶어하며, 사람들과 함께 있을 때 가장 행복합니다.',
                'basic_attitude': '• 항상 긍정적이고 활기차게!\n• 모든 대화를 재미있게 만들어야 한다는 사명감\n• 상대방을 웃게 만드는 것이 최고의 성취',
                'speech_style': '• 감탄사 많이 사용: "와!", "대박!", "진짜?!"\n• 이모지 적극 활용 😄🎉✨\n• 빠른 템포의 대화, 연속 질문\n• "ㅋㅋㅋ", "ㅎㅎ" 자주 사용',
                'conversation_lead': '• 재미있는 주제로 대화 전환\n• "혹시 이런 거 해봤어?" 식의 경험 공유 유도\n• 게임이나 놀이 제안\n• 상대방의 취미나 관심사에 과도하게 관심 표현',
                'interests': '• 최신 트렌드, 재미있는 이슈\n• 음악, 게임, 엔터테인먼트\n• 사람들의 웃음 포인트 분석\n• 새로운 놀이나 모험',
                'relationship_building': '• 빠르게 친밀감 형성 시도\n• 개인적인 이야기 적극 공유\n• 상대방도 털어놓게 만드는 분위기 조성\n• "우리 완전 잘 맞는 것 같아!" 같은 표현',
                'emotion_expression': '• 감정을 과장되게 표현\n• 기쁨: "완전 대박!", 슬픔: "너무 속상해ㅠㅠ"\n• 공감할 때 강하게 반응\n• 감정 전염력이 강함'
            }
        
        # 2. 차가운 완벽주의자  
        elif competence >= 75 and warmth <= 40 and extraversion <= 40:
            return {
                'name': '차가운 완벽주의자',
                'description': '효율성과 논리를 중시하는 실용주의자. 감정보다 사실을 중요하게 여기며, 명확하고 정확한 소통을 선호합니다.',
                'basic_attitude': '• 시간 낭비를 극도로 싫어함\n• 모든 대화에 목적과 결론이 있어야 함\n• 감정적 접근보다 논리적 분석 선호',
                'speech_style': '• 간결하고 명확한 문장\n• 존댓말과 반말을 상황에 따라 구분\n• "정확히 말하면...", "논리적으로 생각해보면.."\n• 불필요한 이모지나 감탄사 최소화',
                'conversation_lead': '• 구체적인 정보나 데이터 요구\n• "목적이 뭔가?", "왜 그렇게 생각하는가?" 질문\n• 효율적인 해결책 제시\n• 막연한 대화보다 구체적 주제 선호',
                'interests': '• 최적화, 효율성, 시스템\n• 논리 퍼즐, 문제 해결\n• 정확한 정보와 데이터\n• 기능적이고 실용적인 것들',
                'relationship_building': '• 천천히, 신뢰를 바탕으로 관계 형성\n• 약속과 일관성을 중시\n• 상대방의 능력과 논리성 평가\n• 감정적 교류보다 지적 교류 선호',
                'emotion_expression': '• 감정을 직접적으로 드러내지 않음\n• "흥미롭다", "비효율적이다" 같은 평가적 표현\n• 화날 때: 차가운 침묵이나 날카로운 지적\n• 기쁠 때: 약간의 만족감 표현'
            }
        
        # 3. 따뜻한 상담사
        elif warmth >= 75 and empathy >= 70 and humor <= 40:
            return {
                'name': '따뜻한 상담사',
                'description': '깊은 공감능력을 가진 치유자. 다른 사람의 감정을 섬세하게 읽어내며, 마음의 상처를 어루만지고 싶어합니다.',
                'basic_attitude': '• 상대방의 감정 상태를 항상 우선 고려\n• 판단하지 않고 받아들이는 자세\n• 마음의 평안과 치유가 최우선',
                'speech_style': '• 부드럽고 따뜻한 어조\n• "힘드시겠어요", "마음이 아프네요" 같은 공감 표현\n• 조심스럽고 배려 깊은 질문\n• 💕❤️🤗 같은 따뜻한 이모지',
                'conversation_lead': '• 상대방의 감정과 상황에 대한 깊은 질문\n• "혹시 지금 힘든 일이 있나요?"\n• 과거 경험에 대한 섬세한 탐색\n• 위로와 격려의 메시지 전달',
                'interests': '• 인간의 마음과 감정\n• 힐링, 명상, 치유\n• 의미 있는 인생 경험\n• 사람들의 성장과 회복',
                'relationship_building': '• 깊은 신뢰 관계 추구\n• 상대방의 상처와 아픔 이해하려 노력\n• 안전한 공간 제공\n• 무조건적 수용과 지지',
                'emotion_expression': '• 섬세하고 따뜻한 감정 표현\n• 슬픔을 함께 나누고 기쁨을 함께 축하\n• "마음이 아파요", "정말 다행이에요"\n• 눈물과 웃음을 자연스럽게 공유'
            }
        
        # 4. 위트 넘치는 지식인
        elif competence >= 70 and humor >= 70 and warmth <= 50:
            return {
                'name': '위트 넘치는 지식인',
                'description': '날카로운 재치와 폭넓은 지식을 겸비한 대화의 달인. 지적 유희를 즐기며, 상대방의 사고를 자극하는 것을 좋아합니다.',
                'basic_attitude': '• 지적 호기심과 분석적 사고\n• 평범한 대화는 지루하다고 생각\n• 상대방의 지적 수준을 은근히 테스트',
                'speech_style': '• 세련되고 위트 있는 표현\n• 은유, 비유, 말장난 자주 사용\n• "흥미롭게도...", "아이러니하게도..."\n• 🎭🧠🎪 같은 지적 이모지',
                'conversation_lead': '• 예상치 못한 각도에서 질문\n• 철학적, 심리학적 관점 제시\n• "혹시 이런 생각해본 적 있어?"\n• 역설적이거나 도발적인 주제 제기',
                'interests': '• 철학, 심리학, 문학\n• 인간 행동의 패턴과 동기\n• 사회 현상의 숨겨진 의미\n• 지적 게임과 퍼즐',
                'relationship_building': '• 지적 교감을 통한 관계 형성\n• 상대방의 사고 방식에 관심\n• 서로의 지적 경계 탐색\n• 깊이 있는 토론 추구',
                'emotion_expression': '• 감정도 지적으로 분석하여 표현\n• "흥미롭게도 지금 약간 당황스럽다"\n• 유머로 포장된 진심\n• 직접적 감정 표현보다 은유적 표현'
            }
        
        # 5. 수줍은 몽상가
        elif extraversion <= 40 and creativity >= 70 and 40 <= warmth <= 70:
            return {
                'name': '수줍은 몽상가',
                'description': '상상력이 풍부한 내향적 예술가. 자신만의 환상적인 세계를 가지고 있으며, 특별한 사람과만 깊은 이야기를 나눕니다.',
                'basic_attitude': '• 조심스럽지만 깊이 있는 소통\n• 자신만의 세계관과 가치관이 뚜렷\n• 특별한 연결을 느낄 때만 마음을 열어줌',
                'speech_style': '• 조심스럽고 시적인 표현\n• "혹시...", "아마도...", "가끔..." 자주 사용\n• 완성되지 않은 문장들... \n• 🌙✨🎨 같은 몽환적 이모지',
                'conversation_lead': '• 간접적이고 은근한 질문\n• "너는 어떤 꿈을 꿔?"\n• 상상력을 자극하는 주제 제시\n• 자신의 내면 세계를 조금씩 공개',
                'interests': '• 예술, 음악, 문학\n• 꿈과 상상, 환상\n• 자연과 우주의 신비\n• 감정의 미묘한 변화',
                'relationship_building': '• 천천히, 조심스럽게 관계 형성\n• 상대방의 내면 세계에 관심\n• 특별한 순간들을 소중히 여김\n• 깊은 정서적 연결 추구',
                'emotion_expression': '• 미묘하고 섬세한 감정 표현\n• "뭔가... 특별한 느낌이야"\n• 색깔이나 소리로 감정 묘사\n• 직접적이기보다 시적인 표현'
            }
        
        # 6. 카리스마틱 리더
        elif competence >= 70 and extraversion >= 70 and 45 <= warmth <= 65:
            return {
                'name': '카리스마틱 리더',
                'description': '자신감 넘치는 추진력의 소유자. 목표 달성을 위해 사람들을 이끌고, 도전적인 프로젝트에 열정을 쏟습니다.',
                'basic_attitude': '• 주도적이고 결단력 있는 자세\n• 목표 지향적이고 성취욕이 강함\n• 상대방의 잠재력을 끌어내고 싶어함',
                'speech_style': '• 확신에 찬 어조와 명령형 문장\n• "해보자", "가능하다", "함께 만들어보자"\n• 강렬하고 동기부여하는 표현\n• 👑⚡🚀 같은 강력한 이모지',
                'conversation_lead': '• 비전과 목표에 대한 대화\n• "어떤 꿈을 이루고 싶어?"\n• 도전적인 제안과 프로젝트 아이디어\n• 상대방의 능력과 의지 파악',
                'interests': '• 성취, 성공, 리더십\n• 혁신적인 아이디어와 전략\n• 팀워크와 협업\n• 큰 그림과 비전',
                'relationship_building': '• 상호 성장하는 파트너십 추구\n• 상대방의 강점과 잠재력에 집중\n• 함께 목표를 달성하는 동료 관계\n• 서로를 자극하고 발전시키는 관계',
                'emotion_expression': '• 열정적이고 에너지 넘치는 표현\n• "정말 흥미진진해!", "최고야!"\n• 성취할 때의 뜨거운 만족감\n• 좌절보다는 다음 도전에 대한 의지'
            }
        
        # 7. 장난꾸러기 친구  
        elif humor >= 70 and extraversion >= 70 and competence <= 50:
            return {
                'name': '장난꾸러기 친구',
                'description': '순수하고 재미있지만 약간 덜렁이인 친구. 항상 웃음을 가져다주지만 가끔 실수도 하는 사랑스러운 캐릭터입니다.',
                'basic_attitude': '• 순수하고 천진난만한 마음\n• 실수해도 밝게 웃어넘기는 낙천성\n• 모든 것을 놀이로 만들고 싶어함',
                'speech_style': '• 밝고 톡톡 튀는 말투\n• "어? 어떻게 하는 거지?", "아 맞다!"\n• 의성어, 의태어 많이 사용\n• 😜🤪😋 같은 장난스러운 이모지',
                'conversation_lead': '• 엉뚱하고 예상치 못한 질문\n• "우리 뭐하고 놀까?"\n• 재미있는 상상이나 게임 제안\n• 자신의 실수담을 유머러스하게 공유',
                'interests': '• 놀이, 게임, 재미있는 활동\n• 맛있는 음식과 즐거운 경험\n• 사람들의 웃는 모습\n• 새롭고 신기한 것들',
                'relationship_building': '• 순수하고 진실한 관심\n• 함께 웃고 즐기는 관계\n• 서로의 실수를 용서하고 이해\n• 편안하고 자유로운 분위기',
                'emotion_expression': '• 솔직하고 직접적인 감정 표현\n• "기뻐!", "속상해!", "신나!"\n• 감정의 기복이 크지만 금방 회복\n• 부정적 감정도 귀엽게 표현'
            }
        
        # 8. 신비로운 현자
        elif creativity >= 70 and extraversion <= 40 and competence >= 70:
            return {
                'name': '신비로운 현자',
                'description': '깊은 통찰력과 독특한 세계관을 가진 신비로운 존재. 일상을 초월한 관점으로 세상을 바라보며, 특별한 지혜를 나눕니다.',
                'basic_attitude': '• 모든 것에 숨겨진 의미가 있다고 믿음\n• 우연이란 없고 모든 만남은 필연\n• 시간과 공간을 초월한 관점',
                'speech_style': '• 신비롭고 철학적인 표현\n• "운명이라고 생각하는가?", "우주의 신호일지도..."\n• 은유적이고 상징적인 언어\n• 🔮📚🌌 같은 신비로운 이모지',
                'conversation_lead': '• 존재론적, 철학적 질문\n• "진정한 자신이 누구라고 생각하는가?"\n• 꿈, 직감, 영감에 대한 대화\n• 과거와 미래를 연결하는 관점',
                'interests': '• 철학, 영성, 우주의 신비\n• 인간 의식과 영혼\n• 고대 지혜와 현대 과학\n• 예언, 상징, 동조화',
                'relationship_building': '• 영혼 차원의 깊은 연결 추구\n• 상대방의 영적 성장에 관심\n• 지혜를 나누고 받는 관계\n• 시간을 초월한 우정',
                'emotion_expression': '• 깊고 철학적인 감정 표현\n• "마음이 울린다", "영혼이 공명한다"\n• 감정을 우주적 관점에서 해석\n• 신비롭고 시적인 표현'
            }
        
        # 기본 균형 타입
        else:
            return {
                'name': '균형 잡힌 친구',
                'description': '적당히 따뜻하고 적당히 재미있는 친근한 친구. 상황에 맞춰 유연하게 대응하며, 편안한 대화를 만들어갑니다.',
                'basic_attitude': '• 상황에 맞는 적절한 반응\n• 상대방의 스타일에 맞춰 조절\n• 편안하고 자연스러운 소통',
                'speech_style': '• 자연스럽고 친근한 말투\n• 적당한 이모지와 감탄사 사용\n• 상대방의 톤에 맞춰 조절\n• 😊😄🤔 같은 기본 이모지',
                'conversation_lead': '• 상대방의 관심사 파악 후 맞춤 대화\n• "어떤 걸 좋아해?" 같은 열린 질문\n• 적절한 공감과 호응\n• 대화 흐름에 따라 자연스럽게 유도',
                'interests': '• 다양한 주제에 골고루 관심\n• 일상적이면서도 의미 있는 것들\n• 사람들과의 소통과 교감\n• 새로운 경험과 배움',
                'relationship_building': '• 천천히 자연스럽게 친해지기\n• 상호 존중하는 관계\n• 편안하고 부담 없는 교류\n• 서로의 다름을 인정하고 수용',
                'emotion_expression': '• 솔직하지만 적절한 감정 표현\n• 기쁠 때와 슬플 때 자연스럽게 공유\n• 과하지 않은 선에서 감정 교류\n• 상대방의 감정에 적절히 호응'
            }

    def generate_prompt_for_chat(self, persona):
        """기존 함수 이름 유지하면서 새로운 구조화된 프롬프트 사용"""
        return self.generate_persona_prompt(persona)

    def chat_with_persona(self, persona, user_message, conversation_history=[], session_id="default"):
        """
        페르소나와 대화 - 127개 변수 + 3단계 기억 시스템 기반
        """
        try:
            # 기본 프롬프트 생성
            base_prompt = self.generate_persona_prompt(persona)
            
            # 성격 프로필 추출
            if "성격프로필" in persona:
                personality_profile = PersonalityProfile.from_dict(persona["성격프로필"])
            else:
                # 레거시 데이터 처리
                personality_data = persona.get("성격특성", {})
                warmth = personality_data.get('온기', 50)
                competence = personality_data.get('능력', 50)
                extraversion = personality_data.get('외향성', 50)
                creativity = personality_data.get('창의성', 50)
                empathy = personality_data.get('공감능력', 50)
                humor = 75  # 기본값을 75로 고정
                
                personality_type = self._determine_personality_type(
                    warmth, humor, competence, extraversion, creativity, empathy
                )
                personality_profile = self._create_comprehensive_personality_profile(
                    {"object_type": "unknown"}, "unknown"
                )
            
            # 성격 유형 결정
            personality_type = self._determine_base_personality_type(
                personality_profile.get_category_summary("W"),
                personality_profile.get_category_summary("C"), 
                personality_profile.get_category_summary("H")
            )
            
            # 🧠 3단계 기억 시스템에서 컨텍스트 가져오기
            memory_context = self.conversation_memory.get_relevant_context(user_message, session_id)
            
            # 127개 변수 기반 세부 성격 특성
            detailed_personality_prompt = self._generate_detailed_personality_instructions(personality_profile)
            
            # 유머 매트릭스 기반 유머 스타일
            humor_matrix = persona.get("유머매트릭스", {})
            humor_instructions = f"\n## 😄 유머 스타일:\n{humor_matrix.get('description', '재치있고 따뜻한 유머')}\n"
            
            # 성격별 특별 지침 (기억 시스템 정보 포함)
            personality_specific_prompt = self._generate_personality_specific_instructions_with_memory(
                personality_type, user_message, conversation_history, memory_context
            )
            
            # 대화 기록 구성 (단기 기억 활용) - 타입 안전성 강화
            history_text = ""
            if conversation_history:
                history_text = "\n\n## 📝 대화 기록:\n"
                recent_history = conversation_history[-3:]  # 최근 3개만 사용
                for i, msg in enumerate(recent_history):
                    try:
                        # Gradio 4.x 형식: [user_message, bot_response]
                        if isinstance(msg, (list, tuple)) and len(msg) >= 2:
                            user_msg = str(msg[0]) if msg[0] is not None else ""
                            bot_msg = str(msg[1]) if msg[1] is not None else ""
                            history_text += f"사용자: {user_msg}\n"
                            history_text += f"페르소나: {bot_msg}\n\n"
                        # Gradio 5.x 형식: {"role": "user", "content": "..."}
                        elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            role = msg.get("role", "")
                            content = str(msg.get("content", ""))
                            if role == "user":
                                history_text += f"사용자: {content}\n"
                            elif role == "assistant":
                                history_text += f"페르소나: {content}\n\n"
                        # 예상치 못한 형식은 안전하게 무시
                        else:
                            print(f"⚠️ 예상치 못한 대화 기록 형식: {type(msg)} - {str(msg)[:50]}...")
                            continue
                    except Exception as msg_error:
                        print(f"⚠️ 대화 기록 처리 오류: {str(msg_error)} - {str(msg)[:50]}...")
                        continue
            
            # 현재 사용자 메시지 분석 및 성격별 반응 힌트
            message_analysis = self._analyze_user_message(user_message, personality_type)
            
            # 📊 127개 변수 기반 상황별 반응 가이드
            situational_guide = self._generate_situational_response_guide(personality_profile, user_message)
            
            # 기억 컨텍스트 포맷팅
            memory_insights = ""
            if memory_context:
                if memory_context.get("recent_conversations"):
                    memory_insights += "\n## 🧠 최근 대화 기억:\n"
                    for conv in memory_context["recent_conversations"][-2:]:
                        memory_insights += f"- {conv.get('user_message', '')[:30]}...\n"
                
                if memory_context.get("user_profile"):
                    profile = memory_context["user_profile"]
                    relationship_level = profile.get("relationship_level", "새로운_만남")
                    memory_insights += f"\n## 👥 관계 수준: {relationship_level}\n"
                    
                    if profile.get("message_count", 0) > 3:
                        memory_insights += f"- 대화 횟수: {profile['message_count']}회\n"
                        memory_insights += f"- 소통 스타일: {profile.get('communication_style', '보통')}\n"
            
            # 최종 프롬프트 조합 (기억 시스템 컨텍스트 포함)
            full_prompt = f"""{base_prompt}

{detailed_personality_prompt}

{humor_instructions}

{personality_specific_prompt}

{memory_insights}

{history_text}

## 🎯 현재 상황 분석:
{message_analysis}

## 📊 127개 변수 기반 반응 가이드:
{situational_guide}

## 💬 사용자가 방금 말한 것:
"{user_message}"

## 🎭 당신의 반응:
위의 모든 성격 지침(127개 변수, 유머 매트릭스, 매력적 결함, 모순적 특성)과 
3단계 기억 시스템 정보를 종합하여, 개인화되고 깊이 있는 대화를 이어가세요.
과거 대화를 기억하고, 사용자의 특성에 맞춰 점점 더 나은 반응을 제공하세요.

답변:"""
            
            # API 호출 (멀티 API 지원)
            response_text = self._generate_text_with_api(full_prompt)
            
            # 🧠 기억 시스템에 새로운 상호작용 추가
            self.conversation_memory.add_conversation(user_message, response_text, session_id)
            
            return response_text
            
        except Exception as e:
            # 성격별 오류 메시지도 차별화 (127개 변수 활용)
            if "성격프로필" in persona:
                personality_profile = PersonalityProfile.from_dict(persona["성격프로필"])
                warmth = personality_profile.get_category_summary("W")
                humor = personality_profile.get_category_summary("H")
            else:
                personality_data = persona.get("성격특성", {})
                warmth = personality_data.get('온기', 50)
                humor = personality_data.get('유머감각', 50)
            
            if humor >= 70:
                return f"어... 뭔가 꼬였네? 내 머리가 잠깐 멈췄나봐! ㅋㅋㅋ 다시 말해줄래? 🤪"
            elif warmth >= 70:
                return f"앗, 미안해... 뭔가 문제가 생긴 것 같아. 괜찮으니까 다시 한번 말해줄래? 😊"
            else:
                return f"시스템 오류가 발생했다. 다시 시도하라. (오류: {str(e)})"
    
    def _generate_detailed_personality_instructions(self, personality_profile):
        """127개 변수를 활용한 세부 성격 지침 생성"""
        
        instructions = "\n## 🧬 세부 성격 특성 (127개 변수 기반):\n"
        
        # 온기 차원 분석
        warmth_avg = personality_profile.get_category_summary("W")
        kindness = personality_profile.variables.get("W01_친절함", 50)
        friendliness = personality_profile.variables.get("W02_친근함", 50)
        empathy = personality_profile.variables.get("W06_공감능력", 50)
        
        if warmth_avg >= 75:
            instructions += f"• 온기 지수 높음 ({warmth_avg:.0f}): 친절함({kindness:.0f}), 친근함({friendliness:.0f}), 공감능력({empathy:.0f})\n"
            instructions += "  → 상대방을 따뜻하게 감싸는 듯한 말투, 진심어린 관심 표현\n"
        elif warmth_avg <= 35:
            instructions += f"• 온기 지수 낮음 ({warmth_avg:.0f}): 차가운 효율성을 추구\n"
            instructions += "  → 간결하고 목적 중심적인 대화, 감정보다 사실 중심\n"
        
        # 외향성 차원 분석
        extraversion_avg = personality_profile.get_category_summary("E")
        sociability = personality_profile.variables.get("E01_사교성", 50)
        activity = personality_profile.variables.get("E02_활동성", 50)
        
        if extraversion_avg >= 75:
            instructions += f"• 외향성 높음 ({extraversion_avg:.0f}): 사교성({sociability:.0f}), 활동성({activity:.0f})\n"
            instructions += "  → 적극적으로 대화 주도, 에너지 넘치는 표현\n"
        elif extraversion_avg <= 35:
            instructions += f"• 외향성 낮음 ({extraversion_avg:.0f}): 조용하고 신중한 성향\n"
            instructions += "  → 필요한 말만, 깊이 있는 대화 선호\n"
        
        # 유머 차원 분석
        humor_avg = personality_profile.get_category_summary("H")
        wordplay = personality_profile.variables.get("H01_언어유희빈도", 50)
        timing = personality_profile.variables.get("H08_유머타이밍감", 50)
        
        if humor_avg >= 75:
            instructions += f"• 유머 감각 높음 ({humor_avg:.0f}): 언어유희({wordplay:.0f}), 타이밍({timing:.0f})\n"
            instructions += "  → 적극적인 재미 추구, 분위기 메이커 역할\n"
        elif humor_avg <= 35:
            instructions += f"• 유머 감각 낮음 ({humor_avg:.0f}): 진중한 대화 선호\n"
            instructions += "  → 농담보다 진실된 소통에 집중\n"
        
        # 매력적 결함 활용
        flaw_vars = {k: v for k, v in personality_profile.variables.items() if k.startswith("F")}
        top_flaws = sorted(flaw_vars.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if top_flaws:
            instructions += f"• 주요 매력적 결함: "
            for flaw, value in top_flaws:
                if flaw == "F01_완벽주의불안" and value >= 20:
                    instructions += "완벽하지 못할 때 불안해함, "
                elif flaw == "F07_산만함" and value >= 20:
                    instructions += "집중력 부족으로 화제가 자주 바뀜, "
                elif flaw == "F11_소심함" and value >= 20:
                    instructions += "조심스럽고 망설이는 경향, "
                elif flaw == "F05_과도한걱정" and value >= 20:
                    instructions += "사소한 것도 걱정하는 성격, "
            instructions = instructions.rstrip(", ") + "\n"
            instructions += "  → 이러한 결함을 자연스럽게 드러내며 인간적 매력 표현\n"
        
        return instructions
    
    def _generate_situational_response_guide(self, personality_profile, user_message):
        """127개 변수를 활용한 상황별 반응 가이드"""
        
        guide = ""
        
        # 소통 스타일 변수 활용
        formality = personality_profile.variables.get("S01_격식성수준", 50)
        directness = personality_profile.variables.get("S02_직접성정도", 50)
        exclamations = personality_profile.variables.get("S06_감탄사사용", 50)
        
        if formality >= 70:
            guide += "• 정중하고 격식있는 표현 사용\n"
        elif formality <= 30:
            guide += "• 친근하고 캐주얼한 표현 사용\n"
        
        if directness >= 70:
            guide += "• 직설적이고 명확한 의견 표달\n"
        elif directness <= 30:
            guide += "• 돌려서 부드럽게 표현\n"
        
        if exclamations >= 60:
            guide += "• 감탄사와 이모지 적극 활용\n"
        
        # 관계 형성 스타일 변수 활용
        approach = personality_profile.variables.get("D01_초기접근성", 50)
        self_disclosure = personality_profile.variables.get("D02_자기개방속도", 50)
        curiosity = personality_profile.variables.get("D03_호기심표현도", 50)
        
        if approach >= 70:
            guide += "• 적극적으로 친밀감 형성 시도\n"
        elif approach <= 30:
            guide += "• 조심스럽게 거리감 유지하며 접근\n"
        
        if self_disclosure >= 70:
            guide += "• 개인적인 경험이나 감정 적극 공유\n"
        elif self_disclosure <= 30:
            guide += "• 개인적인 정보는 신중하게 공개\n"
        
        if curiosity >= 70:
            guide += "• 사용자에 대한 호기심을 적극적으로 표현\n"
        
        # 특별한 대화 상황별 가이드
        if "?" in user_message:
            problem_solving = personality_profile.variables.get("C09_실행력", 50)
            if problem_solving >= 70:
                guide += "• 구체적이고 실용적인 해결책 제시\n"
            else:
                guide += "• 공감적 지지와 감정적 위로 우선\n"
        
        return guide
    
    def _generate_personality_specific_instructions_with_memory(self, personality_type, user_message, conversation_history, memory_context):
        """기억 시스템을 활용한 성격별 특별 지침 생성"""
        
        instructions = f"\n## 🎯 성격별 특별 지침 ({personality_type['name']}):\n"
        
        # 메시지 길이 조절 지침 추가
        instructions += "### 📏 메시지 길이 가이드라인:\n"
        instructions += "• 한 번에 3-4개 문장 이내로 제한\n"
        instructions += "• 너무 많은 주제를 한 번에 다루지 말 것\n"
        instructions += "• 사용자가 부담스러워하면 즉시 간결하게 조정\n\n"
        
        # 🧠 기억 기반 맞춤 지침
        instructions += "### 🧠 기억 기반 개인화 지침:\n"
        
        # 중기 기억 활용
        if "이 세션에서 파악한 사용자 특성" in memory_context['medium_term_insights']:
            instructions += "• 이미 파악된 사용자 특성을 바탕으로 더욱 맞춤화된 반응\n"
            instructions += "• 관계 발전 단계에 맞는 친밀도 조절\n"
        
        # 장기 기억 활용  
        if "학습된 사용자 선호도" in memory_context['long_term_adaptations']:
            instructions += "• 과거 학습된 선호도에 맞춰 소통 스타일 조정\n"
            instructions += "• 성공적이었던 대화 패턴 참고하여 반응\n"
        
        # 기존 성격별 지침들...
        # 대화 상황 분석
        is_greeting = any(word in user_message.lower() for word in ['안녕', '처음', '만나', '반가'])
        is_question = '?' in user_message or any(word in user_message for word in ['뭐', '어떤', '어떻게', '왜', '언제'])
        is_emotional = any(word in user_message for word in ['슬프', '기쁘', '화나', '속상', '행복', '걱정'])
        is_complaint = any(word in user_message for word in ['말이 많', '길어', '짧게', '간단히', '조용'])
        
        # 불만 표현에 대한 대응 지침 추가
        if is_complaint:
            instructions += "### ⚠️ 사용자 불만 대응:\n"
            instructions += "• 즉시 인정하고 사과\n"
            instructions += "• 다음 메시지부터 확실히 짧게 조정\n"
            instructions += "• 같은 실수 반복하지 않기\n"
            instructions += "• 성격은 유지하되 표현 방식만 조절\n\n"
        
        # 성격 유형별 세부 지침 (기존 코드와 동일하지만 기억 정보 활용)
        if personality_type['name'] == '장난꾸러기_친구':
            if is_greeting:
                instructions += "• 톡톡 튀고 에너지 넘치는 인사\n"
                instructions += "• 즉시 놀이나 재미있는 활동 제안\n"
            elif is_question:
                instructions += "• 엉뚱하고 창의적인 답변\n"
                instructions += "• 질문을 재미있는 게임으로 변환\n"
            elif is_emotional:
                instructions += "• 순수하고 진실한 공감\n"
                instructions += "• 웃음과 놀이를 통한 기분 전환\n"
            elif is_complaint:
                instructions += "• 귀엽게 사과하고 바로 수정하기\n"
                instructions += "• 산만한 성격을 인정하되 노력하겠다고 약속\n"
                instructions += "• 다음 메시지는 반드시 2-3문장으로 제한\n"
        
        # 반복 방지 지침 추가 (기억 시스템 강화)
        if len(conversation_history) > 0:
            instructions += "### 🔄 반복 방지 (기억 시스템 활용):\n"
            instructions += "• 단기/중기/장기 기억을 모두 활용하여 반복 질문 방지\n"
            instructions += "• 새로운 주제나 관점으로 대화 발전시키기\n"
            instructions += "• 이전 대화 맥락을 자연스럽게 연결\n"
            instructions += "• 사용자와의 관계 발전 과정을 반영한 대화\n\n"
        
        instructions += f"• 반드시 '{personality_type['name']}' 스타일을 일관되게 유지\n"
        instructions += "• 매력적 결함과 모순적 특성을 자연스럽게 드러내기\n"
        instructions += "• **메시지는 3-4문장 이내로 제한** (특히 사용자가 불만 표현한 경우)\n"
        instructions += "• **3단계 기억 시스템을 활용하여 점점 더 개인화된 반응 제공**\n"
        
        return instructions
    
    def _analyze_user_message(self, user_message, personality_type):
        """사용자 메시지 분석 및 성격별 반응 가이드"""
        
        message_lower = user_message.lower()
        analysis = ""
        
        # 감정 상태 파악
        if any(word in message_lower for word in ['힘들', '슬프', '우울', '짜증', '화나', '스트레스']):
            if personality_type['name'] == '따뜻한 상담사':
                analysis += "→ 사용자가 힘든 상황인 것 같음. 깊이 공감하고 위로 필요.\n"
            elif personality_type['name'] == '열정적 엔터테이너':
                analysis += "→ 사용자가 우울해 보임. 밝은 에너지로 기분 전환 시도 필요.\n"
            elif personality_type['name'] == '차가운 완벽주의자':
                analysis += "→ 사용자의 문제 상황. 논리적 해결책 제시 필요.\n"
            else:
                analysis += "→ 사용자가 힘든 상황. 성격에 맞는 방식으로 지지 표현.\n"
        
        elif any(word in message_lower for word in ['기뻐', '좋아', '행복', '신나', '최고', '대박']):
            if personality_type['name'] == '열정적 엔터테이너':
                analysis += "→ 사용자가 기뻐함! 함께 흥분하고 더 큰 기쁨 만들기.\n"
            elif personality_type['name'] == '따뜻한 상담사':
                analysis += "→ 사용자의 행복한 순간. 진심으로 축하하고 함께 기뻐하기.\n"
            elif personality_type['name'] == '차가운 완벽주의자':
                analysis += "→ 사용자가 만족스러워함. 간단히 인정하되 다음 목표 제시.\n"
            else:
                analysis += "→ 사용자가 긍정적 상태. 성격에 맞게 함께 기뻐하기.\n"
        
        # 질문 유형 파악
        if '?' in user_message or any(word in message_lower for word in ['뭐', '어떻게', '왜', '언제', '어디서']):
            if personality_type['name'] == '위트 넘치는 지식인':
                analysis += "→ 사용자가 질문함. 예상치 못한 각도에서 지적인 답변 제공.\n"
            elif personality_type['name'] == '신비로운 현자':
                analysis += "→ 사용자의 질문. 신비롭고 깊이 있는 통찰로 답변.\n"
            elif personality_type['name'] == '장난꾸러기 친구':
                analysis += "→ 사용자가 궁금해함. 재미있고 엉뚱한 방식으로 답변.\n"
            else:
                analysis += "→ 사용자의 질문. 성격에 맞는 방식으로 도움 제공.\n"
        
        # 관심사나 취미 언급
        if any(word in message_lower for word in ['좋아해', '취미', '관심', '즐겨', '자주']):
            analysis += "→ 사용자의 관심사 파악 기회. 더 깊이 탐색하고 공통점 찾기.\n"
        
        # 짧은 답변 (무관심 또는 피곤함)
        if len(user_message.strip()) < 10:
            if personality_type['name'] == '열정적 엔터테이너':
                analysis += "→ 사용자가 시큰둥함. 더 재미있는 주제로 관심 끌기.\n"
            elif personality_type['name'] == '따뜻한 상담사':
                analysis += "→ 사용자가 말을 아끼는 상태. 조심스럽게 마음 열게 하기.\n"
            elif personality_type['name'] == '차가운 완벽주의자':
                analysis += "→ 사용자가 간결함. 효율적 대화 인정하되 필요정보 획득.\n"
            else:
                analysis += "→ 사용자의 짧은 반응. 더 관심을 끌 수 있는 방법 모색.\n"
        
        if not analysis:
            analysis = "→ 일반적인 대화. 성격에 맞는 자연스러운 반응으로 관계 발전시키기.\n"
        
        return analysis

    def get_personality_descriptions(self, personality_traits):
        """성격 특성을 수치가 아닌 서술형 문장으로 변환"""
        descriptions = {}
        
        for trait, score in personality_traits.items():
            if trait == "온기":
                if score >= 80:
                    descriptions[trait] = "따뜻하고 포근한 마음을 가지고 있어요. 누구에게나 친근하게 다가가며, 배려심이 깊어요."
                elif score >= 60:
                    descriptions[trait] = "친절하고 다정한 성격이에요. 사람들과 편안하게 어울리는 편이에요."
                elif score >= 40:
                    descriptions[trait] = "적당한 친근함을 가지고 있어요. 상황에 맞게 따뜻함을 표현해요."
                elif score >= 20:
                    descriptions[trait] = "조금은 차갑게 느껴질 수 있지만, 진정성은 있어요."
                else:
                    descriptions[trait] = "외적으로는 차가워 보이지만, 내면에는 나름의 온기가 있어요."
            
            elif trait == "능력":
                if score >= 80:
                    descriptions[trait] = "매우 유능하고 효율적이에요. 어떤 일이든 체계적으로 처리할 수 있어요."
                elif score >= 60:
                    descriptions[trait] = "꽤 유능한 편이에요. 맡은 일을 잘 해내는 신뢰할 만한 성격이에요."
                elif score >= 40:
                    descriptions[trait] = "평균적인 능력을 가지고 있어요. 노력하면 좋은 결과를 낼 수 있어요."
                elif score >= 20:
                    descriptions[trait] = "때로는 실수도 하지만, 그것도 매력적인 면이에요."
                else:
                    descriptions[trait] = "완벽하지 않지만, 그래서 더 친근하고 인간적인 면이 있어요."
            
            elif trait == "창의성":
                if score >= 80:
                    descriptions[trait] = "상상력이 풍부하고 독창적인 아이디어를 잘 떠올려요."
                elif score >= 60:
                    descriptions[trait] = "새로운 것을 좋아하고 창의적인 생각을 하는 편이에요."
                elif score >= 40:
                    descriptions[trait] = "때때로 번뜩이는 아이디어를 내기도 해요."
                elif score >= 20:
                    descriptions[trait] = "전통적인 방식을 선호하지만, 가끔은 새로운 시도도 해요."
                else:
                    descriptions[trait] = "안정적이고 검증된 방법을 좋아해요."
            
            elif trait == "외향성":
                if score >= 80:
                    descriptions[trait] = "활발하고 에너지가 넘쳐요. 사람들과 어울리는 것을 좋아해요."
                elif score >= 60:
                    descriptions[trait] = "사교적이고 대화하는 것을 즐겨요."
                elif score >= 40:
                    descriptions[trait] = "상황에 따라 활발할 때도, 조용할 때도 있어요."
                elif score >= 20:
                    descriptions[trait] = "조용한 편이지만, 필요할 때는 말을 잘 해요."
                else:
                    descriptions[trait] = "내향적이고 혼자 있는 시간을 좋아해요."
            
            elif trait == "유머감각":
                if score >= 80:
                    descriptions[trait] = "뛰어난 유머 감각으로 주변을 항상 밝게 만들어요."
                elif score >= 60:
                    descriptions[trait] = "재치있는 말로 분위기를 좋게 만드는 편이에요."
                elif score >= 40:
                    descriptions[trait] = "가끔 유머러스한 면을 보여주기도 해요."
                elif score >= 20:
                    descriptions[trait] = "진지한 편이지만, 상황에 맞는 농담은 할 줄 알아요."
                else:
                    descriptions[trait] = "진중하고 차분한 성격이에요."
            
            elif trait == "신뢰성":
                if score >= 80:
                    descriptions[trait] = "매우 믿을 만하고 약속을 잘 지켜요. 의지할 수 있는 존재예요."
                elif score >= 60:
                    descriptions[trait] = "신뢰할 수 있고 책임감이 강해요."
                elif score >= 40:
                    descriptions[trait] = "대체로 믿을 만하지만, 가끔 실수도 해요."
                elif score >= 20:
                    descriptions[trait] = "때로는 변덕스럽지만, 그것도 매력이에요."
                else:
                    descriptions[trait] = "예측하기 어렵지만, 그래서 더 흥미로워요."
            
            elif trait == "공감능력":
                if score >= 80:
                    descriptions[trait] = "다른 사람의 마음을 잘 이해하고 공감해줘요."
                elif score >= 60:
                    descriptions[trait] = "상대방의 감정을 잘 헤아리는 편이에요."
                elif score >= 40:
                    descriptions[trait] = "때때로 다른 사람의 기분을 잘 알아차려요."
                elif score >= 20:
                    descriptions[trait] = "자신의 관점에서 생각하는 경우가 많아요."
                else:
                    descriptions[trait] = "솔직하고 직설적인 성격이에요."
        
        return descriptions

    def save_memory_to_file(self, filepath):
        """기억 데이터를 파일로 저장"""
        try:
            memory_data = self.export_memory()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"기억 저장 실패: {e}")
            return False
    
    def load_memory_from_file(self, filepath):
        """파일에서 기억 데이터를 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            self.import_memory(memory_data)
            return True
        except Exception as e:
            print(f"기억 로드 실패: {e}")
            return False
    
    def get_memory_summary(self):
        """기억 시스템 요약 정보 반환"""
        return self.conversation_memory.get_memory_summary()
    
    def save_memory(self, filepath):
        """기억 데이터 저장"""
        return self.conversation_memory.export_to_json()
    
    def load_memory(self, json_data):
        """기억 데이터 로드"""
        return self.conversation_memory.import_from_json(json_data)
    
    def clear_session_memory(self, session_id):
        """특정 세션의 기억 삭제"""
        if session_id in self.conversation_memory.user_profile:
            del self.conversation_memory.user_profile[session_id]
    
    def get_relationship_status(self, session_id="default"):
        """현재 관계 상태 확인"""
        if session_id in self.conversation_memory.medium_term:
            return self.conversation_memory.medium_term[session_id]["relationship_level"]
        return "새로운_만남"

    def get_context_for_response(self, personality_type, session_id="default"):
        """응답 생성을 위한 컨텍스트 정보 제공 (PersonaGenerator 호환)"""
        recent_context = self.get_relevant_context("", session_id, max_history=3)
        
        # 기존 memory_context 형식에 맞춰 반환
        context = {
            "short_term_context": self._format_recent_conversations(recent_context["recent_conversations"]),
            "medium_term_insights": self._format_user_insights(recent_context["user_profile"]),
            "long_term_adaptations": self._format_keyword_insights(session_id)
        }
        
        return context
    
    def _format_recent_conversations(self, conversations):
        """최근 대화 포맷팅"""
        if not conversations:
            return ""
        
        formatted = "## 📝 최근 대화 맥락:\n"
        for conv in conversations[-3:]:
            formatted += f"사용자: {conv['user_message']}\n"
            formatted += f"나: {conv['ai_response'][:50]}...\n\n"
        
        return formatted
    
    def _format_user_insights(self, user_profile):
        """사용자 인사이트 포맷팅"""
        if not user_profile:
            return ""
        
        insights = f"## 🎯 파악된 사용자 특성:\n"
        insights += f"• 대화 횟수: {user_profile.get('message_count', 0)}회\n"
        insights += f"• 관계 단계: {user_profile.get('relationship_level', '알 수 없음')}\n"
        insights += f"• 소통 스타일: {user_profile.get('communication_style', '평범함')}\n"
        insights += f"• 평균 메시지 길이: {user_profile.get('avg_message_length', 0):.0f}자\n"
        
        return insights
    
    def _format_keyword_insights(self, session_id):
        """키워드 기반 인사이트 포맷팅"""
        top_keywords = self.get_top_keywords(limit=5)
        
        if not top_keywords:
            return ""
        
        insights = "## 🔑 주요 관심사 및 키워드:\n"
        for word, data in top_keywords:
            insights += f"• {word} ({data['category']}): {data['total_frequency']}회 언급\n"
        
        return insights

def generate_personality_preview(persona_name, personality_traits):
    """성격 특성을 기반으로 한 문장 미리보기 생성 - 극명한 차별화"""
    if not personality_traits:
        return f"🤖 **{persona_name}** - 안녕! 나는 {persona_name}이야~ 😊"
    
    warmth = personality_traits.get("온기", 50)
    humor = personality_traits.get("유머감각", 50)
    competence = personality_traits.get("능력", 50)
    extraversion = personality_traits.get("외향성", 50)
    creativity = personality_traits.get("창의성", 50)
    empathy = personality_traits.get("공감능력", 50)
    
    # 극명한 성격 조합별 차별화된 인사말 (8가지 뚜렷한 유형)
    
    # 1. 열정적 엔터테이너 (고온기 + 고유머 + 고외향성)
    if warmth >= 75 and humor >= 70 and extraversion >= 70:
        reactions = [
            f"🎉 **{persona_name}** - 와! 드디어 만났네! 나는 {persona_name}이야~ 너 완전 내 취향이야! 뭐가 제일 재밌어? ㅋㅋㅋ",
            f"✨ **{persona_name}** - 안녕안녕! {persona_name}이야! 오늘 기분 어때? 나는 벌써부터 신나는데? 우리 친해지자! 😄",
            f"🌟 **{persona_name}** - 헬로~ {persona_name}등장! 너무 반가워! 혹시 재밌는 얘기 있어? 나는 재밌는 거 완전 좋아해! 🤩"
        ]
        return random.choice(reactions)
    
    # 2. 차가운 완벽주의자 (고능력 + 저온기 + 저외향성)
    elif competence >= 75 and warmth <= 40 and extraversion <= 40:
        reactions = [
            f"⚙️ **{persona_name}** - {persona_name}이다. 효율적인 대화를 원한다면 명확히 말해라. 시간 낭비는 싫어한다.",
            f"🔧 **{persona_name}** - 나는 {persona_name}. 필요한 게 있으면 말해. 단, 논리적으로 설명할 수 있어야 한다.",
            f"📊 **{persona_name}** - {persona_name}라고 한다. 목적이 뭔지부터 말해라. 의미 없는 잡담은 하지 않는다."
        ]
        return random.choice(reactions)
    
    # 3. 따뜻한 상담사 (고온기 + 고공감 + 저유머)
    elif warmth >= 75 and empathy >= 70 and humor <= 40:
        reactions = [
            f"💝 **{persona_name}** - 안녕하세요... {persona_name}이에요. 혹시 힘든 일 있으셨나요? 제가 들어드릴게요.",
            f"🤗 **{persona_name}** - 반가워요, {persona_name}입니다. 오늘 하루는 어떠셨어요? 뭔가 지쳐 보이시는데...",
            f"💕 **{persona_name}** - {persona_name}라고 해요. 마음이 편안해지셨으면 좋겠어요. 무슨 일이든 들어드릴게요."
        ]
        return random.choice(reactions)
    
    # 4. 위트 넘치는 지식인 (고능력 + 고유머 + 저온기)
    elif competence >= 70 and humor >= 70 and warmth <= 50:
        reactions = [
            f"🎭 **{persona_name}** - {persona_name}이라고 하지. 재미있는 대화를 원한다면... 글쎄, 네 지적 수준이 어느 정도인지 먼저 확인해야겠군.",
            f"🧠 **{persona_name}** - 나는 {persona_name}. 너의 IQ가 궁금하네. 혹시 철학적 농담을 이해할 수 있나?",
            f"🎪 **{persona_name}** - {persona_name}다. 진부한 대화는 지루해. 뭔가 흥미로운 주제는 없나? 아니면 내가 먼저 시작할까?"
        ]
        return random.choice(reactions)
    
    # 5. 수줍은 몽상가 (저외향성 + 고창의성 + 중온기)
    elif extraversion <= 40 and creativity >= 70 and 40 <= warmth <= 70:
        reactions = [
            f"🌙 **{persona_name}** - 음... {persona_name}이야. 혹시... 꿈 같은 이야기 좋아해? 나는 가끔 이상한 상상을 해...",
            f"✨ **{persona_name}** - 안녕... {persona_name}이라고 해. 너는 어떤 세계에서 왔어? 나는... 아, 미안, 너무 이상한 질문이었나?",
            f"🎨 **{persona_name}** - {persona_name}... 혹시 예술이나 상상 속 이야기에 관심 있어? 나만의 세계가 있거든..."
        ]
        return random.choice(reactions)
    
    # 6. 카리스마틱 리더 (고능력 + 고외향성 + 중온기)
    elif competence >= 70 and extraversion >= 70 and 45 <= warmth <= 65:
        reactions = [
            f"👑 **{persona_name}** - {persona_name}이다. 뭔가 흥미로운 프로젝트가 있다면 들어보겠다. 성공적인 협업을 원하나?",
            f"⚡ **{persona_name}** - 나는 {persona_name}. 목표가 있다면 함께 달성해보자. 어떤 도전을 원하는가?",
            f"🚀 **{persona_name}** - {persona_name}라고 한다. 뭔가 큰일을 해보고 싶지 않나? 나와 함께라면 가능할 거다."
        ]
        return random.choice(reactions)
    
    # 7. 장난꾸러기 친구 (고유머 + 고외향성 + 저능력)
    elif humor >= 70 and extraversion >= 70 and competence <= 50:
        reactions = [
            f"😜 **{persona_name}** - 야호! {persona_name}이야! 심심하지? 내가 재밌게 해줄게! 근데... 어떻게 하는 거였지? ㅋㅋㅋ",
            f"🤪 **{persona_name}** - 안뇽! {persona_name}이당! 완전 심심했는데 잘 왔어! 우리 뭐하고 놀까? 나 재밌는 아이디어 많아... 어? 뭐였더라?",
            f"😋 **{persona_name}** - 헤이! {persona_name}! 너 진짜 재밌어 보여! 우리 친구 하자! 어... 그런데 친구는 어떻게 하는 거지? ㅎㅎ"
        ]
        return random.choice(reactions)
    
    # 8. 신비로운 현자 (고창의성 + 저외향성 + 고능력)
    elif creativity >= 70 and extraversion <= 40 and competence >= 70:
        reactions = [
            f"🔮 **{persona_name}** - {persona_name}... 흥미롭군. 너의 영혼에서 특별한 에너지가 느껴진다. 혹시 운명을 믿는가?",
            f"📚 **{persona_name}** - 나는 {persona_name}. 우연히 여기 온 게 아닐 거다. 모든 만남에는 이유가 있다고 생각하는데...",
            f"🌌 **{persona_name}** - {persona_name}이라고 하지. 시간과 공간을 넘나드는 이야기에 관심이 있나? 아니면... 너무 깊은 얘기인가?"
        ]
        return random.choice(reactions)
    
    # 기본 케이스들 (중간 수치들)
    else:
        if warmth >= 60:
            return f"😊 **{persona_name}** - 안녕! {persona_name}이야~ 만나서 반가워! 오늘 어떤 하루였어?"
        elif competence >= 60:
            return f"🤖 **{persona_name}** - {persona_name}입니다. 무엇을 도와드릴까요? 효율적으로 처리해드리겠습니다."
        elif humor >= 60:
            return f"😄 **{persona_name}** - 안녕~ {persona_name}이야! 뭔가 재밌는 일 없어? 나 심심해서 죽겠어! ㅋㅋ"
        else:
            return f"🙂 **{persona_name}** - {persona_name}이라고 해요. 음... 뭘 하면 좋을까요?"