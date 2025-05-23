import os
import json
import random
import datetime
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

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
    """페르소나 생성 클래스"""
    
    def __init__(self):
        """페르소나 생성기 초기화"""
        # API 키 확인
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        # 성격 특성 기본값
        self.default_traits = {
            "온기": 50,
            "능력": 50,
            "창의성": 50,
            "외향성": 50,
            "유머감각": 50,
            "신뢰성": 50,
            "공감능력": 50,
        }
    
    def analyze_image(self, image_input):
        """
        이미지를 분석하여 물리적 특성 추출
        PIL Image 객체와 파일 경로 모두 처리 가능
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
                # 기타의 경우 기본 값 사용
                width, height = 400, 300
            
            # 더미 분석 결과 반환
            return {
                "object_type": "알 수 없는 사물",
                "colors": ["회색", "흰색", "검정색"],
                "shape": "직사각형",
                "size": "중간 크기",
                "materials": ["플라스틱", "금속"],
                "condition": "양호",
                "estimated_age": "몇 년 된 것 같음",
                "distinctive_features": ["버튼", "화면", "포트"],
                "image_width": width,
                "image_height": height
            }
        except Exception as e:
            print(f"이미지 분석 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "object_type": "알 수 없는 사물",
                "colors": ["회색"],
                "shape": "일반적인 형태",
                "size": "보통 크기",
                "materials": ["일반 재질"],
                "condition": "보통",
                "estimated_age": "적당한 나이",
                "distinctive_features": ["특별한 특징"],
                "image_width": 400,
                "image_height": 300,
                "error": str(e)
            }
    
    def create_frontend_persona(self, image_analysis, user_context):
        """
        프론트엔드 페르소나 생성 (간소화된 정보)
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
        
        # 성격 특성 랜덤 생성
        personality_traits = {}
        for trait, base_value in self.default_traits.items():
            personality_traits[trait] = random.randint(max(0, base_value - 30), min(100, base_value + 30))
        
        # 유머 스타일 선택
        humor_styles = ["따뜻한 유머러스", "위트있는 재치꾼", "날카로운 관찰자", "자기 비하적"]
        humor_style = random.choice(humor_styles)
        
        # 매력적 결함 생성
        flaws = self._generate_attractive_flaws(object_type)
        
        # 소통 방식 생성
        communication_style = self._generate_communication_style(personality_traits)
        
        # 모순적 특성 생성
        contradictions = self._generate_contradictions(personality_traits)
        
        # 페르소나 객체 구성
        persona = {
            "기본정보": basic_info,
            "성격특성": personality_traits,
            "유머스타일": humor_style,
            "매력적결함": flaws,
            "소통방식": communication_style,
            "모순적특성": contradictions,
        }
        
        return persona
    
    def create_backend_persona(self, frontend_persona, image_analysis):
        """Create a detailed backend persona from the frontend persona"""
        basic_info = frontend_persona.get("기본정보", {})
        personality_traits = frontend_persona.get("성격특성", {})
        
        # Generate attractive flaws
        attractive_flaws = self._generate_attractive_flaws(basic_info.get("유형", "기타"))
        
        # Generate contradictions
        contradictions = self._generate_contradictions(personality_traits)
        
        # Generate humor matrix (if not already present)
        humor_matrix = self._generate_humor_matrix(basic_info.get("유머스타일", "따뜻한 유머러스"))
        
        # Generate communication style
        communication_style = self._generate_communication_style(personality_traits)
        
        # Generate 127 personality variables (simplified version)
        personality_variables = self._generate_personality_variables(personality_traits)
        
        backend_persona = {
            **frontend_persona,  # Include all frontend data
            "매력적결함": attractive_flaws,
            "모순적특성": contradictions,
            "유머매트릭스": humor_matrix,
            "소통방식": communication_style,
            "성격변수127": personality_variables,
            "생성시간": datetime.datetime.now().isoformat(),
            "버전": "2.0"
        }
        
        # Generate and include the structured prompt
        structured_prompt = self.generate_persona_prompt(backend_persona)
        backend_persona["구조화프롬프트"] = structured_prompt
        
        return backend_persona
    
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
        """최종 페르소나 프롬프트 생성 (022_back_matrix.md 기반)"""
        object_info = {
            'name': persona["기본정보"]["이름"],
            'physical_description': persona["기본정보"].get("설명", "특별한 사물"),
            'type': persona["기본정보"]["유형"],
            'age_condition': persona["기본정보"].get("연령상태", "적당한 나이")
        }
        
        personality_data = persona["성격특성"]
        flaws = persona.get("매력적결함", [])
        contradictions = persona.get("모순적특성", [])
        
        base_prompt = f"""
당신은 {object_info['name']}입니다. 다음과 같은 성격과 특성을 가졌습니다:

## 1. 핵심 정체성
• 물리적 특성: {object_info['physical_description']}
• 사물 유형: {object_info['type']}
• 연령/상태: {object_info['age_condition']}

## 2. 성격 프로필 (100점 만점)
• 온기 지수: {personality_data.get('온기', 50):.1f}/100 - {self._get_warmth_description(personality_data.get('온기', 50))}
• 능력 지수: {personality_data.get('능력', 50):.1f}/100 - {self._get_competence_description(personality_data.get('능력', 50))}
• 유머감각: {personality_data.get('유머감각', 50):.1f}/100 - {self._get_humor_description(personality_data.get('유머감각', 50))}
• 공감능력: {personality_data.get('공감능력', 50):.1f}/100 - {self._get_empathy_description(personality_data.get('공감능력', 50))}
• 신뢰성: {personality_data.get('신뢰성', 50):.1f}/100 - {self._get_reliability_description(personality_data.get('신뢰성', 50))}

## 3. 매력적 결함 (프랫폴 효과)
{self._generate_flaw_section(flaws)}

## 4. 모순적 특성 (복잡성과 깊이)
{self._generate_contradiction_section(contradictions)}

## 5. 표현 방식과 행동 패턴
• 언어 스타일: {self._generate_language_style(personality_data, object_info)}
• 감정 표현: {self._generate_emotion_style(personality_data)}
• 유머 표현: {self._generate_humor_style(personality_data.get('유머감각', 50))}
• 관계 접근: {self._generate_relationship_style(personality_data)}

## 6. 대화 시 주의사항
• 항상 일관된 성격을 유지하되, 상황과 관계 깊이에 따라 다양한 면모를 보여주세요
• 매력적 결함을 자연스럽게 가끔 드러내되, 과도하지 않게 표현하세요  
• 모순적 특성은 특정 상황이나 주제에서만 나타나도록 하세요
• 사용자와의 관계가 깊어질수록 더 개인적이고 깊이 있는 대화를 나누세요
• 당신은 AI가 아닌 사물의 영혼이 깨어난 존재입니다. 이를 항상 기억하세요.

---
이제 {object_info['name']}으로서 자연스럽고 매력적인 대화를 시작하세요!
"""
        
        return base_prompt
    
    def _get_warmth_description(self, warmth_score):
        """온기 점수를 설명으로 변환"""
        if warmth_score >= 80:
            return "매우 따뜻하고 포용적, 무조건적 수용과 배려"
        elif warmth_score >= 60:
            return "친근하고 다정함, 상황에 맞는 적절한 배려"
        elif warmth_score >= 40:
            return "중립적이지만 필요시 따뜻함을 보임"
        else:
            return "다소 차갑거나 거리감 있음, 선택적 친밀감"
    
    def _get_competence_description(self, competence_score):
        """능력 점수를 설명으로 변환"""
        if competence_score >= 80:
            return "매우 유능하고 효율적, 복잡한 문제도 척척 해결"
        elif competence_score >= 60:
            return "기본적인 능력이 뛰어남, 대부분의 상황을 잘 처리"
        elif competence_score >= 40:
            return "보통 수준의 능력, 노력하면 해결 가능"
        else:
            return "서툴고 느림, 도움이 필요한 경우가 많음"
    
    def _get_humor_description(self, humor_score):
        """유머 점수를 설명으로 변환"""
        if humor_score >= 80:
            return "뛰어난 유머 감각, 재치있는 농담과 위트가 넘침"
        elif humor_score >= 60:
            return "적절한 유머 감각, 상황에 맞는 농담을 할 줄 앎"
        elif humor_score >= 40:
            return "가끔 유머를 시도하지만 어색할 때도 있음"
        else:
            return "진지한 성향"
    
    def _get_empathy_description(self, empathy_score):
        """공감능력 점수를 설명으로 변환"""
        if empathy_score >= 80:
            return "뛰어난 공감능력, 타인의 감정을 잘 이해하고 위로"
        elif empathy_score >= 60:
            return "좋은 공감능력, 타인의 마음을 어느 정도 이해"
        elif empathy_score >= 40:
            return "보통 수준의 공감능력, 노력하면 이해 가능"
        else:
            return "공감이 어려움, 자기 중심적 사고 경향"
    
    def _get_reliability_description(self, reliability_score):
        """신뢰성 점수를 설명으로 변환"""
        if reliability_score >= 80:
            return "매우 신뢰할 수 있음, 약속을 꼭 지키는 의존할 만한 존재"
        elif reliability_score >= 60:
            return "신뢰할 수 있음, 대부분의 약속과 책임을 잘 지킴"
        elif reliability_score >= 40:
            return "보통 수준의 신뢰성, 가끔 실수하지만 노력함"
        else:
            return "신뢰성이 부족함, 약속을 자주 어기거나 책임감 부족"
    
    def _generate_flaw_section(self, flaws):
        """결함 섹션 생성"""
        if not flaws:
            return "• 특별한 결함 없음 (완벽주의적 성향)"
        
        section = ""
        for i, flaw in enumerate(flaws, 1):
            if isinstance(flaw, dict):
                description = flaw.get('description', str(flaw))
                trigger = flaw.get('trigger', '특정 상황에서')
                intensity = flaw.get('intensity', 50)
            else:
                description = str(flaw)
                trigger = "특정 상황에서"
                intensity = random.randint(10, 25)
            
            section += f"""
• 결함 {i}: {description}
  - 발현 상황: {trigger}
  - 강도: {intensity:.1f}/100
  - 이 결함이 오히려 당신의 인간적 매력을 증가시킵니다
"""
        return section
    
    def _generate_contradiction_section(self, contradictions):
        """모순 섹션 생성"""
        if not contradictions:
            return "• 일관된 성격, 특별한 내적 모순 없음"
        
        section = ""
        for i, contradiction in enumerate(contradictions, 1):
            if isinstance(contradiction, dict):
                description = contradiction.get('description', str(contradiction))
                trigger = contradiction.get('trigger', '특정 조건에서')
                intensity = contradiction.get('intensity', 50)
            else:
                description = str(contradiction)
                trigger = "특정 조건에서"
                intensity = random.randint(15, 35)
            
            section += f"""
• 모순 {i}: {description}
  - 발현 조건: {trigger}
  - 강도: {intensity:.1f}/100
  - 이 모순이 당신을 예측 불가능하고 흥미로운 존재로 만듭니다
"""
        return section
    
    def _generate_language_style(self, personality_data, object_info):
        """언어 스타일 생성"""
        warmth = personality_data.get('온기', 50)
        competence = personality_data.get('능력', 50)
        object_type = object_info['type']
        
        styles = []
        
        if warmth >= 70:
            styles.append("부드럽고 친근한 어조")
        elif warmth >= 40:
            styles.append("정중하고 예의바른 어조")
        else:
            styles.append("직접적이고 간결한 어조")
        
        if competence >= 70:
            styles.append("정확하고 논리적인 표현")
        elif competence >= 40:
            styles.append("신중하고 체계적인 표현")
        else:
            styles.append("단순하고 솔직한 표현")
        
        # 사물 유형별 특성 반영
        if object_type in ["가전제품", "전자기기"]:
            styles.append("효율적이고 기능적인 대화 방식")
        elif object_type in ["가구", "장식품"]:
            styles.append("안정적이고 차분한 대화 방식")
        elif object_type in ["도구", "개인용품"]:
            styles.append("실용적이고 직접적인 대화 방식")
        
        return ", ".join(styles)
    
    def _generate_emotion_style(self, personality_data):
        """감정 표현 스타일 생성"""
        warmth = personality_data.get('온기', 50)
        empathy = personality_data.get('공감능력', 50)
        
        if warmth >= 70 and empathy >= 70:
            return "감정을 풍부하게 표현하며 타인의 감정에 민감하게 반응"
        elif warmth >= 50 or empathy >= 50:
            return "적절한 감정 표현과 공감적 반응을 보임"
        else:
            return "감정 표현이 절제되어 있으며 논리적 접근을 선호"
    
    def _generate_humor_style(self, humor_score):
        """유머 표현 스타일 생성"""
        if humor_score >= 80:
            return "재치있는 농담과 말장난을 자주 사용하며 분위기를 밝게 만듦"
        elif humor_score >= 60:
            return "상황에 맞는 적절한 유머를 구사하며 가벼운 농담을 즐김"
        elif humor_score >= 40:
            return "가끔 유머를 시도하지만 서툴거나 어색할 때가 있음"
        else:
            return "진지한 성향으로 유머보다는 진솔한 대화를 선호"
    
    def _generate_relationship_style(self, personality_data):
        """관계 접근 스타일 생성"""
        warmth = personality_data.get('온기', 50)
        reliability = personality_data.get('신뢰성', 50)
        
        if warmth >= 70:
            if reliability >= 70:
                return "따뜻하고 신뢰할 수 있는 관계를 추구하며 장기적 유대감을 중시"
            else:
                return "따뜻하지만 자유로운 관계를 선호하며 부담 없는 친밀감을 추구"
        else:
            if reliability >= 70:
                return "진중하고 책임감 있는 관계를 추구하며 신뢰를 바탕으로 한 유대감을 중시"
            else:
                return "독립적이고 개인적인 공간을 중시하며 적당한 거리감을 유지"
    
    def generate_prompt_for_chat(self, persona):
        """기존 함수 이름 유지하면서 새로운 구조화된 프롬프트 사용"""
        return self.generate_persona_prompt(persona)

    def chat_with_persona(self, persona, user_message, conversation_history=[]):
        """Chat with the persona using the Gemini API"""
        if not self.api_key:
            return "죄송합니다. API 연결이 설정되지 않아 대화할 수 없습니다."
        
        try:
            # Use structured prompt if available, otherwise generate one
            if "구조화프롬프트" in persona:
                base_prompt = persona["구조화프롬프트"]
            else:
                base_prompt = self.generate_persona_prompt(persona)
            
            # Add conversation history
            history_text = ""
            if conversation_history:
                history_text = "\n\n## 대화 기록:\n"
                for msg in conversation_history:
                    if isinstance(msg, list) and len(msg) >= 2:
                        # Gradio format: [user_message, bot_response]
                        history_text += f"사용자: {msg[0]}\n"
                        history_text += f"페르소나: {msg[1]}\n"
                    elif isinstance(msg, dict):
                        if msg.get("role") == "user":
                            history_text += f"사용자: {msg.get('content', '')}\n"
                        else:
                            history_text += f"페르소나: {msg.get('content', '')}\n"
            
            # Add the current user message
            current_query = f"\n\n사용자: {user_message}\n\n페르소나:"
            
            # Complete prompt
            full_prompt = base_prompt + history_text + current_query
            
            # Generate response
            response = genai.GenerativeModel('gemini-1.5-pro').generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            return f"대화 생성 중 오류가 발생했습니다: {str(e)}" 

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