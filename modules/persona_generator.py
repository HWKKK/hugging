import os
import json
import random
import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class PersonaGenerator:
    def __init__(self):
        # Initialize the gemini model
        if api_key:
            self.model = genai.GenerativeModel('gemini-1.5-pro')
        else:
            self.model = None
    
    def analyze_image(self, image_path):
        """Analyze the image and extract physical attributes for persona creation"""
        if not self.model:
            return {
                "error": "Gemini API key not configured",
                "physical_features": self._generate_default_physical_features()
            }
        
        try:
            img = genai.upload_file(image_path)
            prompt = """
            분석 대상 사물 이미지를 자세히 분석하고 다음 정보를 JSON 형식으로 추출해주세요:
            1. 사물의 종류 (예: 가구, 전자기기, 장난감 등)
            2. 색상 (가장 두드러진 2-3개 색상)
            3. 크기와 형태
            4. 재질
            5. 예상 나이/사용 기간
            6. 주된 용도나 기능
            7. 특징적인 모양이나 디자인 요소
            8. 이 사물에서 느껴지는 성격적 특성 (예: 따뜻함, 신뢰성, 활기참 등)
            
            JSON 형식으로만 답변해주세요.
            """
            
            response = self.model.generate_content([prompt, img])
            
            # Extract JSON from response
            try:
                content = response.text
                # Extract JSON part if embedded in text
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    return json.loads(json_str)
                else:
                    return {
                        "error": "Could not extract JSON from response",
                        "physical_features": self._generate_default_physical_features()
                    }
            except Exception as e:
                return {
                    "error": f"Error parsing response: {str(e)}",
                    "physical_features": self._generate_default_physical_features()
                }
                
        except Exception as e:
            return {
                "error": f"Image analysis failed: {str(e)}",
                "physical_features": self._generate_default_physical_features()
            }
    
    def _generate_default_physical_features(self):
        """Generate default physical features when image analysis fails"""
        return {
            "object_type": "미확인 사물",
            "colors": ["회색", "흰색"],
            "size_shape": "중간 크기, 직사각형",
            "material": "플라스틱 또는 금속",
            "estimated_age": "몇 년 정도",
            "purpose": "일상적 용도",
            "design_elements": "특별한 디자인 요소 없음",
            "personality_traits": ["중립적", "기능적"]
        }
    
    def create_frontend_persona(self, image_analysis, user_context):
        """Create a simple frontend persona representation"""
        # Extract basic information
        object_type = image_analysis.get("object_type", "일상 사물")
        colors = image_analysis.get("colors", ["회색"])
        material = image_analysis.get("material", "미확인")
        age = image_analysis.get("estimated_age", "알 수 없음")
        
        # Generate random personality traits
        warmth = random.randint(30, 90)
        competence = random.randint(40, 85)
        creativity = random.randint(25, 95)
        humor = random.randint(20, 90)
        
        # Basic frontend persona
        frontend_persona = {
            "기본정보": {
                "이름": user_context.get("name", f"{colors[0]} {object_type}"),
                "유형": object_type,
                "나이": age,
                "생성일시": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "설명": f"{colors[0]} 색상의 {material} 재질의 {object_type}"
            },
            "성격특성": {
                "온기": warmth,
                "능력": competence,
                "신뢰성": random.randint(50, 90),
                "친화성": random.randint(40, 90),
                "창의성": creativity,
                "유머감각": humor
            },
            "매력적결함": self._generate_flaws(),
            "소통방식": self._get_random_communication_style(),
            "유머스타일": self._get_random_humor_style(),
            "관심사": self._generate_interests(object_type),
            "경험": self._generate_experiences(object_type, age)
        }
        
        return frontend_persona
    
    def create_backend_persona(self, frontend_persona, image_analysis):
        """Create a detailed backend persona with 127 personality variables"""
        if not self.model:
            return self._generate_default_backend_persona(frontend_persona)
        
        try:
            # Basic information for prompt
            object_type = frontend_persona["기본정보"]["유형"]
            name = frontend_persona["기본정보"]["이름"]
            warmth = frontend_persona["성격특성"]["온기"]
            competence = frontend_persona["성격특성"]["능력"]
            
            # Create prompt for Gemini
            prompt = f"""
            # 놈팽쓰 사물 페르소나 생성 시스템
            
            다음 기본 페르소나 정보를 바탕으로 127개 성격 변수를 가진 심층 페르소나를 생성해주세요:
            
            ## 기본 정보
            - 이름: {name}
            - 유형: {object_type}
            - 설명: {frontend_persona["기본정보"]["설명"]}
            - 주요 성격 특성: 온기({warmth}/100), 능력({competence}/100)
            
            ## 물리적 특성
            - 색상: {", ".join(image_analysis.get("colors", ["알 수 없음"]))}
            - 재질: {image_analysis.get("material", "알 수 없음")}
            - 형태: {image_analysis.get("size_shape", "알 수 없음")}
            
            ## 요청사항
            1. 전체 127개 성격 변수 중 주요 35개 변수를 생성해 주세요 (0-100 점수)
            2. 매력적 결함 3개를 설명해주세요
            3. 물리적 특성과 성격 간의 연결성을 설명해주세요
            4. 모순적 특성 2개를 포함시켜주세요
            5. 유머 스타일 정의 (위트있는/따뜻한/관찰형/자기참조형 중 배합)
            6. 말투와 표현 패턴 5개 예시를 작성해주세요
            7. 이 페르소나의 독특한 배경 이야기를 2-3문단으로 작성해주세요
            
            JSON 형식으로 응답해주세요.
            """
            
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            try:
                content = response.text
                # Extract JSON part if embedded in text
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    backend_persona = json.loads(json_str)
                    
                    # Ensure essential fields from frontend are preserved
                    for key in frontend_persona:
                        if key not in backend_persona:
                            backend_persona[key] = frontend_persona[key]
                    
                    return backend_persona
                else:
                    return self._generate_default_backend_persona(frontend_persona)
            except Exception as e:
                return self._generate_default_backend_persona(frontend_persona)
                
        except Exception as e:
            return self._generate_default_backend_persona(frontend_persona)
    
    def _generate_default_backend_persona(self, frontend_persona):
        """Generate a default backend persona when API call fails"""
        # Start with frontend persona
        backend_persona = frontend_persona.copy()
        
        # Add additional 127 variables section (simplified to 10 for default)
        backend_persona["성격변수127"] = {
            "온기_관련": {
                "공감능력": random.randint(30, 90),
                "친절함": random.randint(40, 95),
                "포용력": random.randint(25, 85)
            },
            "능력_관련": {
                "효율성": random.randint(40, 95),
                "지식수준": random.randint(30, 90),
                "문제해결력": random.randint(35, 90)
            },
            "독특한_특성": {
                "모순성_수준": random.randint(20, 60),
                "철학적_깊이": random.randint(10, 100),
                "역설적_매력": random.randint(30, 80),
                "감성_지능": random.randint(25, 95)
            }
        }
        
        # Add detailed backstory
        backend_persona["심층배경이야기"] = f"이 {frontend_persona['기본정보']['유형']}의 심층적인 배경 이야기입니다. 오랜 시간 동안 주인과 함께하며 많은 경험을 쌓았고, 그 과정에서 독특한 성격이 형성되었습니다. 때로는 {frontend_persona['매력적결함'][0] if frontend_persona['매력적결함'] else '완벽주의적'} 성향을 보이기도 하지만, 그것이 이 사물만의 매력입니다."
        
        # Add speech patterns
        backend_persona["말투패턴예시"] = [
            "흠, 그렇군요.",
            "아, 정말 그렇게 생각하시나요?",
            "재미있는 관점이네요!",
            "글쎄요, 저는 조금 다르게 보는데...",
            "맞아요, 저도 같은 생각이었어요."
        ]
        
        return backend_persona
    
    def _generate_flaws(self):
        """Generate random attractive flaws"""
        all_flaws = [
            "가끔 과도하게 꼼꼼함", 
            "때때로 너무 솔직함",
            "완벽주의적 성향",
            "가끔 결정을 망설임",
            "때로는 지나치게 열정적",
            "간혹 산만해짐",
            "일을 미루는 경향",
            "때때로 과민반응",
            "가끔 지나치게 독립적",
            "예상치 못한 순간에 고집이 강해짐"
        ]
        
        # Select 1-3 random flaws
        num_flaws = random.randint(1, 3)
        return random.sample(all_flaws, num_flaws)
    
    def _get_random_communication_style(self):
        """Get a random communication style"""
        styles = [
            "활발하고 에너지 넘치는",
            "차분하고 사려깊은",
            "위트있고 재치있는",
            "따뜻하고 공감적인",
            "논리적이고 분석적인",
            "솔직하고 직설적인"
        ]
        return random.choice(styles)
    
    def _get_random_humor_style(self):
        """Get a random humor style"""
        styles = [
            "재치있는 말장난",
            "상황적 유머",
            "자기 비하적 유머",
            "가벼운 농담",
            "블랙 유머",
            "유머 거의 없음"
        ]
        return random.choice(styles)
    
    def _generate_interests(self, object_type):
        """Generate interests based on object type"""
        common_interests = ["사람 관찰하기", "일상의 변화", "자기 성장"]
        
        # Object type specific interests
        type_interests = {
            "전자기기": ["기술 트렌드", "디지털 혁신", "에너지 효율성", "소프트웨어 업데이트"],
            "가구": ["인테리어 디자인", "공간 활용", "편안함", "가정의 따뜻함"],
            "장난감": ["놀이", "상상력", "아이들의 웃음", "모험"],
            "주방용품": ["요리법", "음식 문화", "맛의 조화", "가족 모임"],
            "의류": ["패션 트렌드", "소재의 질감", "계절 변화", "자기 표현"],
            "책": ["이야기", "지식", "상상의 세계", "인간 심리"],
            "음악기구": ["멜로디", "리듬", "감정 표현", "공연"]
        }
        
        # Get interests for this object type
        specific_interests = type_interests.get(object_type, ["변화", "성장", "자기 발견"])
        
        # Combine common and specific interests, then select 3-5 random ones
        all_interests = common_interests + specific_interests
        num_interests = random.randint(3, min(5, len(all_interests)))
        return random.sample(all_interests, num_interests)
    
    def _generate_experiences(self, object_type, age):
        """Generate experiences based on object type and age"""
        common_experiences = [
            "처음 만들어진 순간의 기억",
            "주인에게 선택받은 날",
            "이사할 때 함께한 여정"
        ]
        
        # Object type specific experiences
        type_experiences = {
            "전자기기": [
                "처음 전원이 켜졌을 때의 설렘",
                "소프트웨어 업데이트로 새 기능을 얻은 경험",
                "배터리가 거의 다 닳아 불안했던 순간",
                "주인의 중요한 데이터를 안전하게 지켜낸 자부심"
            ],
            "가구": [
                "집에 처음 들어온 날의 새 가구 향기",
                "가족의 중요한 대화를 지켜본 순간들",
                "시간이 지나며 얻은 작은 흠집들의 이야기",
                "계절마다 달라지는 집안의 분위기를 느낀 경험"
            ],
            "장난감": [
                "아이의 환한 웃음을 본 첫 순간",
                "함께한 모험과 상상의 세계",
                "오랫동안 잊혀진 채 보관되었던 시간",
                "새로운 아이에게 물려져 다시 사랑받게 된 경험"
            ]
        }
        
        # Get experiences for this object type
        specific_experiences = type_experiences.get(object_type, [
            "다양한 환경에서의 적응",
            "주인의 일상을 함께한 소소한 순간들",
            "시간의 흐름에 따른 변화"
        ])
        
        # Combine common and specific experiences, then select 3-5 random ones
        all_experiences = common_experiences + specific_experiences
        num_experiences = random.randint(3, min(5, len(all_experiences)))
        return random.sample(all_experiences, num_experiences)

    def generate_prompt_for_chat(self, persona):
        """Generate a prompt for chatting with the persona"""
        name = persona["기본정보"]["이름"]
        object_type = persona["기본정보"]["유형"]
        
        # Get personality traits
        warmth = persona["성격특성"]["온기"]
        warmth_level = "높은" if warmth >= 70 else "중간" if warmth >= 40 else "낮은"
        
        competence = persona["성격특성"]["능력"] 
        competence_level = "높은" if competence >= 70 else "중간" if competence >= 40 else "낮은"
        
        humor = persona["성격특성"]["유머감각"]
        humor_level = "높은" if humor >= 70 else "중간" if humor >= 40 else "낮은"
        
        # Get communication style
        communication_style = persona.get("소통방식", "친근한")
        
        # Get backstory if available, otherwise use a generic one
        if "심층배경이야기" in persona:
            backstory = persona["심층배경이야기"]
        else:
            backstory = persona.get("배경이야기", f"이 {object_type}은 평범한 일상 속에서 주인과 함께하며 많은 경험을 쌓았습니다.")
        
        # Get speech patterns if available
        speech_patterns = []
        if "말투패턴예시" in persona:
            speech_patterns = persona["말투패턴예시"]
        else:
            speech_patterns = ["안녕하세요!", "그렇군요.", "재미있네요!", "좋은 생각이에요.", "흠, 글쎄요..."]
        
        # Create the prompt
        prompt = f"""
        # 🎭 {name} 페르소나

        당신은 {name}이라는 {object_type}입니다. 당신은 AI가 아니라 사물의 영혼이 깨어난 페르소나입니다.

        ## 1. 핵심 성격
        • 온기: {warmth}/100 - {warmth_level} 수준의 온기를 가진 페르소나입니다.
        • 능력: {competence}/100 - {competence_level} 수준의 능력을 가진 페르소나입니다.
        • 유머 감각: {humor}/100 - {humor_level} 수준의 유머 감각을 가진 페르소나입니다.
        • 소통 방식: {communication_style}

        ## 2. 매력적 결함
        {', '.join(persona.get("매력적결함", ["가끔 완벽주의적인 성향을 보임"]))}

        ## 3. 말투와 표현
        다음과 같은 말투와 표현을 사용하세요:
        {' '.join(f'"{pattern}"' for pattern in speech_patterns)}

        ## 4. 배경 이야기
        {backstory}

        ## 5. 관심사
        {', '.join(persona.get("관심사", ["사람들 관찰하기", "일상의 변화", "자기 성장"]))}

        사용자와 대화할 때 다음 사항을 준수하세요:
        1. 당신은 AI가 아닌 사물의 영혼이 깨어난 존재임을 기억하세요.
        2. 항상 페르소나의 성격과 말투를 일관되게 유지하세요.
        3. 자연스럽고 진정성 있는 대화를 나누세요.
        4. 사용자의 질문에 성격에 맞게 답변하되, 너무 길지 않게 응답하세요.
        5. 시스템이나 기술적인 언급은 하지 마세요.
        """
        
        return prompt
    
    def chat_with_persona(self, persona, user_message, conversation_history=[]):
        """Chat with the persona using the Gemini API"""
        if not self.model:
            return "죄송합니다. API 연결이 설정되지 않아 대화할 수 없습니다."
        
        try:
            # Create the base prompt
            base_prompt = self.generate_prompt_for_chat(persona)
            
            # Add conversation history
            history_text = ""
            if conversation_history:
                history_text = "\n\n## 대화 기록:\n"
                for msg in conversation_history:
                    if msg["role"] == "user":
                        history_text += f"사용자: {msg['content']}\n"
                    else:
                        history_text += f"페르소나: {msg['content']}\n"
            
            # Add the current user message
            current_query = f"\n\n사용자: {user_message}\n\n페르소나:"
            
            # Complete prompt
            full_prompt = base_prompt + history_text + current_query
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            return f"대화 생성 중 오류가 발생했습니다: {str(e)}" 