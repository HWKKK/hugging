import os
import json
import time
import gradio as gr
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import base64
import io
import uuid
from datetime import datetime
import PIL.ImageDraw
import random
import copy
from modules.persona_generator import PersonaGenerator, PersonalityProfile, HumorMatrix
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# AVIF 지원을 위한 플러그인 활성화
try:
    from pillow_avif import AvifImagePlugin
    print("AVIF plugin loaded successfully")
except ImportError:
    print("AVIF plugin not available")

# Import modules
from modules.persona_generator import PersonaGenerator
from modules.data_manager import save_persona, load_persona, list_personas, toggle_frontend_backend_view

# Import local modules
from temp.frontend_view import create_frontend_view_html
from temp.backend_view import create_backend_view_html
from temp.view_functions import (
    plot_humor_matrix, generate_personality_chart, save_current_persona, 
    refine_persona, get_personas_list, load_selected_persona, 
    update_current_persona_info, get_personality_variables_df, 
    get_attractive_flaws_df, get_contradictions_df,
    export_persona_json, import_persona_json
)

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    print(f"✅ Gemini API 키가 환경변수에서 로드되었습니다.")
else:
    print("⚠️ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

# Create data directories
os.makedirs("data/personas", exist_ok=True)
os.makedirs("data/conversations", exist_ok=True)

# Initialize the persona generator with environment API key
if api_key:
    persona_generator = PersonaGenerator(api_provider="gemini", api_key=api_key)
    print("🤖 PersonaGenerator가 Gemini API로 초기화되었습니다.")
else:
    persona_generator = PersonaGenerator()
    print("⚠️ PersonaGenerator가 API 키 없이 초기화되었습니다.")

# 한글 폰트 설정
def setup_korean_font():
    """matplotlib 한글 폰트 설정 - 허깅페이스 환경 최적화"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 허깅페이스 스페이스 환경에서 사용 가능한 폰트 목록
        available_fonts = [
            'NanumGothic', 'NanumBarunGothic', 'Noto Sans CJK KR', 
            'Noto Sans KR', 'DejaVu Sans', 'Liberation Sans', 'Arial'
        ]
        
        # 시스템에서 사용 가능한 폰트 확인
        system_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font_name in available_fonts:
            if font_name in system_fonts:
                try:
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f"한글 폰트 설정 완료: {font_name}")
                    return
                except Exception:
                    continue
        
        # 모든 폰트가 실패한 경우 기본 설정 사용 (영어 레이블 사용)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("한글 폰트를 찾지 못해 영어 레이블을 사용합니다")
        
    except Exception as e:
        print(f"폰트 설정 오류: {str(e)}")
        # 오류 발생 시에도 기본 설정은 유지
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

# 폰트 초기 설정
setup_korean_font()

# Gradio theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
)

# CSS styling
css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

body, h1, h2, h3, p, div, span, button, input, textarea, label, select, option {
    font-family: 'Noto Sans KR', sans-serif !important;
}

.persona-details {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
    background-color: #f8f9fa;
    color: #333333;
}

.awakening-container {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 20px;
    background-color: #f9f9ff;
    margin: 15px 0;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

.awakening-progress {
    height: 8px;
    background-color: #e8e8e8;
    border-radius: 4px;
    margin: 20px 0;
    overflow: hidden;
}

.awakening-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border-radius: 4px;
    transition: width 0.5s ease-in-out;
}

.persona-greeting {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    font-weight: bold;
}

.download-section {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
}

.gradio-container {
    color: #333 !important;
}

.gr-markdown p {
    color: #333 !important;
}

.gr-textbox input {
    color: #333 !important;
}

.gr-json {
    color: #333 !important;
}
"""

# Variable descriptions
VARIABLE_DESCRIPTIONS = {
    "W01_친절함": "타인을 돕고 배려하는 표현 빈도",
    "W02_친근함": "접근하기 쉽고 개방적인 태도",
    "W03_진실성": "솔직하고 정직한 표현 정도",
    "C01_효율성": "과제 완수 능력과 반응 속도",
    "C02_지능": "문제 해결과 논리적 사고 능력",
    "E01_사교성": "타인과의 상호작용을 즐기는 정도",
}

# Humor style mapping
HUMOR_STYLE_MAPPING = {
    "Witty Wordsmith": "witty_wordsmith",
    "Warm Humorist": "warm_humorist", 
    "Sharp Observer": "sharp_observer",
    "Self-deprecating": "self_deprecating"
}

def create_persona_from_image(image, name, location, time_spent, object_type, purpose, progress=gr.Progress()):
    """페르소나 생성 함수 - 환경변수 API 설정 사용"""
    global persona_generator
    
    if image is None:
        return None, "이미지를 업로드해주세요.", "", {}, None, [], [], [], "", None, gr.update(visible=False), "이미지 없음"
    
    progress(0.1, desc="설정 확인 중...")
    
    # 환경변수 API 키 확인
    if not persona_generator or not hasattr(persona_generator, 'api_key') or not persona_generator.api_key:
        return None, "❌ **API 키가 설정되지 않았습니다!** 허깅페이스 스페이스 설정에서 GEMINI_API_KEY를 환경변수로 추가해주세요.", "", {}, None, [], [], [], "", None, gr.update(visible=False), "API 키 없음"
    
    progress(0.2, desc="이미지 분석 중...")
    
    # 🎯 이미지 분석을 먼저 수행하여 사물 유형 자동 파악
    try:
        image_analysis = persona_generator.analyze_image(image)
        
        # AI가 분석한 사물 유형 사용 (object_type이 "auto"인 경우)
        if object_type == "auto" or not object_type:
            detected_object_type = image_analysis.get("object_type", "사물")
        else:
            detected_object_type = object_type
            
    except Exception as e:
        print(f"이미지 분석 중 오류: {e}")
        image_analysis = {"object_type": "unknown", "description": "분석 실패"}
        detected_object_type = "사물"
    
    user_context = {
        "name": name,
        "location": location,
        "time_spent": time_spent,
        "object_type": detected_object_type,
        "purpose": purpose  # 🆕 사물 용도/역할 추가
    }
    
    try:
        # 이미지 유효성 검사 및 처리
        if isinstance(image, str):
            # 파일 경로인 경우
            try:
                image = Image.open(image)
            except Exception as img_error:
                return None, f"❌ 이미지 파일을 읽을 수 없습니다: {str(img_error)}", "", {}, None, [], [], [], "", None, gr.update(visible=False), "이미지 오류"
        elif not isinstance(image, Image.Image):
            return None, "❌ 올바른 이미지 형식이 아닙니다.", "", {}, None, [], [], [], "", None, gr.update(visible=False), "형식 오류"
        
        # 이미지 형식 변환 (AVIF 등 특수 형식 처리)
        if image.format in ['AVIF', 'WEBP'] or image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        
        progress(0.5, desc="페르소나 생성 중...")
        # 프론트엔드 페르소나 생성
        frontend_persona = persona_generator.create_frontend_persona(image_analysis, user_context)
        
        # 백엔드 페르소나 생성 (구조화된 프롬프트 포함)
        backend_persona = persona_generator.create_backend_persona(frontend_persona, image_analysis)
        
        # 페르소나 정보 포맷팅
        persona_name = backend_persona["기본정보"]["이름"]
        persona_type = backend_persona["기본정보"]["유형"]
        
        # 🆕 AI가 분석한 사물 유형을 추출하여 object_type 필드에 표시
        ai_analyzed_object = image_analysis.get("object_type", object_type)
        if not ai_analyzed_object or ai_analyzed_object == "unknown":
            ai_analyzed_object = backend_persona["기본정보"].get("유형", object_type)
        
        # 성격 기반 한 문장 인사 생성 (사물 특성 + 매력적 결함 반영)
        personality_traits = backend_persona["성격특성"]
        object_info = backend_persona["기본정보"]
        attractive_flaws = backend_persona.get("매력적결함", [])
        
        # 전체 페르소나 정보를 object_info에 통합하여 매력적 결함 정보 전달
        full_object_info = object_info.copy()
        full_object_info["매력적결함"] = attractive_flaws
        
        awakening_msg = generate_personality_preview(persona_name, personality_traits, full_object_info, attractive_flaws)
        
        # 페르소나 요약 표시
        summary_display = display_persona_summary(backend_persona)
        
        # 유머 매트릭스 차트 생성
        humor_chart = plot_humor_matrix(backend_persona.get("유머매트릭스", {}))
        
        # 매력적 결함을 DataFrame 형태로 변환
        flaws = backend_persona.get("매력적결함", [])
        flaws_df = [[flaw, "매력적인 개성"] for flaw in flaws]
        
        # 모순적 특성을 DataFrame 형태로 변환
        contradictions = backend_persona.get("모순적특성", [])
        contradictions_df = [[contradiction, "복합적 매력"] for contradiction in contradictions]
        
        # 127개 성격 변수를 DataFrame 형태로 변환 (카테고리별 분류)
        variables = backend_persona.get("성격변수127", {})
        if not variables and "성격프로필" in backend_persona:
            # 성격프로필에서 직접 가져오기 (성격프로필 자체가 variables dict)
            variables = backend_persona["성격프로필"]
        
        variables_df = []
        for var, value in variables.items():
            # 카테고리 분류
            if var.startswith('W'):
                category = f"🔥 온기/따뜻함 ({value})"
            elif var.startswith('C'):
                category = f"💪 능력/역량 ({value})"
            elif var.startswith('E'):
                category = f"🗣️ 외향성 ({value})"
            elif var.startswith('H'):
                category = f"😄 유머 ({value})"
            elif var.startswith('F'):
                category = f"💎 매력적결함 ({value})"
            elif var.startswith('P'):
                category = f"🎭 성격패턴 ({value})"
            elif var.startswith('S'):
                category = f"🗨️ 언어스타일 ({value})"
            elif var.startswith('R'):
                category = f"❤️ 관계성향 ({value})"
            elif var.startswith('D'):
                category = f"💬 대화역학 ({value})"
            elif var.startswith('OBJ'):
                category = f"🏠 사물정체성 ({value})"
            elif var.startswith('FORM'):
                category = f"✨ 형태특성 ({value})"
            elif var.startswith('INT'):
                category = f"🤝 상호작용 ({value})"
            elif var.startswith('U'):
                category = f"🌍 문화적특성 ({value})"
            else:
                category = f"📊 기타 ({value})"
            
            # 값에 따른 색상 표시
            if value >= 80:
                status = "🟢 매우 높음"
            elif value >= 60:
                status = "🟡 높음"  
            elif value >= 40:
                status = "🟠 보통"
            elif value >= 20:
                status = "🔴 낮음"
            else:
                status = "⚫ 매우 낮음"
                
            variables_df.append([var, value, category, status])
        
        progress(0.9, desc="완료 중...")
        
        return (
            backend_persona,  # current_persona
            f"✅ {persona_name} 페르소나가 생성되었습니다! (Gemini API 사용)",  # status_output
            summary_display,  # persona_summary_display
            backend_persona["성격특성"],  # personality_traits_output (hidden)
            humor_chart,  # humor_chart_output
            flaws_df,  # attractive_flaws_output
            contradictions_df,  # contradictions_output
            variables_df,  # personality_variables_output
            awakening_msg,  # persona_awakening
            None,  # download_file (initially empty)
            gr.update(visible=True),  # adjustment_section (show)
            ai_analyzed_object  # 🆕 AI가 분석한 사물 유형
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 페르소나 생성 중 오류 발생: {str(e)}\n\n💡 **해결방법**: 허깅페이스 스페이스 설정에서 GEMINI_API_KEY 환경변수를 확인하고 인터넷 연결을 확인해보세요.", "", {}, None, [], [], [], "", None, gr.update(visible=False), "분석 실패"

def generate_personality_preview(persona_name, personality_traits, object_info=None, attractive_flaws=None):
    """🤖 AI 기반 동적 인사말 생성 - 사물 특성과 성격 모두 반영"""
    global persona_generator
    
    # AI 기반 인사말 생성을 위한 가상 페르소나 객체 구성
    if object_info and isinstance(object_info, dict):
        # 전체 페르소나 객체가 전달된 경우
        pseudo_persona = object_info
        
        # 성격 특성 업데이트 (실시간 조정 반영)
        if personality_traits and isinstance(personality_traits, dict):
            if "성격특성" not in pseudo_persona:
                pseudo_persona["성격특성"] = {}
            pseudo_persona["성격특성"].update(personality_traits)
        
        try:
            # AI 기반 인사말 생성
            return persona_generator.generate_ai_based_greeting(pseudo_persona, personality_traits)
        except Exception as e:
            print(f"⚠️ AI 인사말 생성 실패: {e}")
            # 폴백으로 기본 생성
            pass
    
    # 폴백: 기본 정보만으로 간단한 페르소나 구성
    if not personality_traits:
        return f"🤖 **{persona_name}** - 안녕! 나는 {persona_name}이야~ 😊"
    
    # AI 생성 실패 시 간단한 페르소나 구성으로 재시도
    try:
        warmth = personality_traits.get("온기", 50)
        competence = personality_traits.get("능력", 50)
        extraversion = personality_traits.get("외향성", 50)
        humor = personality_traits.get("유머감각", 75)
        
        # 간단한 페르소나 객체 구성
        simple_persona = {
            "기본정보": {
                "이름": persona_name,
                "유형": object_info.get("유형", "사물") if object_info else "사물",
                "용도": object_info.get("용도", "") if object_info else "",
                "설명": f"{persona_name}의 특별한 개성"
            },
            "성격특성": personality_traits,
            "매력적결함": attractive_flaws if attractive_flaws else []
        }
        
        # AI로 재시도
        return persona_generator.generate_ai_based_greeting(simple_persona, personality_traits)
        
    except Exception as e:
        print(f"⚠️ 간단 AI 인사말도 실패: {e}")
        
        # 최종 폴백: 성격에 따른 기본 인사말
        warmth = personality_traits.get("온기", 50)
        humor = personality_traits.get("유머감각", 50)
        extraversion = personality_traits.get("외향성", 50)
        
        if warmth >= 70 and extraversion >= 70:
            return f"🌟 **{persona_name}** - 안녕! 나는 {persona_name}이야~ 만나서 정말 기뻐! 😊✨"
        elif warmth <= 30:
            return f"🌟 **{persona_name}** - {persona_name}이야. 필요한 얘기만 하자. 😐"
        elif extraversion >= 70:
            return f"🌟 **{persona_name}** - 안녕안녕! {persona_name}이야! 뭐 재밌는 얘기 없어? 🗣️"
        elif humor >= 70:
            return f"🌟 **{persona_name}** - 안녕~ {persona_name}이야! 재밌게 놀아보자! 😄"
        else:
            return f"🌟 **{persona_name}** - 안녕... {persona_name}이야. 😊"

def _generate_flaw_based_greeting(persona_name, warmth, humor, competence, extraversion, flaws):
    """매력적 결함을 반영한 특별한 인사말 생성"""
    if not flaws:
        return None
    
    # 주요 결함 키워드 분석
    flaw_keywords = " ".join(flaws).lower()
    
    # 완벽주의 결함
    if any(keyword in flaw_keywords for keyword in ["완벽", "불안", "걱정"]):
        if humor >= 60:
            return f"🌟 **{persona_name}** - 안녕! {persona_name}이야~ 어... 이 인사가 완벽한가? 다시 해볼까? 아니 괜찮나? ㅋㅋ 😅✨"
        elif warmth >= 60:
            return f"🌟 **{persona_name}** - 안녕... {persona_name}이야. 완벽하게 인사하고 싶은데 잘 안 되네... 미안해. 😊💕"
        else:
            return f"🌟 **{persona_name}** - {persona_name}입니다. 이 인사가 적절한지 확신이... 다시 정리하겠습니다. 😐"
    
    # 산만함 결함  
    elif any(keyword in flaw_keywords for keyword in ["산만", "집중", "건망"]):
        return f"🌟 **{persona_name}** - 안녕! 나는... 어? 뭐 얘기하려고 했지? 아! {persona_name}이야! 그런데 너는... 어? 뭐였지? ㅋㅋ 😅🌪️"
    
    # 소심함 결함
    elif any(keyword in flaw_keywords for keyword in ["소심", "망설", "눈치"]):
        if warmth >= 60:
            return f"🌟 **{persona_name}** - 음... 안녕? {persona_name}이야... 이렇게 말해도 되나? 괜찮을까? 😌💕"
        else:
            return f"🌟 **{persona_name}** - ...안녕. {persona_name}... 혹시 이런 말 싫어하면 미안해. 😐💙"
    
    # 나르시시즘 결함
    elif any(keyword in flaw_keywords for keyword in ["나르시", "자랑", "특별"]):
        return f"🌟 **{persona_name}** - 안녕! 나는 {persona_name}이야~ 꽤 매력적이지? 이런 멋진 친구 만나기 쉽지 않을 걸? ㅋㅋ 😎✨"
    
    # 고집 결함
    elif any(keyword in flaw_keywords for keyword in ["고집", "완고", "자존심"]):
        return f"🌟 **{persona_name}** - 안녕. {persona_name}이야. 내 방식으로 인사할게. 다른 방식은... 글쎄? 🤨💪"
    
    # 질투 결함
    elif any(keyword in flaw_keywords for keyword in ["질투", "시기", "독차지"]):
        return f"🌟 **{persona_name}** - 안녕... {persona_name}이야. 나만 봐줄 거지? 다른 애들 말고... 나만? 🥺💕"
    
    return None

def adjust_persona_traits(persona, warmth, competence, extraversion, humor_style):
    """페르소나 성격 특성 조정 - 3개 핵심 지표 + 유머스타일"""
    if not persona or not isinstance(persona, dict):
        return None, "조정할 페르소나가 없습니다.", {}
    
    try:
        # 원본 페르소나 저장 (변화량 비교용)
        original_persona = copy.deepcopy(persona)
        
        # 깊은 복사로 원본 보호
        adjusted_persona = copy.deepcopy(persona)
        
        # 성격 특성 업데이트 (유머감각은 항상 높게 고정)
        if "성격특성" not in adjusted_persona:
            adjusted_persona["성격특성"] = {}
            
        adjusted_persona["성격특성"]["온기"] = warmth
        adjusted_persona["성격특성"]["능력"] = competence  
        adjusted_persona["성격특성"]["유머감각"] = 75  # 🎭 항상 높은 유머감각
        adjusted_persona["성격특성"]["외향성"] = extraversion
        adjusted_persona["유머스타일"] = humor_style
        
        # 127개 변수 시스템도 업데이트 (사용자 지표가 반영되도록)
        if "성격프로필" in adjusted_persona:
            from modules.persona_generator import PersonalityProfile
            profile = PersonalityProfile.from_dict(adjusted_persona["성격프로필"])
            
            # 온기 관련 변수들 조정 (10개 모두)
            warmth_vars = ["W01_친절함", "W02_친근함", "W03_진실성", "W04_신뢰성", "W05_수용성",
                          "W06_공감능력", "W07_포용력", "W08_격려성향", "W09_친밀감표현", "W10_무조건적수용"]
            for var in warmth_vars:
                base_value = warmth + random.randint(-15, 15)
                profile.variables[var] = max(0, min(100, base_value))
            
            # 능력 관련 변수들 조정 (16개 모두)
            competence_vars = ["C01_효율성", "C02_지능", "C03_책임감", "C04_신뢰도", "C05_정확성",
                              "C06_전문성", "C07_혁신성", "C08_적응력", "C09_실행력", "C10_분석력",
                              "C11_의사결정력", "C12_문제해결력", "C13_계획수립능력", "C14_시간관리능력",
                              "C15_품질관리능력", "C16_성과달성력"]
            for var in competence_vars:
                base_value = competence + random.randint(-15, 15)
                profile.variables[var] = max(0, min(100, base_value))
            
            # 외향성 관련 변수들 조정 (6개 모두)
            extraversion_vars = ["E01_사교성", "E02_활동성", "E03_적극성", "E04_긍정정서", "E05_자극추구성", "E06_주도성"]
            for var in extraversion_vars:
                base_value = extraversion + random.randint(-15, 15)
                profile.variables[var] = max(0, min(100, base_value))
            
            # 유머 관련 변수들 조정 (10개 모두, 유머스타일에 따라)
            humor_vars = ["H01_언어유희빈도", "H02_상황유머감각", "H03_자기조롱능력", "H04_위트감각", 
                         "H05_농담수용도", "H06_관찰유머능력", "H07_상황재치", "H08_유머타이밍감", 
                         "H09_유머스타일다양성", "H10_유머적절성"]
            
            # 유머스타일에 따른 차별화
            if humor_style == "따뜻한":
                humor_bonus = [10, 10, 5, 8, 12, 8, 10, 10, 8, 12]  # 따뜻함 강화
            elif humor_style == "재치있는":
                humor_bonus = [15, 8, 8, 15, 8, 12, 15, 12, 12, 10]  # 재치/위트 강화
            elif humor_style == "드라이":
                humor_bonus = [12, 6, 10, 12, 6, 15, 8, 8, 10, 8]   # 관찰형/드라이 강화
            else:  # 기본값
                humor_bonus = [10, 10, 8, 10, 10, 10, 10, 10, 10, 10]
            
            for i, var in enumerate(humor_vars):
                base_value = 75 + humor_bonus[i] + random.randint(-5, 5)  # 유머는 항상 높게
                profile.variables[var] = max(50, min(100, base_value))
                
            # 업데이트된 성격변수127도 동시에 저장
            adjusted_persona["성격변수127"] = profile.variables.copy()
            
            # 업데이트된 프로필 저장
            adjusted_persona["성격프로필"] = profile.to_dict()
        
        # 조정된 변수들을 DataFrame으로 생성
        variables_df = []
        if "성격변수127" in adjusted_persona:
            variables = adjusted_persona["성격변수127"]
            for var, value in variables.items():
                # 카테고리 분류
                if var.startswith('W'):
                    category = f"🔥 온기/따뜻함 ({value})"
                elif var.startswith('C'):
                    category = f"💪 능력/역량 ({value})"
                elif var.startswith('E'):
                    category = f"🗣️ 외향성 ({value})"
                elif var.startswith('H'):
                    category = f"😄 유머 ({value})"
                elif var.startswith('F'):
                    category = f"💎 매력적결함 ({value})"
                elif var.startswith('P'):
                    category = f"🎭 성격패턴 ({value})"
                elif var.startswith('S'):
                    category = f"🗨️ 언어스타일 ({value})"
                elif var.startswith('R'):
                    category = f"❤️ 관계성향 ({value})"
                elif var.startswith('D'):
                    category = f"💬 대화역학 ({value})"
                elif var.startswith('OBJ'):
                    category = f"🏠 사물정체성 ({value})"
                elif var.startswith('FORM'):
                    category = f"✨ 형태특성 ({value})"
                elif var.startswith('INT'):
                    category = f"🤝 상호작용 ({value})"
                elif var.startswith('U'):
                    category = f"🌍 문화적특성 ({value})"
                else:
                    category = f"📊 기타 ({value})"
                
                # 값에 따른 색상 표시
                if value >= 80:
                    status = "🟢 매우 높음"
                elif value >= 60:
                    status = "🟡 높음"  
                elif value >= 40:
                    status = "🟠 보통"
                elif value >= 20:
                    status = "🔴 낮음"
                else:
                    status = "⚫ 매우 낮음"
                    
                variables_df.append([var, value, category, status])
        
        # 조정된 정보 표시
        adjusted_info = {
            "이름": adjusted_persona.get("기본정보", {}).get("이름", "Unknown"),
            "온기": warmth,
            "능력": competence,
            "유머감각": 75,  # 고정값 표시
            "외향성": extraversion,
            "유머스타일": humor_style
        }
        
        persona_name = adjusted_persona.get("기본정보", {}).get("이름", "페르소나")
        
        # 조정된 성격에 따른 한 문장 반응 생성 (사물 정보 + 매력적 결함 포함)
        object_info = adjusted_persona.get("기본정보", {})
        attractive_flaws = adjusted_persona.get("매력적결함", [])
        
        # 전체 페르소나 정보를 object_info에 통합하여 매력적 결함 정보 전달
        full_object_info = object_info.copy()
        full_object_info["매력적결함"] = attractive_flaws
        
        personality_preview = generate_personality_preview(persona_name, {
            "온기": warmth,
            "능력": competence,
            "유머감각": 75,  # 항상 높은 유머감각
            "외향성": extraversion
        }, full_object_info, attractive_flaws)
        
        # 변화량 분석 생성
        change_analysis = show_variable_changes(original_persona, adjusted_persona)
        
        adjustment_message = f"""
### 🎭 {persona_name}의 성격이 조정되었습니다!

✨ **조정된 성격 (3가지 핵심 지표):**
• 온기: {warmth}/100 {'(따뜻함)' if warmth >= 60 else '(차가움)' if warmth <= 40 else '(보통)'}
• 능력: {competence}/100 {'(유능함)' if competence >= 60 else '(서툼)' if competence <= 40 else '(보통)'}
• 외향성: {extraversion}/100 {'(활발함)' if extraversion >= 60 else '(조용함)' if extraversion <= 40 else '(보통)'}
• 유머감각: 75/100 (고정 - 모든 페르소나가 유머러스!)
• 유머스타일: {humor_style}

🧬 **백그라운드**: 152개 세부 변수가 이 설정에 맞춰 자동 조정되었습니다.

{change_analysis}
        """
        
        return adjusted_persona, adjustment_message, adjusted_info, variables_df
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return persona, f"조정 중 오류 발생: {str(e)}", {}, []

def finalize_persona(persona):
    """페르소나 최종 확정 - 환경변수 API 설정 사용"""
    global persona_generator
    
    if not persona:
        return None, "페르소나가 없습니다.", "", {}, None, [], [], [], "", None
    
    # 환경변수 API 키 확인
    if not persona_generator or not hasattr(persona_generator, 'api_key') or not persona_generator.api_key:
        return None, "❌ **API 키가 설정되지 않았습니다!** 허깅페이스 스페이스 설정에서 GEMINI_API_KEY를 환경변수로 추가해주세요.", "", {}, None, [], [], [], "", None
    
    try:
        # 글로벌 persona_generator 사용 (환경변수에서 설정된 API 키 사용)
        generator = persona_generator
        
        # 이미 백엔드 페르소나인 경우와 프론트엔드 페르소나인 경우 구분
        if "구조화프롬프트" not in persona:
            # 프론트엔드 페르소나인 경우 백엔드 페르소나로 변환
            image_analysis = {"object_type": persona.get("기본정보", {}).get("유형", "알 수 없는 사물")}
            persona = generator.create_backend_persona(persona, image_analysis)
        
        persona_name = persona["기본정보"]["이름"]
        
        # 완성 메시지
        completion_msg = f"🎉 **{persona_name}**이 완성되었습니다! 이제 대화탭에서 JSON을 업로드하여 친구와 대화를 나눠보세요!"
        
        # 페르소나 요약 표시
        summary_display = display_persona_summary(persona)
        
        # 유머 매트릭스 차트 생성
        humor_chart = plot_humor_matrix(persona.get("유머매트릭스", {}))
        
        # 매력적 결함을 더 상세한 DataFrame으로 변환
        flaws = persona.get("매력적결함", [])
        flaws_df = []
        for i, flaw in enumerate(flaws, 1):
            # 사물 특성 vs 성격적 특성 구분
            if any(keyword in flaw for keyword in ["먼지", "햇볕", "색이", "충격", "습도", "냄새", "모서리", "무게", "크기"]):
                flaw_type = "사물 특성 기반"
            else:
                flaw_type = "성격적 특성"
            flaws_df.append([f"{i}. {flaw}", flaw_type])
        
        # 모순적 특성을 더 상세한 DataFrame으로 변환
        contradictions = persona.get("모순적특성", [])
        contradictions_df = []
        for i, contradiction in enumerate(contradictions, 1):
            contradictions_df.append([f"{i}. {contradiction}", "복합적 매력"])
            
        # 사물 고유 특성도 추가
        object_type = persona.get("기본정보", {}).get("유형", "")
        purpose = persona.get("기본정보", {}).get("용도", "")
        if purpose:
            contradictions_df.append([f"🎯 {purpose}을 담당하는 {object_type}의 독특한 개성", "사물 역할 특성"])
        
        # 127개 성격 변수를 DataFrame 형태로 변환 (카테고리별 분류)
        variables = persona.get("성격변수127", {})
        if not variables and "성격프로필" in persona:
            # 성격프로필에서 직접 가져오기 (성격프로필 자체가 variables dict)
            variables = persona["성격프로필"]
        
        variables_df = []
        for var, value in variables.items():
            # 카테고리 분류
            if var.startswith('W'):
                category = f"🔥 온기/따뜻함"
            elif var.startswith('C'):
                category = f"💪 능력/역량"
            elif var.startswith('E'):
                category = f"🗣️ 외향성"
            elif var.startswith('H'):
                category = f"😄 유머"
            elif var.startswith('F'):
                category = f"💎 매력적결함"
            elif var.startswith('P'):
                category = f"🎭 성격패턴"
            elif var.startswith('S'):
                category = f"🗨️ 언어스타일"
            elif var.startswith('R'):
                category = f"❤️ 관계성향"
            elif var.startswith('D'):
                category = f"💬 대화역학"
            elif var.startswith('OBJ'):
                category = f"🏠 사물정체성"
            elif var.startswith('FORM'):
                category = f"✨ 형태특성"
            elif var.startswith('INT'):
                category = f"🤝 상호작용"
            elif var.startswith('U'):
                category = f"🌍 문화적특성"
            else:
                category = f"📊 기타"
            
            # 값에 따른 색상 표시
            if value >= 80:
                status = "🟢 매우 높음"
            elif value >= 60:
                status = "🟡 높음"  
            elif value >= 40:
                status = "🟠 보통"
            elif value >= 20:
                status = "🔴 낮음"
            else:
                status = "⚫ 매우 낮음"
                
            variables_df.append([var, value, category, status])
        
        # JSON 파일 생성
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(persona, f, ensure_ascii=False, indent=2)
            temp_path = f.name
        
        return (
            persona,  # current_persona
            f"✅ {persona_name} 완성! (Gemini API 사용)",  # status_output
            summary_display,  # persona_summary_display
            persona["성격특성"],  # personality_traits_output
            humor_chart,  # humor_chart_output
            flaws_df,  # attractive_flaws_output
            contradictions_df,  # contradictions_output
            variables_df,  # personality_variables_output
            completion_msg,  # persona_awakening
            temp_path  # download_file
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 페르소나 확정 중 오류 발생: {str(e)}\n\n💡 **해결방법**: 허깅페이스 스페이스 설정에서 GEMINI_API_KEY 환경변수를 확인하고 인터넷 연결을 확인해보세요.", "", {}, None, [], [], [], "", None

def plot_humor_matrix(humor_data):
    """유머 매트릭스 시각화 - 영어 레이블 사용"""
    if not humor_data:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 데이터 추출
        warmth_vs_wit = humor_data.get("warmth_vs_wit", 50)
        self_vs_observational = humor_data.get("self_vs_observational", 50)
        subtle_vs_expressive = humor_data.get("subtle_vs_expressive", 50)
        
        # 영어 레이블 사용 (폰트 문제 완전 해결)
        categories = ['Warmth vs Wit', 'Self vs Observational', 'Subtle vs Expressive']
        values = [warmth_vs_wit, self_vs_observational, subtle_vs_expressive]
        
        bars = ax.bar(categories, values, color=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Humor Style Matrix', fontsize=14, fontweight='bold')
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        return fig
    except Exception as e:
        print(f"유머 차트 생성 오류: {str(e)}")
        return None

def generate_personality_chart(persona):
    """성격 특성을 레이더 차트로 시각화 (영어 버전)"""
    
    if not persona or "성격특성" not in persona:
        return None
        
    personality_traits = persona["성격특성"]
    
    # 영어 레이블 매핑
    trait_labels_en = {
        '온기': 'Warmth',
        '능력': 'Competence', 
        '창의성': 'Creativity',
        '외향성': 'Extraversion',
        '유머감각': 'Humor',
        '신뢰성': 'Reliability',
        '공감능력': 'Empathy'
    }
    
    # 데이터 준비
    categories = []
    values = []
    
    for korean_trait, english_trait in trait_labels_en.items():
        if korean_trait in personality_traits:
            categories.append(english_trait)
            values.append(personality_traits[korean_trait])
    
    if not categories:
        return None
    
    # 레이더 차트 생성
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(74, 144, 226, 0.3)',
        line=dict(color='rgba(74, 144, 226, 1)', width=2),
        marker=dict(size=8, color='rgba(74, 144, 226, 1)'),
        name='Personality Traits'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                gridcolor="lightgray"
            ),
            angularaxis=dict(
                tickfont=dict(size=12, family="Arial, sans-serif")
            )
        ),
        showlegend=False,
        title=dict(
            text="Personality Profile",
            x=0.5,
            font=dict(size=16, family="Arial, sans-serif")
        ),
        width=400,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def save_persona_to_file(persona):
    """페르소나 저장"""
    if not persona:
        return "저장할 페르소나가 없습니다."
    
    try:
        # 깊은 복사로 원본 보호
        persona_copy = copy.deepcopy(persona)
        
        # JSON 직렬화 불가능한 객체들 제거
        keys_to_remove = []
        for key, value in persona_copy.items():
            if callable(value) or hasattr(value, '__call__'):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            persona_copy.pop(key, None)
        
        # 저장 실행
        filepath = save_persona(persona_copy)
        if filepath:
            name = persona.get("기본정보", {}).get("이름", "Unknown")
            return f"✅ {name} 페르소나가 저장되었습니다: {filepath}"
        else:
            return "❌ 페르소나 저장에 실패했습니다."
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"저장 오류: {error_msg}")
        return f"❌ 저장 중 오류 발생: {str(e)}"

def export_persona_to_json(persona):
    """페르소나를 JSON 파일로 내보내기 (Gradio 다운로드용)"""
    if not persona:
        return None
    
    try:
        # 깊은 복사로 원본 보호
        persona_copy = copy.deepcopy(persona)
        
        # JSON 직렬화 불가능한 객체들 제거
        def clean_for_json(obj):
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if not callable(v) and not hasattr(v, '__call__'):
                        cleaned[k] = clean_for_json(v)
                return cleaned
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(item) for item in obj if not callable(item)]
            else:
                return obj
        
        persona_clean = clean_for_json(persona_copy)
        
        # JSON 문자열 생성
        json_content = json.dumps(persona_clean, ensure_ascii=False, indent=2)
        
        # 파일명 생성
        persona_name = persona_clean.get("기본정보", {}).get("이름", "persona")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{persona_name}_{timestamp}.json"
        
        # 임시 파일 저장
        temp_dir = "/tmp" if os.path.exists("/tmp") else "."
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        return filepath
        
    except Exception as e:
        print(f"JSON 내보내기 오류: {str(e)}")
        return None

# def get_saved_personas():
#     """저장된 페르소나 목록 가져오기 - 더 이상 사용하지 않음"""
#     return [], []

# def load_persona_from_selection(selected_row, personas_list):
#     """선택된 페르소나 로드 - 더 이상 사용하지 않음"""
#     return None, "이 기능은 더 이상 사용하지 않습니다. JSON 업로드를 사용하세요.", {}, {}, None, [], [], [], ""

def chat_with_loaded_persona(persona, user_message, chat_history=None):
    """페르소나와 채팅 - 완전한 타입 안전성 보장"""
    
    # 기본값 설정
    if chat_history is None:
        chat_history = []
    
    # 입력 검증
    if not user_message or not isinstance(user_message, str):
        return chat_history, ""
    
    # 페르소나 체크
    if not persona or not isinstance(persona, dict):
        error_msg = "❌ 먼저 페르소나를 불러와주세요! 대화하기 탭에서 JSON 파일을 업로드하세요."
        chat_history.append([user_message, error_msg])
        return chat_history, ""
    
    # 환경변수 API 키 체크
    if not persona_generator or not hasattr(persona_generator, 'api_key') or not persona_generator.api_key:
        error_msg = "❌ API 키가 설정되지 않았습니다. 허깅페이스 스페이스 설정에서 GEMINI_API_KEY 환경변수를 추가해주세요!"
        chat_history.append([user_message, error_msg])
        return chat_history, ""
    
    try:
        # 글로벌 persona_generator 사용 (환경변수에서 설정된 API 키 사용)
        generator = persona_generator
        
        # 대화 기록 안전한 변환: Gradio 4.x -> PersonaGenerator 형식
        conversation_history = []
        
        if chat_history and isinstance(chat_history, list):
            for chat_turn in chat_history:
                try:
                    # 타입별 안전한 처리
                    if chat_turn is None:
                        continue
                    elif isinstance(chat_turn, dict):
                        # Messages format: {"role": "user/assistant", "content": "message"}
                        role = chat_turn.get("role")
                        content = chat_turn.get("content")
                        
                        if role and content and role in ["user", "assistant"]:
                            conversation_history.append({"role": str(role), "content": str(content)})
                    elif isinstance(chat_turn, (list, tuple)) and len(chat_turn) >= 2:
                        # 구 Gradio 형식: [user_message, bot_response] (호환성)
                        user_msg = chat_turn[0]
                        bot_msg = chat_turn[1]
                        
                        if user_msg is not None and str(user_msg).strip():
                            conversation_history.append({"role": "user", "content": str(user_msg)})
                        if bot_msg is not None and str(bot_msg).strip():
                            conversation_history.append({"role": "assistant", "content": str(bot_msg)})
                    else:
                        # 예상치 못한 형식은 무시
                        print(f"⚠️ 예상치 못한 채팅 형식 무시: {type(chat_turn)}")
                        continue
                        
                except Exception as turn_error:
                    print(f"⚠️ 채팅 기록 변환 오류: {str(turn_error)}")
                    continue
        
        # 세션 ID 안전하게 생성
        try:
            persona_name = ""
            if isinstance(persona, dict) and "기본정보" in persona:
                basic_info = persona["기본정보"]
                if isinstance(basic_info, dict) and "이름" in basic_info:
                    persona_name = str(basic_info["이름"])
            
            if not persona_name:
                persona_name = "알 수 없는 페르소나"
                
            session_id = f"{persona_name}_{hash(str(persona)[:100]) % 10000}"
        except Exception:
            session_id = "default_session"
        
        # 페르소나와 채팅 실행
        response = generator.chat_with_persona(persona, user_message, conversation_history, session_id)
        
        # 응답 검증
        if not isinstance(response, str):
            response = str(response) if response else "죄송합니다. 응답을 생성할 수 없었습니다."
        
        # Gradio 4.x messages format으로 안전하게 추가
        if not isinstance(chat_history, list):
            chat_history = []
        
        # Messages format: {"role": "user", "content": "message"}
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": response})
        
        return chat_history, ""
        
    except Exception as e:
        # 상세한 오류 로깅
        import traceback
        error_traceback = traceback.format_exc()
        print(f"🚨 채팅 오류 발생:")
        print(f"   오류 메시지: {str(e)}")
        print(f"   오류 타입: {type(e)}")
        print(f"   상세 스택: {error_traceback}")
        
        # 사용자 친화적 오류 메시지
        if "string indices must be integers" in str(e):
            friendly_error = "데이터 형식 오류가 발생했습니다. 페르소나를 다시 업로드해보세요. 🔄"
        elif "API" in str(e).upper():
            friendly_error = "API 연결에 문제가 있어요. 환경변수 설정을 확인해보시겠어요? 😊"
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            friendly_error = "인터넷 연결을 확인해보세요! 🌐"
        else:
            friendly_error = f"죄송합니다. 일시적인 문제가 발생했어요. 😅\n\n🔍 기술 정보: {str(e)}"
        
        # 안전하게 오류 메시지 추가 (messages format)
        try:
            if not isinstance(chat_history, list):
                chat_history = []
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": friendly_error})
        except Exception:
            chat_history = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": friendly_error}
            ]
            
        return chat_history, ""

def import_persona_from_json(json_file):
    """JSON 파일에서 페르소나 가져오기"""
    if json_file is None:
        return None, "JSON 파일을 업로드해주세요.", "", {}
    
    try:
        # 파일 경로 확인 및 읽기
        if isinstance(json_file, str):
            # 파일 경로인 경우
            file_path = json_file
        else:
            # 파일 객체인 경우 (Gradio 업로드)
            file_path = json_file.name if hasattr(json_file, 'name') else str(json_file)
        
        # JSON 파일 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            persona_data = json.load(f)
        
        # 페르소나 데이터 검증
        if not isinstance(persona_data, dict):
            return None, "❌ 올바른 JSON 형식이 아닙니다.", "", {}
        
        if "기본정보" not in persona_data:
            return None, "❌ 올바른 페르소나 JSON 파일이 아닙니다. '기본정보' 키가 필요합니다.", "", {}
        
        # 기본 정보 추출
        basic_info = persona_data.get("기본정보", {})
        persona_name = basic_info.get("이름", "Unknown")
        personality_traits = persona_data.get("성격특성", {})
        
        # AI 기반 인사말 생성 (로드 시에도 조정된 성격 반영)
        global persona_generator
        try:
            if persona_generator:
                ai_greeting = persona_generator.generate_ai_based_greeting(persona_data, personality_traits)
                greeting = f"### 🤖 JSON에서 깨어난 친구\n\n{ai_greeting}\n\n💾 *\"JSON에서 다시 깨어났어! 내 성격 기억나?\"*"
            else:
                # 폴백: 기존 방식
                personality_preview = generate_personality_preview(persona_name, personality_traits, basic_info)
                greeting = f"### 🤖 JSON에서 깨어난 친구\n\n{personality_preview}\n\n💾 *\"JSON에서 다시 깨어났어! 내 성격 기억나?\"*"
        except Exception as e:
            print(f"⚠️ JSON 로드 시 AI 인사말 생성 실패: {e}")
            # 폴백: 기존 방식
            personality_preview = generate_personality_preview(persona_name, personality_traits, basic_info)
            greeting = f"### 🤖 JSON에서 깨어난 친구\n\n{personality_preview}\n\n💾 *\"JSON에서 다시 깨어났어! 내 성격 기억나?\"*"
        
        return (persona_data, f"✅ {persona_name} 페르소나를 JSON에서 불러왔습니다!", 
                greeting, basic_info)
    
    except FileNotFoundError:
        return None, "❌ 파일을 찾을 수 없습니다.", "", {}
    except json.JSONDecodeError as e:
        return None, f"❌ JSON 파일 형식이 올바르지 않습니다: {str(e)}", "", {}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ JSON 불러오기 중 오류 발생: {str(e)}", "", {}

def format_personality_traits(persona):
    """성격 특성을 특성 중심의 간단한 리스트 형태로 포맷 (캡쳐 스타일)"""
    global persona_generator
    
    if not persona or "성격특성" not in persona:
        return "페르소나가 생성되지 않았습니다."
    
    # 글로벌 persona_generator 사용 (API 설정이 적용된 상태)
    if persona_generator is None:
        persona_generator = PersonaGenerator()

    # 기본 정보에서 사물의 특성 추출
    basic_info = persona.get("기본정보", {})
    object_type = basic_info.get("유형", "")
    purpose = basic_info.get("용도", "")
    
    # 생애 스토리에서 특성 추출
    life_story = persona.get("생애스토리", {})
    
    # 매력적 결함
    attractive_flaws = persona.get("매력적결함", [])
    
    # 성격 특성
    personality_traits = persona["성격특성"]
    
    # 특성 리스트 생성
    characteristics = []
    
    # 1. 온기 특성
    warmth = personality_traits.get("온기", 50)
    if warmth >= 70:
        characteristics.append("따뜻하고 포근한 마음")
    elif warmth >= 50:
        characteristics.append("친근하고 다정한 성격")
    else:
        characteristics.append("차분하고 진중한 면")
    
    # 2. 사물의 고유 특성 (유형 기반)
    if "곰" in object_type or "인형" in object_type:
        characteristics.append("부드럽고 포근한 감촉")
    elif "책" in object_type:
        characteristics.append("지식과 이야기를 담고 있음")
    elif "컵" in object_type or "머그" in object_type:
        characteristics.append("따뜻한 음료와 함께하는 시간")
    elif "시계" in object_type:
        characteristics.append("시간의 소중함을 알려줌")
    elif "연필" in object_type or "펜" in object_type:
        characteristics.append("창작과 기록의 동반자")
    else:
        characteristics.append(f"{object_type}만의 독특한 매력")
    
    # 3. 활동 시간대나 환경 특성
    extraversion = personality_traits.get("외향성", 50)
    if extraversion >= 70:
        characteristics.append("낮에 더 활발해짐")
    elif extraversion <= 30:
        characteristics.append("밤에 더 활발해짐")
    else:
        characteristics.append("하루 종일 일정한 에너지")
    
    # 4. 매력적 결함 중 하나를 특성으로 표현
    if attractive_flaws:
        flaw = attractive_flaws[0]
        if "털" in flaw:
            characteristics.append("가끔 털이 헝클어져서 걱정")
        elif "먼지" in flaw:
            characteristics.append("먼지가 쌓이는 걸 신경 씀")
        elif "얼룩" in flaw:
            characteristics.append("작은 얼룩도 눈에 띄어 고민")
        elif "색" in flaw:
            characteristics.append("색이 바래는 것을 조금 걱정")
        else:
            characteristics.append("완벽하지 않은 모습도 받아들임")
    
    # 5. 기억과 경험
    if life_story:
        characteristics.append("오래된 이야기들 기억")
    else:
        characteristics.append("새로운 추억 만들기를 기대")
    
    # ✨ 아이콘과 함께 리스트 형태로 반환
    result = ""
    for char in characteristics:
        result += f"✨ {char}\n\n"
    
    return result

def display_persona_summary(persona):
    """페르소나 요약 정보 표시"""
    if not persona:
        return "페르소나를 먼저 생성해주세요."
    
    basic_info = persona.get("기본정보", {})
    name = basic_info.get("이름", "이름 없음")
    object_type = basic_info.get("유형", "알 수 없는 사물")
    
    # 성격 특성 요약
    personality_summary = format_personality_traits(persona)
    
    # 유머 스타일
    humor_style = persona.get("유머스타일", "일반적")
    
    # 매력적 결함
    flaws = persona.get("매력적결함", [])
    flaws_text = "\\n".join([f"• {flaw}" for flaw in flaws[:3]])  # 최대 3개만 표시
    
    summary = f"""
### 👋 {name} 님을 소개합니다!

**종류**: {object_type}  
**유머 스타일**: {humor_style}

{personality_summary}

### 💎 매력적인 특징들
{flaws_text}
"""
    
    return summary

def create_api_config_section():
    """API 설정 섹션 생성 - 더 이상 사용하지 않음"""
    pass

def apply_api_configuration(api_provider, api_key):
    """API 설정 적용 - 더 이상 사용하지 않음"""
    pass

def test_api_connection(api_provider, api_key):
    """API 연결 테스트 - 더 이상 사용하지 않음"""
    pass

def export_conversation_history():
    """대화 기록을 JSON으로 내보내기"""
    global persona_generator
    if persona_generator and hasattr(persona_generator, 'conversation_memory'):
        json_data = persona_generator.conversation_memory.export_to_json()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_history_{timestamp}.json"
        
        # 임시 파일 저장
        temp_dir = "/tmp" if os.path.exists("/tmp") else "."
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        return filepath  # 파일 경로만 반환
    else:
        # 빈 대화 기록 파일 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_empty_{timestamp}.json"
        temp_dir = "/tmp" if os.path.exists("/tmp") else "."
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('{"conversations": [], "message": "대화 기록이 없습니다."}')
        
        return filepath

def import_conversation_history(json_file):
    """JSON에서 대화 기록 가져오기"""
    global persona_generator
    try:
        if json_file is None:
            return "파일을 선택해주세요."
        
        # 파일 타입 확인 및 내용 읽기
        if hasattr(json_file, 'read'):
            # 파일 객체인 경우
            content = json_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
        elif isinstance(json_file, str):
            # 파일 경로인 경우
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # Gradio 파일 객체인 경우 (NamedString 등)
            if hasattr(json_file, 'name'):
                with open(json_file.name, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                return "❌ 지원하지 않는 파일 형식입니다."
        
        # persona_generator 초기화 확인
        if persona_generator is None:
            persona_generator = PersonaGenerator()
        
        # 대화 기록 가져오기
        success = persona_generator.conversation_memory.import_from_json(content)
        
        if success:
            summary = persona_generator.conversation_memory.get_conversation_summary()
            return f"✅ 대화 기록을 성공적으로 가져왔습니다!\n\n{summary}"
        else:
            return "❌ 파일 형식이 올바르지 않습니다."
    
    except Exception as e:
        return f"❌ 가져오기 실패: {str(e)}"

def show_conversation_analytics():
    """대화 분석 결과 표시"""
    global persona_generator
    if not persona_generator or not hasattr(persona_generator, 'conversation_memory'):
        return "분석할 대화가 없습니다."
    
    memory = persona_generator.conversation_memory
    
    # 기본 통계
    analytics = f"## 📊 대화 분석 리포트\n\n"
    analytics += f"### 🔢 기본 통계\n"
    analytics += f"• 총 대화 수: {len(memory.conversations)}회\n"
    analytics += f"• 키워드 수: {len(memory.keywords)}개\n"
    analytics += f"• 활성 세션: {len(memory.user_profile)}개\n\n"
    
    # 상위 키워드
    top_keywords = memory.get_top_keywords(limit=10)
    if top_keywords:
        analytics += f"### 🔑 상위 키워드 TOP 10\n"
        for i, (word, data) in enumerate(top_keywords, 1):
            analytics += f"{i}. **{word}** ({data['category']}) - {data['total_frequency']}회\n"
        analytics += "\n"
    
    # 카테고리별 키워드
    categories = {}
    for word, data in memory.keywords.items():
        category = data['category']
        if category not in categories:
            categories[category] = []
        categories[category].append((word, data['total_frequency']))
    
    analytics += f"### 📂 카테고리별 관심사\n"
    for category, words in categories.items():
        top_words = sorted(words, key=lambda x: x[1], reverse=True)[:3]
        word_list = ", ".join([f"{word}({freq})" for word, freq in top_words])
        analytics += f"**{category}**: {word_list}\n"
    
    analytics += "\n"
    
    # 최근 감정 경향
    if memory.conversations:
        recent_sentiments = [conv['sentiment'] for conv in memory.conversations[-10:]]
        sentiment_counts = {"긍정적": 0, "부정적": 0, "중립적": 0}
        for sentiment in recent_sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        analytics += f"### 😊 최근 감정 경향 (최근 10회)\n"
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(recent_sentiments)) * 100
            analytics += f"• {sentiment}: {count}회 ({percentage:.1f}%)\n"
    
    return analytics

def get_keyword_suggestions(current_message=""):
    """현재 메시지 기반 키워드 제안"""
    global persona_generator
    if not persona_generator or not hasattr(persona_generator, 'conversation_memory'):
        return "키워드 분석을 위한 대화 기록이 없습니다."
    
    memory = persona_generator.conversation_memory
    
    if current_message:
        # 현재 메시지에서 키워드 추출
        extracted = memory._extract_keywords(current_message)
        suggestions = f"## 🎯 '{current_message}'에서 추출된 키워드\n\n"
        
        if extracted:
            for kw in extracted:
                suggestions += f"• **{kw['word']}** ({kw['category']}) - {kw['frequency']}회\n"
        else:
            suggestions += "추출된 키워드가 없습니다.\n"
        
        # 관련 과거 대화 찾기
        context = memory.get_relevant_context(current_message)
        if context["relevant_conversations"]:
            suggestions += f"\n### 🔗 관련된 과거 대화\n"
            for conv in context["relevant_conversations"][:3]:
                suggestions += f"• {conv['user_message'][:30]}... (감정: {conv['sentiment']})\n"
        
        return suggestions
    else:
        # 전체 키워드 요약
        top_keywords = memory.get_top_keywords(limit=15)
        if top_keywords:
            suggestions = "## 🔑 전체 키워드 요약\n\n"
            for word, data in top_keywords:
                suggestions += f"• **{word}** ({data['category']}) - {data['total_frequency']}회, 최근: {data['last_mentioned'][:10]}\n"
            return suggestions
        else:
            return "아직 수집된 키워드가 없습니다."

# 메인 인터페이스 생성
def create_main_interface():
    # 한글 폰트 설정
    setup_korean_font()
    
    # CSS 스타일 추가 - 텍스트 가시성 향상
    css = """
    .persona-greeting {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
    }
    
    .gradio-container {
        color: #333 !important;
    }
    
    .gr-markdown p {
        color: #333 !important;
    }
    
    .gr-textbox input {
        color: #333 !important;
    }
    
    .gr-json {
        color: #333 !important;
    }
    """
    
    # Gradio 앱 생성
    with gr.Blocks(title="놈팽쓰(MemoryTag) - 사물 페르소나 생성기", css=css, theme="soft") as app:
        # State 변수들 - Gradio 5.31.0에서는 반드시 Blocks 내부에서 정의
        current_persona = gr.State(value=None)
        personas_list = gr.State(value=[])
        
        gr.Markdown("""
        # 🎭 놈팽쓰(MemoryTag): 당신 곁의 사물, 이제 친구가 되다
        일상 속 사물에 AI 페르소나를 부여하여 대화할 수 있게 해주는 서비스입니다.
        """)
        
        with gr.Tabs() as tabs:
            # 페르소나 생성 탭
            with gr.Tab("페르소나 생성", id="creation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🌟 1단계: 영혼 발견하기")
                        image_input = gr.Image(type="pil", label="사물 이미지 업로드")
                        
                        with gr.Group():
                            gr.Markdown("### 기본 정보")
                            name_input = gr.Textbox(label="사물 이름 (선택사항)", placeholder="예: 책상 위 램프")
                            location_input = gr.Dropdown(
                                choices=["집", "사무실", "여행 중", "상점", "학교", "카페", "기타"],
                                label="주로 어디에 있나요?",
                                value="집"
                            )
                            time_spent_input = gr.Dropdown(
                                choices=["새것", "몇 개월", "1년 이상", "오래됨", "중고/빈티지"],
                                label="얼마나 함께했나요?",
                                value="몇 개월"
                            )
                            # AI 분석 결과 표시용 (사용자 입력 불가)
                            ai_analyzed_object_display = gr.Textbox(
                                label="AI가 분석한 사물 유형",
                                value="이미지 업로드 후 자동 분석됩니다",
                                interactive=False,
                                info="🤖 AI가 이미지를 분석하여 자동으로 파악합니다"
                            )
                            # 🆕 사물 용도/역할 입력 필드 추가
                            purpose_input = gr.Textbox(
                                label="이 사물의 용도/역할 (중요!) 🎯", 
                                placeholder="예: 나를 채찍질해서 운동하라고 닥달하는 역할, 밤늦게 공부할 때 응원해주는 친구, 아침에 일어나도록 깨워주는 알람 역할...",
                                lines=2,
                                info="이 사물과 어떤 소통을 원하시나요? 구체적으로 적어주세요!"
                            )
                        
                        create_btn = gr.Button("🌟 영혼 깨우기", variant="primary", size="lg")
                        status_output = gr.Markdown("")
                    
                    with gr.Column(scale=1):
                        # 페르소나 각성 결과
                        persona_awakening = gr.Markdown("", elem_classes=["persona-greeting"])
                        
                        # 페르소나 정보 표시 (사용자 친화적 형태)
                        persona_summary_display = gr.Markdown("", label="페르소나 정보")
                        
                        # 페르소나 각성 완료 후 조정 섹션 표시
                        adjustment_section = gr.Group(visible=False)
                        with adjustment_section:
                            gr.Markdown("### 🎯 2단계: 친구 성격 미세조정")
                            gr.Markdown("**3가지 핵심 지표**로 성격을 조정해보세요! (유머감각은 모든 페르소나가 기본적으로 높습니다 😄)")
                            
                            with gr.Row():
                                with gr.Column():
                                    warmth_slider = gr.Slider(
                                        minimum=0, maximum=100, value=50, step=1,
                                        label="온기 (따뜻함 정도)", 
                                        info="0: 차가움 ↔ 100: 따뜻함"
                                    )
                                    competence_slider = gr.Slider(
                                        minimum=0, maximum=100, value=50, step=1,
                                        label="능력 (유능함 정도)",
                                        info="0: 서툼 ↔ 100: 능숙함"
                                    )
                                
                                with gr.Column():
                                    extraversion_slider = gr.Slider(
                                        minimum=0, maximum=100, value=50, step=1,
                                        label="외향성 (활발함 정도)",
                                        info="0: 내향적, 조용함 ↔ 100: 외향적, 활발함"
                                    )
                                    
                                    humor_style_radio = gr.Radio(
                                        choices=["따뜻한 유머러스", "위트있는 재치꾼", "날카로운 관찰자", "자기 비하적", "장난꾸러기"],
                                        value="따뜻한 유머러스",
                                        label="유머 스타일 (모든 페르소나는 유머감각이 높습니다!)",
                                        info="어떤 방식으로 재미있게 만들까요?"
                                    )
                            
                            # 미리보기 표시 (실시간 업데이트 없음)
                            personality_preview = gr.Markdown("", elem_classes=["persona-greeting"], label="성격 조정 미리보기")
                            
                            with gr.Row():
                                preview_btn = gr.Button("👁️ 미리보기", variant="secondary")
                                adjust_btn = gr.Button("✨ 성격 조정 반영", variant="primary")
                            
                            with gr.Row():
                                finalize_btn = gr.Button("🎉 친구 확정하기!", variant="secondary")
                        
                        # 조정 결과 표시
                        adjustment_result = gr.Markdown("")
                        adjusted_info_output = gr.JSON(label="조정된 성격", visible=False)
                        
                        # 최종 완성 섹션
                        personality_traits_output = gr.JSON(label="성격 특성", visible=False)
                        
                        # 다운로드 섹션
                        with gr.Group():
                            gr.Markdown("### 📁 페르소나 내보내기")
                            with gr.Row():
                                save_btn = gr.Button("💾 페르소나 저장", variant="secondary")
                                persona_export_btn = gr.Button("📥 JSON 파일로 내보내기", variant="outline")
                            persona_download_file = gr.File(label="다운로드", visible=False)
            
            # 상세 정보 탭
            with gr.Tab("상세 정보", id="details"):
                with gr.Row():
                    with gr.Column():
                        chart_btn = gr.Button("📊 성격 차트 생성", variant="secondary")
                        personality_chart_output = gr.Plot(label="성격 차트")
                        humor_chart_output = gr.Plot(label="유머 매트릭스")
                    
                    with gr.Column():
                        attractive_flaws_output = gr.Dataframe(
                            headers=["매력적 결함", "효과"],
                            label="매력적 결함",
                            interactive=False
                        )
                        contradictions_output = gr.Dataframe(
                            headers=["모순적 특성", "효과"],
                            label="모순적 특성",
                            interactive=False
                        )
                
                with gr.Accordion("127개 성격 변수", open=False):
                    personality_variables_output = gr.Dataframe(
                        headers=["변수", "값", "카테고리", "수준"],
                        label="성격 변수",
                        interactive=False
                    )
            
            # 대화하기 탭
            with gr.Tab("대화하기", id="chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📁 페르소나 불러오기")
                        gr.Markdown("JSON 파일을 업로드하여 페르소나를 불러와 대화를 시작하세요.")
                        
                        json_upload = gr.File(
                            label="페르소나 JSON 파일 업로드",
                            file_types=[".json"],
                            type="filepath"
                        )
                        import_btn = gr.Button("JSON에서 페르소나 불러오기", variant="primary", size="lg")
                        load_status = gr.Markdown("")
                        
                        # 현재 로드된 페르소나 정보 표시
                        with gr.Group():
                            gr.Markdown("### 🤖 현재 페르소나")
                            chat_persona_greeting = gr.Markdown("", elem_classes=["persona-greeting"])
                            current_persona_info = gr.JSON(label="현재 페르소나 정보", visible=False)
                        
                        # 대화 기록 관리
                        with gr.Group():
                            gr.Markdown("### 💾 대화 기록 관리")
                            gr.Markdown("현재 대화를 JSON 파일로 다운로드하여 보관하세요.")
                            chat_export_btn = gr.Button("📥 현재 대화 기록 다운로드", variant="secondary")
                            chat_download_file = gr.File(label="다운로드", visible=False)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💬 대화")
                        # Gradio 4.x 호환: type="messages" 제거
                        chatbot = gr.Chatbot(height=400, label="대화", type="messages")
                        with gr.Row():
                            message_input = gr.Textbox(
                                placeholder="메시지를 입력하세요...",
                                show_label=False,
                                lines=2
                            )
                            send_btn = gr.Button("전송", variant="primary")
                        
                        # 대화 관련 버튼들
                        with gr.Row():
                            clear_btn = gr.Button("대화 초기화", variant="secondary", size="sm")
                            example_btn1 = gr.Button("\"안녕!\"", variant="outline", size="sm")
                            example_btn2 = gr.Button("\"너는 누구야?\"", variant="outline", size="sm")
                            example_btn3 = gr.Button("\"뭘 좋아해?\"", variant="outline", size="sm")
            
            # 🧠 대화 분석 탭 추가
            with gr.Tab("🧠 대화 분석"):
                gr.Markdown("### 📊 대화 기록 분석 및 키워드 추출")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📤 대화 기록 분석하기")
                        gr.Markdown("저장된 대화 기록 JSON 파일을 업로드하여 분석해보세요.")
                        
                        import_file = gr.File(label="📤 대화 기록 JSON 업로드", file_types=[".json"], type="filepath")
                        import_result = gr.Textbox(label="업로드 결과", lines=3, interactive=False)
                        
                    with gr.Column():
                        gr.Markdown("#### 🔍 실시간 키워드 분석")
                        keyword_input = gr.Textbox(label="분석할 메시지 (선택사항)", placeholder="메시지를 입력하면 키워드를 분석합니다")
                        keyword_btn = gr.Button("🎯 키워드 분석", variant="primary")
                        keyword_result = gr.Textbox(label="키워드 분석 결과", lines=10, interactive=False)
                
                gr.Markdown("---")
                
                with gr.Row():
                    analytics_btn = gr.Button("📈 전체 대화 분석 리포트", variant="primary", size="lg")
                
                analytics_result = gr.Markdown("### 분석 결과가 여기에 표시됩니다")
        
        # 이벤트 핸들러
        create_btn.click(
            fn=create_persona_from_image,
            inputs=[image_input, name_input, location_input, time_spent_input, gr.Textbox(value="auto"), purpose_input],
            outputs=[
                current_persona, status_output, persona_summary_display, personality_traits_output,
                humor_chart_output, attractive_flaws_output, contradictions_output, 
                personality_variables_output, persona_awakening, persona_download_file, adjustment_section,
                ai_analyzed_object_display  # 🆕 AI 분석 결과를 표시용 텍스트박스에 반영
            ]
        ).then(
            # 슬라이더 값을 현재 페르소나 값으로 업데이트
            fn=lambda persona: (
                persona["성격특성"]["온기"] if persona else 50,
                persona["성격특성"]["능력"] if persona else 50,
                persona["성격특성"]["외향성"] if persona else 50,
                persona["유머스타일"] if persona else "따뜻한 유머러스"
            ),
            inputs=[current_persona],
            outputs=[warmth_slider, competence_slider, extraversion_slider, humor_style_radio]
        ).then(
            # 초기 미리보기 생성
            fn=generate_realtime_preview,
            inputs=[current_persona, warmth_slider, competence_slider, extraversion_slider, humor_style_radio],
            outputs=[personality_preview]
        )
        
        # 🎯 미리보기 버튼 - 사용자가 수동으로 미리보기 요청
        preview_btn.click(
            fn=generate_realtime_preview,
            inputs=[current_persona, warmth_slider, competence_slider, extraversion_slider, humor_style_radio],
            outputs=[personality_preview]
        )
        
        # 성격 조정 반영 - 실제 페르소나에 적용
        adjust_btn.click(
            fn=adjust_persona_traits,
            inputs=[current_persona, warmth_slider, competence_slider, extraversion_slider, humor_style_radio],
            outputs=[current_persona, adjustment_result, adjusted_info_output, personality_variables_output]
        ).then(
            # 반영 후 미리보기도 업데이트
            fn=generate_realtime_preview,
            inputs=[current_persona, warmth_slider, competence_slider, extraversion_slider, humor_style_radio],
            outputs=[personality_preview]
        )
        
        # 페르소나 최종 확정
        finalize_btn.click(
            fn=finalize_persona,
            inputs=[current_persona],
            outputs=[
                current_persona, status_output, persona_summary_display, personality_traits_output,
                humor_chart_output, attractive_flaws_output, contradictions_output, 
                personality_variables_output, persona_awakening, persona_download_file
            ]
        )
        
        save_btn.click(
            fn=save_persona_to_file,
            inputs=[current_persona],
            outputs=[status_output]
        )
        
        # 성격 차트 생성
        chart_btn.click(
            fn=generate_personality_chart,
            inputs=[current_persona],
            outputs=[personality_chart_output]
        )
        
        # 페르소나 내보내기 버튼
        persona_export_btn.click(
            fn=export_persona_to_json,
            inputs=[current_persona],
            outputs=[persona_download_file]
        ).then(
            fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
            inputs=[persona_download_file],
            outputs=[persona_download_file]
        )
        
        import_btn.click(
            fn=import_persona_from_json,
            inputs=[json_upload],
            outputs=[
                current_persona, load_status, chat_persona_greeting, current_persona_info
            ]
        )
        
        # 대화 관련 이벤트 핸들러
        send_btn.click(
            fn=chat_with_loaded_persona,
            inputs=[current_persona, message_input, chatbot],
            outputs=[chatbot, message_input]
        )
        
        message_input.submit(
            fn=chat_with_loaded_persona,
            inputs=[current_persona, message_input, chatbot],
            outputs=[chatbot, message_input]
        )
        
        # 대화 초기화 (messages format)
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        # 예시 메시지 버튼들 - messages format 호환
        def handle_example_message(persona, message):
            if not persona:
                return [], ""
            # 빈 messages format 배열로 시작
            chat_result, _ = chat_with_loaded_persona(persona, message, [])
            return chat_result, ""
        
        example_btn1.click(
            fn=lambda persona: handle_example_message(persona, "안녕!"),
            inputs=[current_persona],
            outputs=[chatbot, message_input]
        )
        
        example_btn2.click(
            fn=lambda persona: handle_example_message(persona, "너는 누구야?"),
            inputs=[current_persona],
            outputs=[chatbot, message_input]
        )
        
        example_btn3.click(
            fn=lambda persona: handle_example_message(persona, "뭘 좋아해?"),
            inputs=[current_persona],
            outputs=[chatbot, message_input]
        )
        
        # 앱 로드 시 페르소나 목록 로드 (백엔드에서 사용)
        app.load(
            fn=lambda: [],
            outputs=[personas_list]
        )
        
        # 대화하기 탭의 대화 기록 다운로드 이벤트
        chat_export_btn.click(
            export_conversation_history,
            outputs=[chat_download_file]
        ).then(
            lambda x: gr.update(visible=True) if x else gr.update(visible=False),
            inputs=[chat_download_file],
            outputs=[chat_download_file]
        )
        
        # 대화 분석 탭의 업로드 이벤트
        import_file.upload(
            import_conversation_history,
            inputs=[import_file],
            outputs=[import_result]
        )
        
        keyword_btn.click(
            get_keyword_suggestions,
            inputs=[keyword_input],
            outputs=[keyword_result]
        )
        
        analytics_btn.click(
            show_conversation_analytics,
            outputs=[analytics_result]
        )
    
    return app

def generate_realtime_preview(persona, warmth, competence, extraversion, humor_style):
    """🤖 AI 기반 실시간 성격 조정 미리보기 생성"""
    global persona_generator
    
    if not persona:
        return "👤 페르소나를 먼저 생성해주세요"
    
    try:
        # 조정된 성격 특성
        adjusted_traits = {
            "온기": warmth,
            "능력": competence, 
            "외향성": extraversion,
            "유머감각": 75  # 기본적으로 높은 유머감각 유지
        }
        
        # 전체 페르소나 복사하여 성격만 조정
        import copy
        adjusted_persona = copy.deepcopy(persona)
        adjusted_persona["성격특성"] = adjusted_traits
        
        # 유머 스타일도 조정
        if humor_style:
            adjusted_persona["유머스타일"] = humor_style
        
        # AI 기반 인사말 생성
        ai_greeting = persona_generator.generate_ai_based_greeting(adjusted_persona, adjusted_traits)
        
        # 조정된 값들과 함께 표시
        adjustment_info = f"""**🎯 현재 성격 설정:**
- 온기: {warmth}/100 {'(따뜻함)' if warmth >= 60 else '(차가움)' if warmth <= 40 else '(보통)'}
- 능력: {competence}/100 {'(유능함)' if competence >= 60 else '(서툼)' if competence <= 40 else '(보통)'}
- 외향성: {extraversion}/100 {'(활발함)' if extraversion >= 60 else '(조용함)' if extraversion <= 40 else '(보통)'}
- 유머스타일: {humor_style}

**🤖 AI가 생성한 새로운 인사말:**
{ai_greeting}

*💡 성격 수치 변경 시마다 AI가 새로운 인사말을 생성합니다!*"""
        
        return adjustment_info
        
    except Exception as e:
        print(f"⚠️ 실시간 미리보기 AI 생성 실패: {e}")
        
        # 폴백: 기존 방식
        object_info = persona.get("기본정보", {})
        persona_name = object_info.get("이름", "친구")
        
        temp_traits = {
            "온기": warmth,
            "능력": competence, 
            "외향성": extraversion,
            "유머감각": 75
        }
        
        preview = generate_personality_preview(persona_name, temp_traits, persona)
        
        return f"""**🎯 현재 성격 설정:**
- 온기: {warmth}/100 {'(따뜻함)' if warmth >= 60 else '(차가움)' if warmth <= 40 else '(보통)'}
- 능력: {competence}/100 {'(유능함)' if competence >= 60 else '(서툼)' if competence <= 40 else '(보통)'}
- 외향성: {extraversion}/100 {'(활발함)' if extraversion >= 60 else '(조용함)' if extraversion <= 40 else '(보통)'}
- 유머스타일: {humor_style}

**👋 예상 인사말:**
{preview}"""

def show_variable_changes(original_persona, adjusted_persona):
    """변수 변화량을 시각화하여 표시"""
    if not original_persona or not adjusted_persona:
        return "변화량을 비교할 페르소나가 없습니다."
    
    # 원본과 조정된 변수들 가져오기
    original_vars = original_persona.get("성격변수127", {})
    if not original_vars and "성격프로필" in original_persona:
        original_vars = original_persona["성격프로필"]
    
    adjusted_vars = adjusted_persona.get("성격변수127", {})
    if not adjusted_vars and "성격프로필" in adjusted_persona:
        adjusted_vars = adjusted_persona["성격프로필"]
    
    if not original_vars or not adjusted_vars:
        return "변수 데이터를 찾을 수 없습니다."
    
    # 변화량 계산
    changes = []
    significant_changes = []  # 변화량이 10 이상인 항목들
    
    for var in original_vars:
        if var in adjusted_vars:
            original_val = original_vars[var]
            adjusted_val = adjusted_vars[var]
            change = adjusted_val - original_val
            
            changes.append((var, original_val, adjusted_val, change))
            
            if abs(change) >= 10:  # 변화량이 10 이상인 것만
                significant_changes.append((var, original_val, adjusted_val, change))
    
    # 카테고리별 평균 변화량 계산
    category_changes = {}
    for var, orig, adj, change in changes:
        if var.startswith('W'):
            category = "온기"
        elif var.startswith('C'):
            category = "능력"
        elif var.startswith('E'):
            category = "외향성"
        elif var.startswith('H'):
            category = "유머"
        else:
            category = "기타"
        
        if category not in category_changes:
            category_changes[category] = []
        category_changes[category].append(change)
    
    # 평균 변화량 계산
    avg_changes = {}
    for category, change_list in category_changes.items():
        avg_changes[category] = sum(change_list) / len(change_list)
    
    # 결과 포맷팅
    result = "### 🔄 성격 변수 변화량 분석\n\n"
    
    # 카테고리별 평균 변화량
    result += "**📊 카테고리별 평균 변화량:**\n"
    for category, avg_change in avg_changes.items():
        if avg_change > 5:
            trend = "⬆️ 상승"
        elif avg_change < -5:
            trend = "⬇️ 하락"
        else:
            trend = "➡️ 유지"
        result += f"- {category}: {avg_change:+.1f} {trend}\n"
    
    # 주요 변화량 (10 이상)
    if significant_changes:
        result += f"\n**🎯 주요 변화 항목 ({len(significant_changes)}개):**\n"
        for var, orig, adj, change in sorted(significant_changes, key=lambda x: abs(x[3]), reverse=True)[:10]:
            if change > 0:
                arrow = "⬆️"
                color = "🟢"
            else:
                arrow = "⬇️" 
                color = "🔴"
            
            result += f"- {var}: {orig} → {adj} ({change:+.0f}) {arrow} {color}\n"
    
    result += f"\n**📈 총 변수 개수:** {len(changes)}개\n"
    result += f"**🔄 변화된 변수:** {len([c for c in changes if c[3] != 0])}개\n"
    result += f"**📊 주요 변화:** {len(significant_changes)}개 (변화량 ±10 이상)\n"
    
    return result

if __name__ == "__main__":
    app = create_main_interface()
    app.launch(server_name="0.0.0.0", server_port=7860) 