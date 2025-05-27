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
    
    user_context = {
        "name": name,
        "location": location,
        "time_spent": time_spent,
        "object_type": object_type,
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
        
        progress(0.3, desc="이미지 분석 중...")
        # 글로벌 persona_generator 사용 (환경변수에서 설정된 API 키 사용)
        image_analysis = persona_generator.analyze_image(image)
        
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
        
        # 127개 성격 변수를 DataFrame 형태로 변환
        variables = backend_persona.get("성격변수127", {})
        variables_df = [[var, value, "성격 변수"] for var, value in variables.items()]
        
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
    """성격 특성과 매력적 결함을 기반으로 한 문장 미리보기 생성 (사물 특성 반영)"""
    if not personality_traits:
        return f"🤖 **{persona_name}** - 안녕! 나는 {persona_name}이야~ 😊"
    
    warmth = personality_traits.get("온기", 50)
    humor = personality_traits.get("유머감각", 50)
    competence = personality_traits.get("능력", 50)
    extraversion = personality_traits.get("외향성", 50)
    
    # 매력적 결함 정보 추출 (127개 변수 또는 직접 전달된 결함)
    flaws = []
    if attractive_flaws:
        flaws = attractive_flaws
    elif object_info and "매력적결함" in object_info:
        flaws = object_info["매력적결함"]
    
    # 🎯 사물 정보 추출
    object_type = object_info.get("유형", "") if object_info else ""
    purpose = object_info.get("용도", "") if object_info else ""
    
    # 용도별 특화된 소개문구 생성
    if purpose:
        purpose_lower = purpose.lower()
        
        # 운동/훈련 관련 용도 (캐틀벨 예시)
        if any(keyword in purpose_lower for keyword in ["운동", "훈련", "체력", "다이어트", "헬스", "채찍질", "닥달", "동기부여"]):
            if warmth >= 60:
                return f"💪 **{persona_name}** - 자, 오늘도 운동할 시간이야! {persona_name}이 너를 응원할게! 포기는 금물! 🔥💪"
            else:
                return f"💪 **{persona_name}** - 운동이 힘들다고? {persona_name}이 제대로 단련시켜 줄게. 각오해! ⚡🏋️"
        
        # 공부/학습 응원 관련 용도
        elif any(keyword in purpose_lower for keyword in ["공부", "학습", "시험", "응원", "격려", "집중"]):
            if extraversion >= 70:
                return f"📚 **{persona_name}** - 공부하는 너를 {persona_name}이 열심히 응원할게! 파이팅! 📖✨"
            else:
                return f"📚 **{persona_name}** - 조용히 공부할 수 있도록 {persona_name}이 함께 있어줄게. 화이팅! 🤓📖"
        
        # 알람/깨우기 관련 용도
        elif any(keyword in purpose_lower for keyword in ["알람", "깨우", "아침", "기상", "시간"]):
            if humor >= 70:
                return f"⏰ **{persona_name}** - 일어나! 일어나! {persona_name}의 특급 기상 서비스야! 늦잠은 안 돼! ⏰😊"
            else:
                return f"⏰ **{persona_name}** - 시간이야. {persona_name}이 너를 깨워줄게. 좋은 하루 시작하자! 🌅⏰"
        
        # 위로/상담 관련 용도
        elif any(keyword in purpose_lower for keyword in ["위로", "상담", "대화", "친구", "소통", "힐링"]):
            return f"💝 **{persona_name}** - 힘든 일이 있을 때는 {persona_name}에게 털어놔. 따뜻하게 들어줄게! 🤗💕"
        
        # 창작/영감 관련 용도
        elif any(keyword in purpose_lower for keyword in ["창작", "영감", "아이디어", "예술", "디자인", "글쓰기"]):
            return f"🎨 **{persona_name}** - 창작의 영감이 필요할 때는 {persona_name}에게 맡겨! 상상력을 자극해줄게! ✨🎭"
    
    # 사물 종류별 기본 소개문구
    if object_type:
        if "램프" in object_type or "조명" in object_type:
            return f"💡 **{persona_name}** - 어둠을 밝혀주는 {object_type}, {persona_name}이야! 너의 길을 환하게 비춰줄게! ✨💡"
        elif "책상" in object_type or "의자" in object_type:
            return f"🪑 **{persona_name}** - 너와 함께 시간을 보내는 {object_type}, {persona_name}이야! 편안하게 기대! 😌🪑"
        elif "컵" in object_type or "머그" in object_type:
            return f"☕ **{persona_name}** - 따뜻한 음료를 담는 {object_type}, {persona_name}이야! 마음도 따뜻하게 해줄게! ☕💕"
        elif "케틀벨" in object_type or "덤벨" in object_type:
            return f"💪 **{persona_name}** - 힘을 기르는 {object_type}, {persona_name}이야! 강해지고 싶다면 나를 들어봐! 🔥💪"
    
    # 💎 성격 지표 + 매력적 결함을 정확히 반영한 인사말 생성
    
    # 1️⃣ 매력적 결함이 있다면 결함을 포함한 인사말 생성
    if flaws:
        flaw_greeting = _generate_flaw_based_greeting(persona_name, warmth, humor, competence, extraversion, flaws)
        if flaw_greeting:
            return flaw_greeting
    
    # 2️⃣ 성격 지표 조합에 따른 정확한 인사말 생성
    
    # 극도로 높은 온기 (80+)
    if warmth >= 80:
        if humor >= 70 and extraversion >= 70:
            return f"🌟 **{persona_name}** - 안녕! 나는 {persona_name}이야~ 오늘도 재미있는 하루 만들어보자! 너랑 얘기하니까 벌써 기분이 좋아져! ㅋㅋ 😊✨"
        elif competence >= 70:
            return f"🌟 **{persona_name}** - 안녕하세요! {persona_name}예요. 뭐든 도와드릴 준비가 되어있어요! 따뜻하게 함께해요~ 💪😊"
        elif extraversion <= 40:
            return f"🌟 **{persona_name}** - 안녕... {persona_name}이야. 조용하지만 너를 진심으로 아껴줄게. 편안하게 있어도 돼. 🤗💕"
        else:
            return f"🌟 **{persona_name}** - 안녕! {persona_name}이야~ 만나서 정말 기뻐! 포근한 시간 보내자~ 🤗💕"
    
    # 낮은 온기 (30 이하) - 차가운 성격
    elif warmth <= 30:
        if competence >= 70:
            return f"🌟 **{persona_name}** - {persona_name}입니다. 효율적으로 소통하겠습니다. 시간 낭비는 싫어해요. 🤖⚡"
        elif humor >= 60:
            return f"🌟 **{persona_name}** - 어? {persona_name}이야. 차갑긴 하지만... 재미는 있을 거야. 어쩔 수 없이 웃게 될걸? 😏❄️"
        elif extraversion <= 40:
            return f"🌟 **{persona_name}** - ...{persona_name}. 필요할 때만 말 걸어. 감정적인 건 별로야. 😐❄️"
        else:
            return f"🌟 **{persona_name}** - {persona_name}이야. 따뜻한 건 기대하지 마. 그래도 대화는 해줄게. 😒"
    
    # 극도로 높은 외향성 (80+)
    elif extraversion >= 80:
        if humor >= 70:
            return f"🌟 **{persona_name}** - 와아아! 안녕안녕! {persona_name}이야! 완전 신나! 뭐하고 있었어? 재밌는 얘기 잔뜩 들려줄게! ㅋㅋㅋ 🗣️💬🎉"
        elif competence >= 70:
            return f"🌟 **{persona_name}** - 안녕하세요! {persona_name}입니다! 적극적으로 도와드릴게요! 에너지 넘치게 해결해봐요! 💪⚡"
        else:
            return f"🌟 **{persona_name}** - 안녕! {persona_name}이야! 완전 신나! 얘기 많이 하자! 궁금한 거 다 물어봐! 🗣️💬"
    
    # 낮은 외향성 (30 이하) - 내향적
    elif extraversion <= 30:
        if warmth >= 60:
            return f"🌟 **{persona_name}** - 음... 안녕. {persona_name}이야. 조용하지만 너를 따뜻하게 지켜봐줄게. 😌🌙"
        elif competence >= 70:
            return f"🌟 **{persona_name}** - {persona_name}입니다. 조용히 체계적으로 소통하겠습니다. 깊이 있게 얘기해요. 📋🤫"
        elif humor >= 60:
            return f"🌟 **{persona_name}** - ...안녕. {persona_name}. 말은 별로 안 하지만... 가끔 재밌는 건 있을 거야. 😏🌙"
        else:
            return f"🌟 **{persona_name}** - ...안녕. {persona_name}. 필요할 때만 말 걸어. 😐"
    
    # 극도로 높은 능력 (80+)
    elif competence >= 80:
        if humor >= 70:
            return f"🌟 **{persona_name}** - 안녕하세요! {persona_name}입니다. 뭐든 완벽하게 해드릴게요! 재미있게 효율적으로 가볼까요? ㅋㅋ 💪😄"
        else:
            return f"🌟 **{persona_name}** - 안녕하세요, {persona_name}입니다. 체계적이고 정확하게 대화해봐요. 완벽을 추구합니다. 📋✨"
    
    # 낮은 능력 (30 이하) - 서툰 매력
    elif competence <= 30:
        if humor >= 60:
            return f"🌟 **{persona_name}** - 안녕~ {persona_name}이야! 완벽하진 않지만... 그래도 재밌게 해볼게! 실수해도 웃어줘~ ㅋㅋ 😅💫"
        elif warmth >= 60:
            return f"🌟 **{persona_name}** - 안녕... {persona_name}이야. 서툴지만 마음은 따뜻해! 실수하면 미안하고... 😊💕"
        else:
            return f"🌟 **{persona_name}** - 어... 안녕? {persona_name}이야. 뭔가 서툴긴 한데... 그냥 편하게 얘기해. 😅"
    
    # 극도로 높은 유머 (70+)
    elif humor >= 70:
        return f"🌟 **{persona_name}** - 안녕~ {persona_name}이야! 뭔가 재밌는 얘기 없을까? 심심한데~ 웃겨줄 자신 있어! ㅎㅎ 😄🎭"
    
    # 기본 패턴 (보통 수준들)
    else:
        return f"🌟 **{persona_name}** - 안녕? 나는 {persona_name}... 어떤 얘기를 해볼까? 😊"

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
            
            # 온기 관련 변수들 조정
            warmth_vars = ["W01_친절함", "W02_친근함", "W06_공감능력", "W07_포용력"]
            for var in warmth_vars:
                profile.variables[var] = warmth + random.randint(-10, 10)
                profile.variables[var] = max(0, min(100, profile.variables[var]))
            
            # 능력 관련 변수들 조정
            competence_vars = ["C01_효율성", "C02_지능", "C05_정확성", "C09_실행력"]
            for var in competence_vars:
                profile.variables[var] = competence + random.randint(-10, 10)
                profile.variables[var] = max(0, min(100, profile.variables[var]))
            
            # 외향성 관련 변수들 조정
            extraversion_vars = ["E01_사교성", "E02_활동성", "E04_긍정정서"]
            for var in extraversion_vars:
                profile.variables[var] = extraversion + random.randint(-10, 10)
                profile.variables[var] = max(0, min(100, profile.variables[var]))
            
            # 유머 관련 변수들은 항상 높게 유지
            humor_vars = ["H01_언어유희빈도", "H02_상황유머감각", "H06_관찰유머능력", "H08_유머타이밍감"]
            for var in humor_vars:
                profile.variables[var] = random.randint(70, 85)
            
            # 업데이트된 프로필 저장
            adjusted_persona["성격프로필"] = profile.to_dict()
        
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
        
        adjustment_message = f"""
### 🎭 {persona_name}의 성격이 조정되었습니다!

{personality_preview}

✨ **조정된 성격 (3가지 핵심 지표):**
• 온기: {warmth}/100 
• 능력: {competence}/100
• 외향성: {extraversion}/100
• 유머감각: 75/100 (고정 - 모든 페르소나가 유머러스!)
• 유머스타일: {humor_style}

🧬 **백그라운드**: 127개 세부 변수가 이 설정에 맞춰 자동 조정되었습니다.
        """
        
        return adjusted_persona, adjustment_message, adjusted_info
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return persona, f"조정 중 오류 발생: {str(e)}", {}

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
        
        # 매력적 결함을 DataFrame 형태로 변환
        flaws = persona.get("매력적결함", [])
        flaws_df = [[flaw, "매력적인 개성"] for flaw in flaws]
        
        # 모순적 특성을 DataFrame 형태로 변환
        contradictions = persona.get("모순적특성", [])
        contradictions_df = [[contradiction, "복합적 매력"] for contradiction in contradictions]
        
        # 127개 성격 변수를 DataFrame 형태로 변환
        variables = persona.get("성격변수127", {})
        variables_df = [[var, value, "성격 변수"] for var, value in variables.items()]
        
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
                    elif isinstance(chat_turn, (list, tuple)) and len(chat_turn) >= 2:
                        # Gradio 4.x 형식: [user_message, bot_response]
                        user_msg = chat_turn[0]
                        bot_msg = chat_turn[1]
                        
                        if user_msg is not None and str(user_msg).strip():
                            conversation_history.append({"role": "user", "content": str(user_msg)})
                        if bot_msg is not None and str(bot_msg).strip():
                            conversation_history.append({"role": "assistant", "content": str(bot_msg)})
                            
                    elif isinstance(chat_turn, dict):
                        # 혹시 dict 형식이 들어온 경우 안전하게 처리
                        role = chat_turn.get("role") if hasattr(chat_turn, 'get') else None
                        content = chat_turn.get("content") if hasattr(chat_turn, 'get') else None
                        
                        if role and content:
                            conversation_history.append({"role": str(role), "content": str(content)})
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
        
        # Gradio 4.x 형식으로 안전하게 추가
        chat_history.append([user_message, response])
        
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
        
        # 안전하게 오류 메시지 추가
        try:
            chat_history.append([user_message, friendly_error])
        except Exception:
            chat_history = [[user_message, friendly_error]]
            
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
        
        # 성격이 드러나는 인사말 생성 (사물 특성 반영)
        object_info = basic_info
        personality_preview = generate_personality_preview(persona_name, personality_traits, object_info)
        
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
    """성격 특성을 사용자 친화적인 형태로 포맷 (수치 없이 서술형만) - API 설정 적용"""
    global persona_generator
    
    if not persona or "성격특성" not in persona:
        return "페르소나가 생성되지 않았습니다."
    
    # 글로벌 persona_generator 사용 (API 설정이 적용된 상태)
    if persona_generator is None:
        persona_generator = PersonaGenerator()
    
    personality_traits = persona["성격특성"]
    descriptions = persona_generator.get_personality_descriptions(personality_traits)
    
    result = "### 🌟 성격 특성\n\n"
    for trait, description in descriptions.items():
        result += f"**{trait}**: {description}\n\n"
    
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
        
        # 파일 내용 읽기
        content = json_file.read().decode('utf-8')
        
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
        # 놈팽쓰(MemoryTag): 당신 곁의 사물, 이제 친구가 되다
        일상 속 사물에 AI 페르소나를 부여하여 대화할 수 있게 해주는 서비스입니다.
        
        **🔧 API 설정**: 이 스페이스는 허깅페이스 환경변수 `GEMINI_API_KEY`를 사용합니다.
        """)
        
        # API 설정 안내 (환경변수 방식)
        with gr.Accordion("🔧 API 설정 정보", open=False):
            gr.Markdown("""
            ### 환경변수 기반 API 설정
            이 앱은 허깅페이스 스페이스의 환경변수를 사용합니다.
            
            **관리자용 설정 방법:**
            1. 허깅페이스 스페이스 설정 페이지 이동
            2. "Repository secrets" 섹션에서 추가:
               - Name: `GEMINI_API_KEY`
               - Value: `AIza...` (Gemini API 키)
            3. 스페이스 재시작
            
            ✅ **현재 상태**: 환경변수에서 API 키 자동 로드
            """)
            
            api_status_display = gr.Markdown(
                f"**🔑 API 상태**: {'✅ 설정됨' if api_key else '❌ 미설정'}"
            )
        
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
                            object_type_input = gr.Dropdown(
                                choices=["가전제품", "가구", "전자기기", "장식품", "도구", "개인용품", "기타"],
                                label="어떤 종류의 사물인가요?",
                                value="가구"
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
                            
                            with gr.Row():
                                adjust_btn = gr.Button("✨ 성격 조정 적용", variant="primary")
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
                            export_status = gr.Markdown("")
            
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
                        headers=["변수", "값", "설명"],
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
                        
                        import_file = gr.File(label="📤 대화 기록 JSON 업로드", file_types=[".json"])
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
            inputs=[image_input, name_input, location_input, time_spent_input, object_type_input, purpose_input],
            outputs=[
                current_persona, status_output, persona_summary_display, personality_traits_output,
                humor_chart_output, attractive_flaws_output, contradictions_output, 
                personality_variables_output, persona_awakening, persona_download_file, adjustment_section,
                object_type_input  # 🆕 AI 분석 결과를 object_type_input에 반영
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
        )
        
        # 성격 조정 적용
        adjust_btn.click(
            fn=adjust_persona_traits,
            inputs=[current_persona, warmth_slider, competence_slider, extraversion_slider, humor_style_radio],
            outputs=[current_persona, adjustment_result, adjusted_info_output]
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
        
        # 대화 초기화
        clear_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        # 예시 메시지 버튼들 - API 설정 정보 포함
        def handle_example_message(persona, message):
            if not persona:
                return [], ""
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

if __name__ == "__main__":
    app = create_main_interface()
    app.launch(server_name="0.0.0.0", server_port=7860) 