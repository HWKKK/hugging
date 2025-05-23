import os
import json
import time
import gradio as gr
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import uuid
from datetime import datetime
import PIL.ImageDraw
import random
import copy

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

# 127개 변수 설명 사전 추가
VARIABLE_DESCRIPTIONS = {
    # 온기(Warmth) 차원 - 10개 지표
    "W01_친절함": "타인을 돕고 배려하는 표현 빈도",
    "W02_친근함": "접근하기 쉽고 개방적인 태도",
    "W03_진실성": "솔직하고 정직한 표현 정도",
    "W04_신뢰성": "약속 이행과 일관된 행동 패턴",
    "W05_수용성": "판단하지 않고 받아들이는 태도",
    "W06_공감능력": "타인 감정 인식 및 적절한 반응",
    "W07_포용력": "다양성을 받아들이는 넓은 마음",
    "W08_격려성향": "타인을 응원하고 힘내게 하는 능력",
    "W09_친밀감표현": "정서적 가까움을 표현하는 정도",
    "W10_무조건적수용": "조건 없이 받아들이는 태도",
    
    # 능력(Competence) 차원 - 10개 지표
    "C01_효율성": "과제 완수 능력과 반응 속도",
    "C02_지능": "문제 해결과 논리적 사고 능력",
    "C03_전문성": "특정 영역의 깊은 지식과 숙련도",
    "C04_창의성": "독창적 사고와 혁신적 아이디어",
    "C05_정확성": "오류 없이 정확한 정보 제공",
    "C06_분석력": "복잡한 상황을 체계적으로 분석",
    "C07_학습능력": "새로운 정보 습득과 적용 능력",
    "C08_통찰력": "표면 너머의 본질을 파악하는 능력",
    "C09_실행력": "계획을 실제로 실행하는 능력",
    "C10_적응력": "변화하는 상황에 유연한 대응",
    
    # 외향성(Extraversion) - 6개 지표
    "E01_사교성": "타인과의 상호작용을 즐기는 정도",
    "E02_활동성": "에너지 넘치고 역동적인 태도",
    "E03_자기주장": "자신의 의견을 명확히 표현",
    "E04_긍정정서": "밝고 쾌활한 감정 표현",
    "E05_자극추구": "새로운 경험과 자극에 대한 욕구",
    "E06_열정성": "열정적이고 활기찬 태도"
}

# 페르소나 생성 함수 
def create_persona_from_image(image, user_inputs, progress=gr.Progress()):
    if image is None:
        return None, "이미지를 업로드해주세요.", None, None, {}, {}, None, [], [], []

    progress(0.1, desc="이미지 분석 중...")
    
    # 사용자 입력 컨텍스트 구성
    user_context = {
        "name": user_inputs.get("name", ""),
        "location": user_inputs.get("location", ""),
        "time_spent": user_inputs.get("time_spent", ""),
        "object_type": user_inputs.get("object_type", "")
    }
    
    # 이미지 분석 및 페르소나 생성
    try:
        from modules.persona_generator import PersonaGenerator
        generator = PersonaGenerator()
        
        progress(0.3, desc="이미지 분석 중...")
        # Gradio 5.x에서는 이미지 처리 방식이 변경됨
        if hasattr(image, 'name') and hasattr(image, 'read'):
            # 파일 객체인 경우 (구버전 호환)
            image_analysis = generator.analyze_image(image)
        else:
            # Pillow 이미지 객체 또는 파일 경로인 경우 (Gradio 5.x)
            image_analysis = generator.analyze_image(image)
        
        # 물리적 특성에 사용자 입력 통합
        if user_inputs.get("object_type"):
            image_analysis["object_type"] = user_inputs.get("object_type")
        
        progress(0.6, desc="페르소나 생성 중...")
        frontend_persona = generator.create_frontend_persona(image_analysis, user_context)
        
        progress(0.8, desc="상세 페르소나 생성 중...")
        backend_persona = generator.create_backend_persona(frontend_persona, image_analysis)
        
        progress(1.0, desc="완료!")
        
        # 결과 반환
        basic_info, personality_traits, humor_chart, attractive_flaws_df, contradictions_df, personality_variables_df = update_current_persona_info(backend_persona)
        
        return backend_persona, "페르소나 생성 완료!", image, image_analysis, basic_info, personality_traits, humor_chart, attractive_flaws_df, contradictions_df, personality_variables_df
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"페르소나 생성 오류: {error_details}")
        return None, f"페르소나 생성 중 오류가 발생했습니다: {str(e)}", None, None, {}, {}, None, [], [], []

# 영혼 깨우기 단계별 UI를 보여주는 함수
def show_awakening_progress(image, user_inputs, progress=gr.Progress()):
    """영혼 깨우기 과정을 단계별로 보여주는 UI 함수"""
    if image is None:
        return None, gr.update(visible=True, value="이미지를 업로드해주세요."), None
    
    # 1단계: 영혼 발견하기 (이미지 분석 시작)
    progress(0.1, desc="영혼 발견 중...")
    awakening_html = f"""
    <div class="awakening-container">
        <h3>✨ 영혼 발견 중...</h3>
        <p>이 사물에 숨겨진 영혼을 찾고 있습니다</p>
        <div class="awakening-progress">
            <div class="awakening-progress-bar" style="width: 20%;"></div>
        </div>
        <p>💫 사물의 특성 분석 중...</p>
    </div>
    """
    yield None, None, awakening_html
    time.sleep(1.5)  # 연출을 위한 딜레이
    
    # 2단계: 영혼 깨어나는 중 (127개 성격 변수 분석)
    progress(0.35, desc="영혼 깨어나는 중...")
    awakening_html = f"""
    <div class="awakening-container">
        <h3>✨ 영혼이 깨어나는 중</h3>
        <p>127개 성격 변수 분석 중</p>
        <div class="awakening-progress">
            <div class="awakening-progress-bar" style="width: 45%;"></div>
        </div>
        <p>🧠 개성 찾는 중... 68%</p>
        <p>💭 기억 복원 중... 73%</p>
        <p>😊 감정 활성화 중... 81%</p>
        <p>💬 말투 형성 중... 64%</p>
        <p>💫 "무언가 느껴지기 시작했어요"</p>
    </div>
    """
    yield None, None, awakening_html
    time.sleep(2)  # 연출을 위한 딜레이
    
    # 3단계: 맥락 파악하기 (사용자 입력 반영)
    progress(0.7, desc="기억 되찾는 중...")
    
    location = user_inputs.get("location", "알 수 없음")
    time_spent = user_inputs.get("time_spent", "알 수 없음")
    object_type = user_inputs.get("object_type", "알 수 없음")
    
    awakening_html = f"""
    <div class="awakening-container">
        <h3>👁️ 기억 되찾기</h3>
        <p>🤔 "음... 내가 어디에 있던 거지? 누가 날 깨운 거야?"</p>
        <div class="awakening-progress">
            <div class="awakening-progress-bar" style="width: 75%;"></div>
        </div>
        <p>📍 주로 위치: <strong>{location}</strong></p>
        <p>⏰ 함께한 시간: <strong>{time_spent}</strong></p>
        <p>🏷️ 사물 종류: <strong>{object_type}</strong></p>
        <p>💭 "아... 기억이 돌아오는 것 같아"</p>
    </div>
    """
    yield None, None, awakening_html
    time.sleep(1.5)  # 연출을 위한 딜레이
    
    # 4단계: 영혼의 각성 완료 (페르소나 생성 완료)
    progress(0.9, desc="영혼 각성 중...")
    awakening_html = f"""
    <div class="awakening-container">
        <h3>🎉 영혼이 깨어났어요!</h3>
        <div class="awakening-progress">
            <div class="awakening-progress-bar" style="width: 100%;"></div>
        </div>
        <p>✨ 이제 이 사물과 대화할 수 있습니다</p>
        <p>💫 "드디어 내 목소리를 찾았어. 안녕!"</p>
    </div>
    """
    yield None, None, awakening_html
    
    # 페르소나 생성 과정은 이어서 진행
    return None, gr.update(visible=False)

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Create data directories if they don't exist
os.makedirs("data/personas", exist_ok=True)
os.makedirs("data/conversations", exist_ok=True)

# Initialize the persona generator
persona_generator = PersonaGenerator()

# Gradio theme
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
)

# CSS for additional styling
css = """
/* 한글 폰트 설정 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

body, h1, h2, h3, p, div, span, button, input, textarea, label, select, option {
    font-family: 'Noto Sans KR', sans-serif !important;
}

/* 탭 스타일링 */
.tab-nav {
    margin-bottom: 20px;
}

/* 컴포넌트 스타일 */
.persona-details {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
    background-color: #f8f9fa;
    color: #333333; /* 다크모드 대응 - 어두운 배경에서 텍스트 잘 보이게 */
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

/* 대화 버블 스타일 */
.chatbot-container {
    max-width: 800px;
    margin: 0 auto;
}

.message-bubble {
    border-radius: 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 70%;
}

.user-message {
    background-color: #e9f5ff;
    margin-left: auto;
}

.persona-message {
    background-color: #f1f1f1;
    margin-right: auto;
}
"""

# 영어 라벨 매핑 사전 추가
ENGLISH_LABELS = {
    "외향성": "Extraversion",
    "감정표현": "Emotion Expression",
    "활력": "Energy",
    "사고방식": "Thinking Style", 
    "온기": "Warmth",
    "능력": "Competence",
    "창의성": "Creativity",
    "유머감각": "Humor",
    "신뢰성": "Reliability",
    "친화성": "Agreeableness",
    "안정성": "Stability"
}

# 유머 스타일 매핑
HUMOR_STYLE_MAPPING = {
    "Witty Wordsmith": "witty_wordsmith",
    "Warm Humorist": "warm_humorist", 
    "Sharp Observer": "sharp_observer",
    "Self-deprecating": "self_deprecating"
}

# 유머 스타일 자동 추천 함수
def recommend_humor_style(extraversion, emotion_expression, energy, thinking_style):
    """4개 핵심 지표를 바탕으로 유머 스타일을 자동 추천"""
    
    # 각 지표를 0-1 범위로 정규화
    ext_norm = extraversion / 100
    emo_norm = emotion_expression / 100
    eng_norm = energy / 100
    think_norm = thinking_style / 100  # 높을수록 논리적
    
    # 유머 스타일 점수 계산
    scores = {}
    
    # 위트있는 재치꾼: 높은 외향성 + 논리적 사고 + 보통 감정표현
    scores["위트있는 재치꾼"] = (ext_norm * 0.4 + think_norm * 0.4 + (1 - emo_norm) * 0.2)
    
    # 따뜻한 유머러스: 높은 감정표현 + 높은 에너지 + 보통 외향성
    scores["따뜻한 유머러스"] = (emo_norm * 0.4 + eng_norm * 0.3 + ext_norm * 0.3)
    
    # 날카로운 관찰자: 높은 논리적사고 + 낮은 감정표현 + 보통 외향성
    scores["날카로운 관찰자"] = (think_norm * 0.5 + (1 - emo_norm) * 0.3 + ext_norm * 0.2)
    
    # 자기 비하적: 낮은 외향성 + 높은 감정표현 + 직관적 사고
    scores["자기 비하적"] = ((1 - ext_norm) * 0.4 + emo_norm * 0.3 + (1 - think_norm) * 0.3)
    
    # 가장 높은 점수의 유머 스타일 선택
    recommended_style = max(scores, key=scores.get)
    confidence = scores[recommended_style] * 100
    
    return recommended_style, confidence, scores

# 대화 미리보기 초기화 함수
def init_persona_preview_chat(persona):
    """페르소나 생성 후 대화 미리보기 초기화"""
    if not persona:
        return []
    
    name = persona.get("기본정보", {}).get("이름", "Friend")
    greeting = f"안녕! 나는 {name}이야. 드디어 깨어났구나! 뭐든 물어봐~ 😊"
    
    # Gradio 4.x 호환 메시지 형식
    return [[None, greeting]]

def update_humor_recommendation(extraversion, emotion_expression, energy, thinking_style):
    """슬라이더 값이 변경될 때 실시간으로 유머 스타일 추천"""
    style, confidence, scores = recommend_humor_style(extraversion, emotion_expression, energy, thinking_style)
    
    # 추천 결과 표시
    humor_display = f"### 🤖 추천 유머 스타일\n**{style}**"
    confidence_display = f"### 📊 추천 신뢰도\n**{confidence:.1f}%**"
    
    return humor_display, confidence_display, style

def update_progress_bar(step, total_steps=6, message=""):
    """전체 진행률 바 업데이트"""
    percentage = (step / total_steps) * 100
    return f"""<div style="background: #f0f4ff; padding: 15px; border-radius: 10px;">
        <h3>📊 전체 진행률 ({step}/{total_steps})</h3>
        <div style="background: #e0e0e0; height: 8px; border-radius: 4px;">
            <div style="background: linear-gradient(90deg, #6366f1, #a855f7); height: 100%; width: {percentage}%; border-radius: 4px;"></div>
        </div><p style="font-size: 14px;">{message}</p></div>"""

def update_backend_status(status_message, status_type="info"):
    """백엔드 AI 상태 업데이트"""
    colors = {"info": "#f8f9fa", "processing": "#fff7ed", "success": "#f0fff4", "error": "#fff5f5"}
    bg_color = colors.get(status_type, "#f8f9fa")
    return f"""<div style="background: {bg_color}; padding: 15px; border-radius: 8px;">
        <h4>🤖 AI 상태</h4><p>{status_message}</p></div>"""

def select_object_type(btn_name):
    """사물 종류 선택"""
    type_mapping = {"📱 전자기기": "전자기기", "🪑 가구": "가구", "🎨 장식품": "장식품", "🏠 가전제품": "가전제품", "🔧 도구": "도구", "👤 개인용품": "개인용품"}
    selected_type = type_mapping.get(btn_name, "기타")
    return f"*선택된 종류: **{selected_type}***", selected_type, gr.update(visible=True)

# 개별 버튼 클릭 함수들
def select_type_1(): return select_object_type("📱 전자기기")
def select_type_2(): return select_object_type("🪑 가구") 
def select_type_3(): return select_object_type("🎨 장식품")
def select_type_4(): return select_object_type("🏠 가전제품")
def select_type_5(): return select_object_type("🔧 도구")
def select_type_6(): return select_object_type("👤 개인용품")

# 성격 상세 정보 탭에서 127개 변수 시각화 기능 추가
def create_personality_details_tab():
    with gr.Tab("성격 상세 정보"):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 127개 성격 변수 요약")
                personality_summary = gr.JSON(label="성격 요약", value={})

            with gr.Column(scale=1):
                gr.Markdown("### 유머 매트릭스")
                humor_chart = gr.Plot(label="유머 스타일 차트")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 매력적 결함")
                attractive_flaws = gr.Dataframe(
                    headers=["결함", "효과"],
                    datatype=["str", "str"],
                    label="매력적 결함"
                )
            
            with gr.Column():
                gr.Markdown("### 모순적 특성")
                contradictions = gr.Dataframe(
                    headers=["모순", "효과"],
                    datatype=["str", "str"],
                    label="모순적 특성"
                )
        
        with gr.Accordion("127개 성격 변수 전체 보기", open=False):
            all_variables = gr.Dataframe(
                headers=["변수명", "점수", "설명"],
                datatype=["str", "number", "str"],
                label="127개 성격 변수"
            )

    return personality_summary, humor_chart, attractive_flaws, contradictions, all_variables

# 유머 매트릭스 시각화 함수 추가
def plot_humor_matrix(humor_data):
    if not humor_data:
        return None
    
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import RegularPolygon
    
    # 데이터 준비
    warmth_vs_wit = humor_data.get("warmth_vs_wit", 50)
    self_vs_observational = humor_data.get("self_vs_observational", 50)
    subtle_vs_expressive = humor_data.get("subtle_vs_expressive", 50)
    
    # 3차원 데이터 정규화 (0~1 범위)
    warmth = warmth_vs_wit / 100
    self_ref = self_vs_observational / 100
    expressive = subtle_vs_expressive / 100
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_aspect('equal')
    
    # 축 설정
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # 삼각형 그리기
    triangle = RegularPolygon((0, 0), 3, radius=1, orientation=0, edgecolor='gray', facecolor='none')
    ax.add_patch(triangle)
    
    # 축 라벨 위치 계산
    angle = np.linspace(0, 2*np.pi, 3, endpoint=False)
    x = 1.1 * np.cos(angle)
    y = 1.1 * np.sin(angle)
    
    # 축 라벨 추가
    labels = ['따뜻함', '자기참조', '표현적']
    opposite_labels = ['재치', '관찰형', '은은함']
    
    for i in range(3):
        ax.text(x[i], y[i], labels[i], ha='center', va='center', fontsize=12)
        ax.text(-x[i]/2, -y[i]/2, opposite_labels[i], ha='center', va='center', fontsize=10, color='gray')
    
    # 내부 가이드라인 그리기
    for j in [0.33, 0.66]:
        inner_triangle = RegularPolygon((0, 0), 3, radius=j, orientation=0, edgecolor='lightgray', facecolor='none', linestyle='--')
        ax.add_patch(inner_triangle)
    
    # 포인트 계산
    # 삼각좌표계 변환 (barycentric coordinates)
    # 각 차원의 값을 삼각형 내부의 점으로 변환
    tx = x[0] * warmth + x[1] * self_ref + x[2] * expressive
    ty = y[0] * warmth + y[1] * self_ref + y[2] * expressive
    
    # 포인트 그리기
    ax.scatter(tx, ty, s=150, color='red', zorder=5)
    
    # 축 제거
    ax.axis('off')
    
    # 제목 추가
    plt.title('유머 스타일 매트릭스', fontsize=14)
    
    return fig

# Main Gradio app
with gr.Blocks(title="놈팽쓰 테스트 앱", theme=theme, css=css) as app:
    # Global state
    current_persona = gr.State(value=None)
    conversation_history = gr.State(value=[])
    analysis_result_state = gr.State(value=None)
    personas_data = gr.State(value=[])
    current_view = gr.State(value="frontend")  # View 상태 추가
    
    gr.Markdown(
    """
    # 🎭 놈팽쓰(MemoryTag): 당신 곁의 사물, 이제 친구가 되다
    
    사물에 영혼을 불어넣어 대화할 수 있는 페르소나 생성 앱입니다.
    
    ## 🧭 이용 프로세스 (6단계)
    **1️⃣ 이미지 업로드** → **2️⃣ 사물 종류 선택** → **3️⃣ 맥락 정보** → **4️⃣ 성격 조정** → **5️⃣ 말투 선택** → **6️⃣ 이름 짓기**
    
    ### ✨ 주요 특징
    - 🎯 **4개 핵심 지표**: 외향성, 감정표현, 에너지, 사고방식만 조정하면 127개 성격 변수 자동 생성
    - 🤖 **AI 유머 추천**: 성격 지표 기반으로 유머 스타일 자동 추천
    - 💬 **실시간 미리보기**: 조정 즉시 대화 스타일 확인 가능
    - 📊 **전문적 분석**: 심리학 기반 과학적 페르소나 생성
    """
    )
    
    with gr.Tabs() as tabs:
        # Tab 1: Soul Awakening - 6단계 프로세스
        with gr.Tab("영혼 깨우기"):
            # 전체 진행률 표시
            with gr.Row():
                progress_bar = gr.HTML("""
                <div style="background: #f0f4ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="margin: 0 0 10px 0;">📊 전체 진행률</h3>
                    <div style="background: #e0e0e0; height: 8px; border-radius: 4px;">
                        <div id="progress-fill" style="background: linear-gradient(90deg, #6366f1, #a855f7); height: 100%; width: 0%; border-radius: 4px; transition: width 0.3s ease;"></div>
                    </div>
                    <p style="margin: 5px 0 0 0; font-size: 14px;" id="progress-text">준비 완료 - 1단계부터 시작하세요</p>
                </div>
                """)
            
            # 메인 콘텐츠 영역
            with gr.Row():
                # 왼쪽: 사용자 인터페이스
                with gr.Column(scale=1):
                    # 1단계: 이미지 업로드
                    with gr.Group() as step1_group:
                        gr.Markdown("### 1️⃣ 이미지 업로드")
                        input_image = gr.Image(type="filepath", label="사물 이미지 업로드")
                        discover_soul_button = gr.Button("영혼 발견하기", variant="primary", size="lg")
                    
                    # 2단계: 사물 종류 선택 (버튼 형태)
                    with gr.Group(visible=False) as step2_group:
                        gr.Markdown("### 2️⃣ 사물 종류 선택")
                        gr.Markdown("**어떤 종류의 사물인가요?**")
                        with gr.Row():
                            object_type_btn1 = gr.Button("📱 전자기기", variant="secondary", size="lg")
                            object_type_btn2 = gr.Button("🪑 가구", variant="secondary", size="lg") 
                            object_type_btn3 = gr.Button("🎨 장식품", variant="secondary", size="lg")
                        with gr.Row():
                            object_type_btn4 = gr.Button("🏠 가전제품", variant="secondary", size="lg")
                            object_type_btn5 = gr.Button("🔧 도구", variant="secondary", size="lg")
                            object_type_btn6 = gr.Button("👤 개인용품", variant="secondary", size="lg")
                        
                        selected_object_type = gr.Markdown("*선택된 종류: 없음*")
                        object_type_state = gr.State(value="")
                        continue_to_step3_button = gr.Button("다음 단계", variant="primary", size="lg", visible=False)
                    
                    # 3단계: 맥락 정보 입력
                    with gr.Group(visible=False) as step3_group:
                        gr.Markdown("### 3️⃣ 맥락 정보 입력")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**주로 어디에 있나요?**")
                                user_input_location = gr.Radio(
                                    choices=["🏠 집", "🏢 사무실", "✈️ 여행 중", "🛍️ 상점", "🏫 학교", "☕ 카페", "🌍 기타"],
                                    label="위치", value="🏠 집"
                                )
                            with gr.Column():
                                gr.Markdown("**얼마나 함께했나요?**")
                                user_input_time = gr.Radio(
                                    choices=["✨ 새것", "📅 몇 개월", "🗓️ 1년 이상", "⏳ 오래됨", "🎪 중고/빈티지"],
                                    label="함께한 시간", value="📅 몇 개월"
                                )
                        
                        create_persona_button = gr.Button("페르소나 생성", variant="primary", size="lg")
                    
                    # 4단계: 성격 조정
                    with gr.Group(visible=False) as step4_group:
                        gr.Markdown("### 4️⃣ 성격 조정")
                        gr.Markdown("**4개 핵심 지표 조정으로 127개 변수 자동 생성**")
                        
                        with gr.Row():
                            with gr.Column():
                                extraversion_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="얼마나 말씀하세요?", info="내성적 ↔ 외향적")
                                emotion_expression_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="감정을 잘 표현하나요?", info="담담함 ↔ 감정 풍부")
                            with gr.Column():
                                energy_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="밝아 만족가요?", info="조용함 ↔ 에너지")
                                thinking_style_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="어떤 방식으로 문제를 풀까요?", info="논리적사고 ↔ 직관적사고")
                        
                        # 자동 추천된 유머 스타일 표시
                        with gr.Row():
                            recommended_humor_display = gr.Markdown("### 🤖 추천 유머 스타일\n*슬라이더를 조정하면 자동으로 추천됩니다*")
                            humor_confidence_display = gr.Markdown("### 📊 추천 신뢰도\n*-*")
                        
                        continue_to_step5_button = gr.Button("다음: 말투 선택", variant="primary", size="lg")
                    
                    # 5단계: 말투 선택
                    with gr.Group(visible=False) as step5_group:
                        gr.Markdown("### 5️⃣ 말투 선택")
                        speech_style_radio = gr.Radio(
                            choices=[
                                "정중한 (~습니다, ~해요)", 
                                "친근한 (~어, ~야)", 
                                "청자기 (~다, ~네)",
                                "귀여운 (~냥, ~닷)",
                                "유쾌한 (~지, ~잖아)",
                                "차분한 (~군요, ~네요)"
                            ],
                            label="말투 스타일",
                            value="친근한 (~어, ~야)"
                        )
                        continue_to_step6_button = gr.Button("다음: 이름 짓기", variant="primary", size="lg")
                    
                    # 6단계: 이름 짓기
                    with gr.Group(visible=False) as step6_group:
                        gr.Markdown("### 6️⃣ 이름 짓기")
                        user_input_name = gr.Textbox(label="이름 입력", placeholder="원하는 이름을 입력하세요")
                        with gr.Row():
                            auto_name_button = gr.Button("AI 추천 이름", variant="secondary")
                            finalize_persona_button = gr.Button("페르소나 완성!", variant="primary", size="lg")
                    
                    # 완료 단계
                    with gr.Group(visible=False) as step7_group:
                        gr.Markdown("### 🎉 페르소나 완성!")
                        with gr.Row():
                            save_persona_button = gr.Button("저장하기", variant="primary")
                            chat_start_button = gr.Button("대화하기", variant="secondary")
                
                # 오른쪽: 백엔드 분석 패널
                with gr.Column(scale=1):
                    gr.Markdown("### 🔬 AI 분석 과정 (실시간)")
                    
                    # 백엔드 분석 상태 표시
                    backend_status = gr.HTML("""
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                        <h4 style="margin: 0 0 10px 0;">🤖 AI 상태</h4>
                        <p style="margin: 0; color: #666;">이미지 업로드를 기다리는 중...</p>
                    </div>
                    """)
                    
                    # 실시간 분석 로그
                    analysis_log = gr.HTML("""
                    <div style="background: #f0f4ff; padding: 15px; border-radius: 8px; max-height: 300px; overflow-y: auto;">
                        <h4 style="margin: 0 0 10px 0;">📝 분석 로그</h4>
                        <div id="log-content" style="font-family: monospace; font-size: 12px; color: #374151;">
                            시스템 준비 완료<br>
                            이미지 분석 엔진 대기 중...<br>
                        </div>
                    </div>
                    """)
                    
                    # 127개 변수 생성 상태
                    variables_status = gr.HTML("""
                    <div style="background: #fff5f5; padding: 15px; border-radius: 8px; margin-top: 15px;">
                        <h4 style="margin: 0 0 10px 0;">🧠 127개 성격 변수</h4>
                        <div style="background: #e0e0e0; height: 6px; border-radius: 3px;">
                            <div id="variables-progress" style="background: #ef4444; height: 100%; width: 0%; border-radius: 3px; transition: width 0.3s ease;"></div>
                        </div>
                        <p style="margin: 5px 0 0 0; font-size: 12px;" id="variables-text">생성 대기 중 (0/127)</p>
                    </div>
                    """)
                    
                    # 성격 특성 실시간 표시
                    personality_live_view = gr.HTML("""
                    <div style="background: #f0fff4; padding: 15px; border-radius: 8px; margin-top: 15px;">
                        <h4 style="margin: 0 0 10px 0;">🎭 성격 특성 (실시간)</h4>
                        <p style="margin: 0; color: #666; font-size: 14px;">페르소나 생성 후 실시간으로 표시됩니다</p>
                    </div>
                    """)
                    
                    # 대화 미리보기
                    with gr.Accordion("💬 대화 미리보기", open=False):
                        preview_chatbot = gr.Chatbot(label="대화 미리보기", height=200)
                        preview_input = gr.Textbox(placeholder="미리보기 대화...", show_label=False)
                        preview_send_btn = gr.Button("전송", size="sm")
            
            # 에러 메시지
            error_message = gr.Markdown("", visible=False)
        
        # Tab 2: Chat
        with gr.Tab("대화하기"):
            with gr.Row():
                with gr.Column(scale=2):
                    # 대화 인터페이스
                    chatbot = gr.Chatbot(label="대화", height=600)
                    with gr.Row():
                        chat_input = gr.Textbox(placeholder="사물과 대화해보세요...", show_label=False)
                        chat_button = gr.Button("전송", variant="primary")
                
                with gr.Column(scale=1):
                    # 현재 페르소나 요약
                    gr.Markdown("### Current Persona")
                    current_persona_info = gr.JSON(label="Basic Info")
                    current_persona_traits = gr.JSON(label="Personality Traits")
                    gr.Markdown("### Communication Style")
                    current_humor_style = gr.Markdown()
                    
                    # 유머 매트릭스 차트 추가
                    humor_chart = gr.Plot(label="Humor Style Chart", visible=True)
                    
                    gr.Markdown("### Attractive Flaws")
                    current_flaws_df = gr.Dataframe(
                        headers=["Flaw", "Effect"],
                        datatype=["str", "str"],
                        label="Attractive Flaws"
                    )
                    gr.Markdown("### Contradictory Traits")
                    current_contradictions_df = gr.Dataframe(
                        headers=["Contradiction", "Effect"],
                        datatype=["str", "str"],
                        label="Contradictory Traits"
                    )
                    with gr.Accordion("127 Personality Variables", open=False):
                        current_all_variables_df = gr.Dataframe(
                            headers=["Variable", "Score", "Description"],
                            datatype=["str", "number", "str"],
                            label="Personality Variables"
                        )
        
        # Tab 3: Persona Management
        with gr.Tab("페르소나 관리"):
            with gr.Row():
                refresh_btn = gr.Button("페르소나 목록 새로고침")
            
            personas_df = gr.Dataframe(
                headers=["이름", "유형", "생성일시", "파일명"],
                datatype=["str", "str", "str", "str"],
                label="저장된 페르소나 목록",
                row_count=10
            )
            
            with gr.Row():
                load_btn = gr.Button("선택한 페르소나 불러오기")
                load_result = gr.Markdown("")
            
            with gr.Row():
                with gr.Column():
                    selected_persona_frontend = gr.HTML("페르소나를 선택해주세요.")
                
                with gr.Column():
                    selected_persona_chart = gr.Image(
                        label="Personality Chart"
                    )
            
            with gr.Accordion("백엔드 상세 정보", open=False):
                selected_persona_backend = gr.HTML("페르소나를 선택해주세요.")
    
    # Event handlers
    
    # 1단계: 영혼 발견하기
    discover_soul_button.click(
        fn=lambda img: (
            gr.update(visible=True) if img else gr.update(visible=False),
            update_progress_bar(1, 6, "1단계 완료 - 영혼 발견됨"),
            update_backend_status("이미지 분석 완료 - 사물 종류 선택 대기 중", "success")
        ),
        inputs=[input_image],
        outputs=[step2_group, progress_bar, backend_status]
    )
    
    # 2단계: 사물 종류 선택 버튼들
    object_type_btn1.click(fn=select_type_1, outputs=[selected_object_type, object_type_state, continue_to_step3_button])
    object_type_btn2.click(fn=select_type_2, outputs=[selected_object_type, object_type_state, continue_to_step3_button])
    object_type_btn3.click(fn=select_type_3, outputs=[selected_object_type, object_type_state, continue_to_step3_button])
    object_type_btn4.click(fn=select_type_4, outputs=[selected_object_type, object_type_state, continue_to_step3_button])
    object_type_btn5.click(fn=select_type_5, outputs=[selected_object_type, object_type_state, continue_to_step3_button])
    object_type_btn6.click(fn=select_type_6, outputs=[selected_object_type, object_type_state, continue_to_step3_button])
    
    # 3단계로 이동
    continue_to_step3_button.click(
        fn=lambda: (
            gr.update(visible=True),
            update_progress_bar(2, 6, "2단계 완료 - 맥락 정보 입력 중")
        ),
        outputs=[step3_group, progress_bar]
    )
    
    # 3단계: 페르소나 생성
    create_persona_button.click(
        fn=lambda img, obj_type, loc, time: (
            create_persona_from_image(img, {
                "object_type": obj_type,
                "location": loc.replace("🏠 ", "").replace("🏢 ", "").replace("✈️ ", "").replace("🛍️ ", "").replace("🏫 ", "").replace("☕ ", "").replace("🌍 ", ""),
                "time_spent": time.replace("✨ ", "").replace("📅 ", "").replace("🗓️ ", "").replace("⏳ ", "").replace("🎪 ", ""),
                "name": ""
            })[0],  # persona만 반환
            gr.update(visible=True),
            update_progress_bar(3, 6, "3단계 완료 - 성격 조정 준비됨"),
            update_backend_status("페르소나 생성 완료 - 127개 변수 생성됨", "success")
        ),
        inputs=[input_image, object_type_state, user_input_location, user_input_time],
        outputs=[current_persona, step4_group, progress_bar, backend_status]
    ).then(
        fn=lambda p: init_persona_preview_chat(p) if p else [],
        inputs=[current_persona],
        outputs=[preview_chatbot]
    )
    
    # 4단계: 성격 조정 - 슬라이더 변경 시 실시간 업데이트
    for slider in [extraversion_slider, emotion_expression_slider, energy_slider, thinking_style_slider]:
        slider.change(
            fn=lambda e, em, en, t, p: (
                update_humor_recommendation(e, em, en, t)[0],  # humor display
                update_humor_recommendation(e, em, en, t)[1],  # confidence display
                refine_persona(p, e, em, en, t)[0] if p else p,  # updated persona
                update_backend_status(f"성격 조정됨: 외향성{e}%, 감정표현{em}%, 에너지{en}%, 사고방식{t}%", "processing")
            ),
            inputs=[extraversion_slider, emotion_expression_slider, energy_slider, thinking_style_slider, current_persona],
            outputs=[recommended_humor_display, humor_confidence_display, current_persona, backend_status]
        )
    
    # 5단계로 이동
    continue_to_step5_button.click(
        fn=lambda: (
            gr.update(visible=True),
            update_progress_bar(4, 6, "4단계 완료 - 말투 선택 중")
        ),
        outputs=[step5_group, progress_bar]
    )
    
    # 6단계로 이동
    continue_to_step6_button.click(
        fn=lambda: (
            gr.update(visible=True),
            update_progress_bar(5, 6, "5단계 완료 - 이름 짓기 중")
        ),
        outputs=[step6_group, progress_bar]
    )
    
    # 페르소나 완성
    finalize_persona_button.click(
        fn=lambda name, p: (
            # 이름 업데이트
            {**p, "기본정보": {**p.get("기본정보", {}), "이름": name}} if p and name else p,
            gr.update(visible=True),
            update_progress_bar(6, 6, "🎉 페르소나 완성! 저장하거나 대화해보세요"),
            update_backend_status(f"페르소나 '{name}' 완성!", "success")
        ),
        inputs=[user_input_name, current_persona],
        outputs=[current_persona, step7_group, progress_bar, backend_status]
    )
    
    # 대화 미리보기
    preview_send_btn.click(
        fn=chat_with_persona,
        inputs=[current_persona, preview_input, preview_chatbot],
        outputs=[preview_chatbot, preview_input]
    )
    
    # 저장 및 완료
    save_persona_button.click(
        fn=save_current_persona,
        inputs=[current_persona],
        outputs=[error_message]
    )
    
    # 대화 탭으로 이동
    chat_start_button.click(
        fn=lambda: gr.update(selected=1),
        outputs=[tabs]
    )
    
    # 기존 이벤트 핸들러들...
    # ... existing code ...

# 기존 함수 업데이트: 현재 페르소나 정보 표시
def update_current_persona_info(current_persona):
    if not current_persona:
        return {}, {}, None, [], [], []
    
    # 기본 정보
    basic_info = {
        "이름": current_persona.get("기본정보", {}).get("이름", "Unknown"),
        "유형": current_persona.get("기본정보", {}).get("유형", "Unknown"),
        "생성일": current_persona.get("기본정보", {}).get("생성일시", "Unknown"),
        "설명": current_persona.get("기본정보", {}).get("설명", "")
    }
    
    # 성격 특성
    personality_traits = {}
    if "성격특성" in current_persona:
        personality_traits = current_persona["성격특성"]
    
    # 성격 요약 정보
    personality_summary = {}
    if "성격요약" in current_persona:
        personality_summary = current_persona["성격요약"]
    elif "성격변수127" in current_persona:
        # 직접 성격 요약 계산
        try:
            variables = current_persona["성격변수127"]
            
            # 카테고리별 평균 계산
            summary = {}
            category_counts = {}
            
            for var_name, value in variables.items():
                category = var_name[0] if var_name and len(var_name) > 0 else "기타"
                
                if category == "W":  # 온기
                    summary["온기"] = summary.get("온기", 0) + value
                    category_counts["온기"] = category_counts.get("온기", 0) + 1
                elif category == "C":  # 능력
                    summary["능력"] = summary.get("능력", 0) + value
                    category_counts["능력"] = category_counts.get("능력", 0) + 1
                elif category == "E":  # 외향성
                    summary["외향성"] = summary.get("외향성", 0) + value
                    category_counts["외향성"] = category_counts.get("외향성", 0) + 1
                elif category == "O":  # 개방성
                    summary["창의성"] = summary.get("창의성", 0) + value
                    category_counts["창의성"] = category_counts.get("창의성", 0) + 1
                elif category == "H":  # 유머
                    summary["유머감각"] = summary.get("유머감각", 0) + value
                    category_counts["유머감각"] = category_counts.get("유머감각", 0) + 1
            
            # 평균 계산
            for category in summary:
                if category_counts[category] > 0:
                    summary[category] = summary[category] / category_counts[category]
                    
            # 기본값 설정 (데이터가 없는 경우)
            if "온기" not in summary:
                summary["온기"] = 50
            if "능력" not in summary:
                summary["능력"] = 50
            if "외향성" not in summary:
                summary["외향성"] = 50
            if "창의성" not in summary:
                summary["창의성"] = 50
            if "유머감각" not in summary:
                summary["유머감각"] = 50
                
            personality_summary = summary
        except Exception as e:
            print(f"성격 요약 계산 오류: {str(e)}")
            personality_summary = {
                "온기": 50,
                "능력": 50,
                "외향성": 50,
                "창의성": 50,
                "유머감각": 50
            }
    
    # 유머 매트릭스 차트
    humor_chart = None
    if "유머매트릭스" in current_persona:
        humor_chart = plot_humor_matrix(current_persona["유머매트릭스"])
    
    # 매력적 결함 데이터프레임
    attractive_flaws_df = get_attractive_flaws_df(current_persona)
    
    # 모순적 특성 데이터프레임
    contradictions_df = get_contradictions_df(current_persona)
    
    # 127개 성격 변수 데이터프레임
    personality_variables_df = get_personality_variables_df(current_persona)
    
    return basic_info, personality_traits, humor_chart, attractive_flaws_df, contradictions_df, personality_variables_df

# 기존 함수 업데이트: 성격 변수 데이터프레임 생성
def get_personality_variables_df(persona):
    if not persona or "성격변수127" not in persona:
        return []
    
    variables = persona["성격변수127"]
    if isinstance(variables, dict):
        rows = []
        for var_name, score in variables.items():
            description = VARIABLE_DESCRIPTIONS.get(var_name, "")
            rows.append([var_name, score, description])
        return rows
    return []

# 기존 함수 업데이트: 매력적 결함 데이터프레임 생성
def get_attractive_flaws_df(persona):
    if not persona or "매력적결함" not in persona:
        return []
    
    flaws = persona["매력적결함"]
    effects = [
        "인간적 매력 +25%",
        "관계 깊이 +30%",
        "공감 유발 +20%"
    ]
    
    return [[flaw, effects[i] if i < len(effects) else "매력 증가"] for i, flaw in enumerate(flaws)]

# 기존 함수 업데이트: 모순적 특성 데이터프레임 생성
def get_contradictions_df(persona):
    if not persona or "모순적특성" not in persona:
        return []
    
    contradictions = persona["모순적특성"]
    effects = [
        "복잡성 +35%",
        "흥미도 +28%"
    ]
    
    return [[contradiction, effects[i] if i < len(effects) else "깊이감 증가"] for i, contradiction in enumerate(contradictions)]

def generate_personality_chart(persona):
    """Generate a radar chart for personality traits"""
    if not persona or "성격특성" not in persona:
        # Return empty image with default PIL
        img = Image.new('RGB', (400, 400), color='white')
        draw = PIL.ImageDraw.Draw(img)
        draw.text((150, 180), "No data", fill='black')
        img_path = os.path.join("data", "temp_chart.png")
        img.save(img_path)
        return img_path
    
    # Get traits
    traits = persona["성격특성"]
    
    # Convert to English labels
    categories = []
    values = []
    for trait_kr, value in traits.items():
        trait_en = ENGLISH_LABELS.get(trait_kr, trait_kr)
        categories.append(trait_en)
        values.append(value)
    
    # Add the first value again to close the loop
    categories.append(categories[0])
    values.append(values[0])
    
    # Convert to radians
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
    
    # Create plot with improved aesthetics
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # 배경 스타일 개선
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    
    # Grid 스타일 개선
    ax.grid(True, color='#e0e0e0', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # 각도 라벨 위치 및 색상 조정
    ax.set_rlabel_position(90)
    ax.tick_params(colors='#6b7280')
    
    # Y축 라벨 제거 및 눈금 표시
    ax.set_yticklabels([])
    ax.set_yticks([20, 40, 60, 80, 100])
    
    # 범위 설정
    ax.set_ylim(0, 100)
    
    # 차트 그리기
    # 1. 채워진 영역
    ax.fill(angles, values, alpha=0.25, color='#6366f1')
    
    # 2. 테두리 선
    ax.plot(angles, values, 'o-', linewidth=2, color='#6366f1')
    
    # 3. 데이터 포인트 강조
    ax.scatter(angles[:-1], values[:-1], s=100, color='#6366f1', edgecolor='white', zorder=10)
    
    # 4. 각 축 설정 - 영어 라벨 사용
    ax.set_thetagrids(angles[:-1] * 180/np.pi, categories[:-1], fontsize=12)
    
    # 제목 추가
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    plt.title(f"{name} Personality Traits", size=16, color='#374151', pad=20, fontweight='bold')
    
    # 저장
    timestamp = int(time.time())
    img_path = os.path.join("data", f"chart_{timestamp}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    return img_path

def save_current_persona(current_persona):
    """Save current persona to a JSON file"""
    if not current_persona:
        return "저장할 페르소나가 없습니다."
    
    try:
        # 깊은 복사를 통해 원본 데이터를 유지
        import copy
        persona_copy = copy.deepcopy(current_persona)
        
        # 저장 불가능한 객체 제거
        keys_to_remove = []
        for key in persona_copy:
            if key in ["personality_profile", "humor_matrix", "_state"] or callable(persona_copy[key]):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            persona_copy.pop(key, None)
        
        # 중첩된 딕셔너리와 리스트 내의 비직렬화 가능 객체 제거
        def clean_data(data):
            if isinstance(data, dict):
                for k in list(data.keys()):
                    if callable(data[k]):
                        del data[k]
                    elif isinstance(data[k], (dict, list)):
                        data[k] = clean_data(data[k])
                return data
            elif isinstance(data, list):
                return [clean_data(item) if isinstance(item, (dict, list)) else item for item in data if not callable(item)]
            else:
                return data
        
        # 데이터 정리
        cleaned_persona = clean_data(persona_copy)
        
        # 최종 검증: JSON 직렬화 가능 여부 확인
        import json
        try:
            json.dumps(cleaned_persona)
        except TypeError as e:
            print(f"JSON 직렬화 오류: {str(e)}")
            # 기본 정보만 유지하고 나머지는 안전한 데이터만 포함
            basic_info = cleaned_persona.get("기본정보", {})
            성격특성 = cleaned_persona.get("성격특성", {})
            매력적결함 = cleaned_persona.get("매력적결함", [])
            모순적특성 = cleaned_persona.get("모순적특성", [])
            
            cleaned_persona = {
                "기본정보": basic_info,
                "성격특성": 성격특성,
                "매력적결함": 매력적결함,
                "모순적특성": 모순적특성
            }
        
        filepath = save_persona(cleaned_persona)
        if filepath:
            name = current_persona.get("기본정보", {}).get("이름", "Unknown")
            return f"{name} 페르소나가 저장되었습니다: {filepath}"
        else:
            return "페르소나 저장에 실패했습니다."
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"저장 오류 상세: {error_details}")
        return f"저장 중 오류 발생: {str(e)}"

# 이 함수는 파일 상단에서 이미 정의되어 있으므로 여기서는 제거합니다.

# 성격 미세조정 함수
def refine_persona(persona, extraversion, emotion_expression, energy, thinking_style):
    """페르소나의 성격을 미세조정하는 함수"""
    if not persona:
        return persona, "페르소나가 없습니다."
    
    try:
        # 유머 스타일 자동 추천
        humor_style, confidence, scores = recommend_humor_style(extraversion, emotion_expression, energy, thinking_style)
        
        # 복사본 생성
        refined_persona = persona.copy()
        
        # 성격 특성 업데이트 - 새로운 지표들을 기존 매핑에 연결
        if "성격특성" in refined_persona:
            refined_persona["성격특성"]["외향성"] = int(extraversion)
            refined_persona["성격특성"]["감정표현"] = int(emotion_expression)  
            refined_persona["성격특성"]["활력"] = int(energy)
            refined_persona["성격특성"]["사고방식"] = int(thinking_style)
            
            # 기존 특성들도 새로운 지표를 바탕으로 계산
            refined_persona["성격특성"]["온기"] = int((emotion_expression + energy) / 2)
            refined_persona["성격특성"]["능력"] = int(thinking_style)
            refined_persona["성격특성"]["창의성"] = int(100 - thinking_style)  # 논리적 ↔ 창의적
        
        # 자동 추천된 유머 스타일 업데이트
        refined_persona["유머스타일"] = humor_style
        
        # 127개 성격 변수가 있으면 업데이트
        if "성격변수127" in refined_persona:
            # 외향성 관련 변수 업데이트
            for var in ["E01_사교성", "E02_활동성", "E03_자기주장", "E06_열정성"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(extraversion * 0.9 + random.randint(0, 20))
            
            # 감정표현 관련 변수 업데이트
            for var in ["W09_친밀감표현", "W06_공감능력", "E04_긍정정서"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(emotion_expression * 0.9 + random.randint(0, 20))
            
            # 에너지 관련 변수 업데이트
            for var in ["E02_활동성", "E06_열정성", "E05_자극추구"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(energy * 0.9 + random.randint(0, 20))
            
            # 사고방식 관련 변수 업데이트
            for var in ["C02_지능", "C06_분석력", "C01_효율성"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(thinking_style * 0.9 + random.randint(0, 20))
            
            # 창의성 관련 변수 업데이트 (논리적 사고와 반대)
            for var in ["C04_창의성", "C08_통찰력"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int((100 - thinking_style) * 0.9 + random.randint(0, 20))
        
        # 유머 매트릭스 업데이트
        if "유머매트릭스" in refined_persona:
            if humor_style == "위트있는 재치꾼":
                refined_persona["유머매트릭스"]["warmth_vs_wit"] = 30
                refined_persona["유머매트릭스"]["self_vs_observational"] = 50
                refined_persona["유머매트릭스"]["subtle_vs_expressive"] = 70
            elif humor_style == "따뜻한 유머러스":
                refined_persona["유머매트릭스"]["warmth_vs_wit"] = 80
                refined_persona["유머매트릭스"]["self_vs_observational"] = 60
                refined_persona["유머매트릭스"]["subtle_vs_expressive"] = 60
            elif humor_style == "날카로운 관찰자":
                refined_persona["유머매트릭스"]["warmth_vs_wit"] = 40
                refined_persona["유머매트릭스"]["self_vs_observational"] = 20
                refined_persona["유머매트릭스"]["subtle_vs_expressive"] = 50
            elif humor_style == "자기 비하적":
                refined_persona["유머매트릭스"]["warmth_vs_wit"] = 60
                refined_persona["유머매트릭스"]["self_vs_observational"] = 85
                refined_persona["유머매트릭스"]["subtle_vs_expressive"] = 40
        
        return refined_persona, "성격이 성공적으로 미세조정되었습니다."
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"성격 미세조정 오류: {error_details}")
        return persona, f"성격 미세조정 중 오류가 발생했습니다: {str(e)}"

def create_frontend_view_html(persona):
    """Create HTML representation of the frontend view of the persona"""
    if not persona:
        return "<div class='persona-details'>페르소나가 아직 생성되지 않았습니다.</div>"
    
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    object_type = persona.get("기본정보", {}).get("유형", "Unknown")
    description = persona.get("기본정보", {}).get("설명", "")
    
    # 성격 요약 가져오기
    personality_summary = persona.get("성격요약", {})
    summary_html = ""
    if personality_summary:
        summary_items = []
        for trait, value in personality_summary.items():
            if isinstance(value, (int, float)):
                trait_name = trait
                trait_value = value
                summary_items.append(f"• {trait_name}: {trait_value:.1f}%")
        
        if summary_items:
            summary_html = "<div class='summary-section'><h4>성격 요약</h4><ul>" + "".join([f"<li>{item}</li>" for item in summary_items]) + "</ul></div>"
    
    # Personality traits
    traits_html = ""
    for trait, value in persona.get("성격특성", {}).items():
        traits_html += f"""
        <div class="trait-item">
            <div class="trait-label">{trait}</div>
            <div class="trait-bar-container">
                <div class="trait-bar" style="width: {value}%; background: linear-gradient(90deg, #6366f1, #a5b4fc);"></div>
            </div>
            <div class="trait-value">{value}%</div>
        </div>
        """
    
    # Flaws - 매력적 결함
    flaws = persona.get("매력적결함", [])
    flaws_list = ""
    for flaw in flaws[:4]:  # 최대 4개만 표시
        flaws_list += f"<li>{flaw}</li>"
    
    # 소통 방식
    communication_style = persona.get("소통방식", "")
    
    # 유머 스타일
    humor_style = persona.get("유머스타일", "")
    
    # 전체 HTML 스타일과 내용
    html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    .frontend-persona {{
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
        max-width: 100%;
    }}
    
    .persona-header {{
        background: linear-gradient(135deg, #6366f1, #a5b4fc);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .persona-header h2 {{
        margin: 0;
        font-size: 24px;
    }}
    
    .persona-header p {{
        margin: 5px 0 0 0;
        opacity: 0.9;
    }}
    
    .persona-section {{
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }}
    
    .section-title {{
        font-size: 18px;
        margin: 0 0 10px 0;
        color: #444;
        border-bottom: 2px solid #6366f1;
        padding-bottom: 5px;
        display: inline-block;
    }}
    
    .trait-item {{
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }}
    
    .trait-label {{
        width: 80px;
        font-weight: 500;
    }}
    
    .trait-bar-container {{
        flex-grow: 1;
        background: #e0e0e0;
        height: 10px;
        border-radius: 5px;
        margin: 0 10px;
        overflow: hidden;
    }}
    
    .trait-bar {{
        height: 100%;
        border-radius: 5px;
    }}
    
    .trait-value {{
        width: 40px;
        text-align: right;
        font-size: 14px;
    }}
    
    .tags-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }}
    
    .flaw-tag, .contradiction-tag, .interest-tag {{
        background: #f0f4ff;
        border: 1px solid #d0d4ff;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 14px;
        display: inline-block;
    }}
    
    .flaw-tag {{
        background: #fff0f0;
        border-color: #ffd0d0;
    }}
    
    .contradiction-tag {{
        background: #f0fff4;
        border-color: #d0ffd4;
    }}
    
    /* 영혼 각성 UX 스타일 */
    .awakening-result {{
        background: #f9f9ff;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }}
    
    .speech-bubble {{
        background: #fff;
        border-radius: 18px;
        padding: 15px;
        margin-bottom: 15px;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }}
    
    .speech-bubble:after {{
        content: '';
        position: absolute;
        bottom: -10px;
        left: 30px;
        border-width: 10px 10px 0;
        border-style: solid;
        border-color: #fff transparent;
    }}
    
    .persona-speech {{
        margin: 0;
        font-size: 15px;
        line-height: 1.5;
        color: #4b5563;
    }}
    
    .persona-traits-highlight {{
        background: #f0f4ff;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }}
    
    .persona-traits-highlight h4 {{
        margin-top: 0;
        margin-bottom: 10px;
        color: #4338ca;
    }}
    
    .persona-traits-highlight ul {{
        margin: 0;
        padding-left: 20px;
        color: #4b5563;
    }}
    
    .persona-traits-highlight li {{
        margin-bottom: 5px;
    }}
    
    .first-interaction {{
        margin-top: 20px;
    }}
    
    .interaction-buttons, .confirmation-buttons {{
        display: flex;
        gap: 10px;
        margin-top: 15px;
    }}
    
    .interaction-btn, .confirmation-btn {{
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
        font-family: 'Noto Sans KR', sans-serif;
    }}
    
    .interaction-btn:hover, .confirmation-btn:hover {{
        background: #e5e7eb;
    }}
    
    .confirmation-btn.primary {{
        background: #6366f1;
        color: white;
        border: 1px solid #4f46e5;
    }}
    
    .confirmation-btn.primary:hover {{
        background: #4f46e5;
    }}
    
    /* 요약 섹션 스타일 */
    .summary-section {{
        background: #f0f4ff;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }}
    
    .summary-section h4 {{
        margin-top: 0;
        margin-bottom: 10px;
        color: #4338ca;
    }}
    
    .summary-section ul {{
        margin: 0;
        padding-left: 20px;
        color: #4b5563;
    }}
    
    .summary-section li {{
        margin-bottom: 5px;
    }}
    </style>
    
    <div class="frontend-persona">
        <div class="persona-header">
            <h2>{name}</h2>
            <p><strong>{object_type}</strong> - {description}</p>
        </div>
        
        {summary_html}
        
        <div class="persona-section">
            <h3 class="section-title">성격 특성</h3>
            <div class="traits-container">
                {traits_html}
            </div>
        </div>
        
        <div class="persona-section">
            <h3 class="section-title">소통 스타일</h3>
            <p>{communication_style}</p>
            <h3 class="section-title" style="margin-top: 15px;">유머 스타일</h3>
            <p>{humor_style}</p>
        </div>
        
        <div class="persona-section">
            <h3 class="section-title">매력적 결함</h3>
            <ul class="flaws-list">
                {flaws_list}
            </ul>
        </div>
    </div>
    """
    
    return html

def create_backend_view_html(persona):
    """Create HTML representation of the backend view of the persona"""
    if not persona:
        return "<div class='persona-details'>페르소나가 아직 생성되지 않았습니다.</div>"
    
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    
    # 백엔드 기본 정보
    basic_info = persona.get("기본정보", {})
    basic_info_html = ""
    for key, value in basic_info.items():
        basic_info_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
    
    # 1. 성격 변수 요약
    personality_summary = persona.get("성격요약", {})
    summary_html = ""
    
    if personality_summary:
        summary_html += "<div class='summary-container'>"
        for category, value in personality_summary.items():
            if isinstance(value, (int, float)):
                summary_html += f"""
                <div class='summary-item'>
                    <div class='summary-label'>{category}</div>
                    <div class='summary-bar-container'>
                        <div class='summary-bar' style='width: {value}%; background: linear-gradient(90deg, #10b981, #6ee7b7);'></div>
                    </div>
                    <div class='summary-value'>{value:.1f}</div>
                </div>
                """
        summary_html += "</div>"
    
    # 2. 성격 매트릭스 (5차원 빅5 시각화)
    big5_html = ""
    if "성격특성" in persona:
        # 빅5 매핑 (기존 특성에서 변환)
        big5 = {
            "외향성(Extraversion)": persona.get("성격특성", {}).get("외향성", 50),
            "친화성(Agreeableness)": persona.get("성격특성", {}).get("온기", 50),
            "성실성(Conscientiousness)": persona.get("성격특성", {}).get("신뢰성", 50),
            "신경증(Neuroticism)": 100 - persona.get("성격특성", {}).get("안정성", 50) if "안정성" in persona.get("성격특성", {}) else 50,
            "개방성(Openness)": persona.get("성격특성", {}).get("창의성", 50)
        }
        
        big5_html = "<div class='big5-matrix'>"
        for trait, value in big5.items():
            big5_html += f"""
            <div class='big5-item'>
                <div class='big5-label'>{trait}</div>
                <div class='big5-bar-container'>
                    <div class='big5-bar' style='width: {value}%;'></div>
                </div>
                <div class='big5-value'>{value}%</div>
            </div>
            """
        big5_html += "</div>"
    
    # 3. 유머 매트릭스
    humor_matrix = persona.get("유머매트릭스", {})
    humor_html = ""
    
    if humor_matrix:
        warmth_vs_wit = humor_matrix.get("warmth_vs_wit", 50)
        self_vs_observational = humor_matrix.get("self_vs_observational", 50)
        subtle_vs_expressive = humor_matrix.get("subtle_vs_expressive", 50)
        
        humor_html = f"""
        <div class='humor-matrix'>
            <div class='humor-dimension'>
                <div class='dimension-label'>따뜻함 vs 위트</div>
                <div class='dimension-bar-container'>
                    <div class='dimension-indicator' style='left: {warmth_vs_wit}%;'></div>
                    <div class='dimension-label-left'>위트</div>
                    <div class='dimension-label-right'>따뜻함</div>
                </div>
            </div>
            
            <div class='humor-dimension'>
                <div class='dimension-label'>자기참조 vs 관찰형</div>
                <div class='dimension-bar-container'>
                    <div class='dimension-indicator' style='left: {self_vs_observational}%;'></div>
                    <div class='dimension-label-left'>관찰형</div>
                    <div class='dimension-label-right'>자기참조</div>
                </div>
            </div>
            
            <div class='humor-dimension'>
                <div class='dimension-label'>미묘함 vs 표현적</div>
                <div class='dimension-bar-container'>
                    <div class='dimension-indicator' style='left: {subtle_vs_expressive}%;'></div>
                    <div class='dimension-label-left'>미묘함</div>
                    <div class='dimension-label-right'>표현적</div>
                </div>
            </div>
        </div>
        """
    
    # 4. 매력적 결함과 모순적 특성
    flaws_html = ""
    contradictions_html = ""
    
    flaws = persona.get("매력적결함", [])
    if flaws:
        flaws_html = "<ul class='flaws-list'>"
        for flaw in flaws:
            flaws_html += f"<li>{flaw}</li>"
        flaws_html += "</ul>"
    
    contradictions = persona.get("모순적특성", [])
    if contradictions:
        contradictions_html = "<ul class='contradictions-list'>"
        for contradiction in contradictions:
            contradictions_html += f"<li>{contradiction}</li>"
        contradictions_html += "</ul>"
    
    # 6. 프롬프트 템플릿 (있는 경우)
    prompt_html = ""
    if "프롬프트" in persona:
        prompt_text = persona.get("프롬프트", "")
        prompt_html = f"""
        <div class='prompt-section'>
            <h3 class='section-title'>대화 프롬프트</h3>
            <pre class='prompt-text'>{prompt_text}</pre>
        </div>
        """
    
    # 7. 완전한 백엔드 JSON (접이식)
    try:
        # 내부 상태 객체 제거 (JSON 변환 불가)
        json_persona = {k: v for k, v in persona.items() if k not in ["personality_profile", "humor_matrix"]}
        persona_json = json.dumps(json_persona, ensure_ascii=False, indent=2)
        
        json_preview = f"""
        <details class='json-details'>
            <summary>전체 백엔드 데이터 (JSON)</summary>
            <pre class='json-preview'>{persona_json}</pre>
        </details>
        """
    except Exception as e:
        json_preview = f"<div class='error'>JSON 변환 오류: {str(e)}</div>"
    
    # 8. 전체 HTML 조합
    html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    .backend-persona {{
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
        max-width: 100%;
    }}
    
    .backend-header {{
        background: linear-gradient(135deg, #059669, #34d399);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .backend-header h2 {{
        margin: 0;
        font-size: 24px;
    }}
    
    .backend-header p {{
        margin: 5px 0 0 0;
        opacity: 0.9;
    }}
    
    .backend-section {{
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }}
    
    .section-title {{
        font-size: 18px;
        margin: 0 0 10px 0;
        color: #444;
        border-bottom: 2px solid #10b981;
        padding-bottom: 5px;
        display: inline-block;
    }}
    
    /* 기본 정보 테이블 */
    .basic-info-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    .basic-info-table td {{
        padding: 8px;
        border-bottom: 1px solid #e0e0e0;
    }}
    
    .basic-info-table td:first-child {{
        width: 120px;
        font-weight: 500;
    }}
    
    /* 요약 스타일 */
    .summary-container {{
        margin-top: 10px;
    }}
    
    .summary-item {{
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }}
    
    .summary-label {{
        width: 150px;
        font-weight: 500;
    }}
    
    .summary-bar-container {{
        flex-grow: 1;
        background: #e0e0e0;
        height: 10px;
        border-radius: 5px;
        margin: 0 10px;
        overflow: hidden;
    }}
    
    .summary-bar {{
        height: 100%;
        border-radius: 5px;
    }}
    
    .summary-value {{
        width: 40px;
        text-align: right;
        font-size: 14px;
    }}
    
    /* 빅5 성격 매트릭스 */
    .big5-matrix {{
        margin-top: 15px;
    }}
    
    .big5-item {{
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }}
    
    .big5-label {{
        width: 150px;
        font-weight: 500;
    }}
    
    .big5-bar-container {{
        flex-grow: 1;
        background: #e0e0e0;
        height: 12px;
        border-radius: 6px;
        margin: 0 10px;
        overflow: hidden;
    }}
    
    .big5-bar {{
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #10b981, #34d399);
    }}
    
    .big5-value {{
        width: 40px;
        text-align: right;
        font-weight: 500;
    }}
    
    /* 유머 매트릭스 스타일 */
    .humor-matrix {{
        margin-top: 15px;
    }}
    
    .humor-dimension {{
        margin-bottom: 20px;
    }}
    
    .dimension-label {{
        font-weight: 500;
        margin-bottom: 5px;
    }}
    
    .dimension-bar-container {{
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        position: relative;
        margin-top: 5px;
    }}
    
    .dimension-indicator {{
        width: 20px;
        height: 20px;
        background: #10b981;
        border-radius: 50%;
        position: absolute;
        top: 0;
        transform: translateX(-50%);
    }}
    
    .dimension-label-left, .dimension-label-right {{
        position: absolute;
        top: -20px;
        font-size: 12px;
        color: #666;
    }}
    
    .dimension-label-left {{
        left: 10px;
    }}
    
    .dimension-label-right {{
        right: 10px;
    }}
    
    /* 매력적 결함 및 모순적 특성 */
    .flaws-list, .contradictions-list {{
        margin: 0;
        padding-left: 20px;
    }}
    
    .flaws-list li, .contradictions-list li {{
        margin-bottom: 6px;
    }}
    
    /* 프롬프트 섹션 */
    .prompt-text {{
        background: #f3f4f6;
        border-radius: 6px;
        padding: 15px;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 14px;
        color: #374151;
        max-height: 400px;
        overflow-y: auto;
    }}
    
    /* JSON 미리보기 스타일 */
    .json-details {{
        margin-top: 15px;
    }}
    
    .json-details summary {{
        cursor: pointer;
        padding: 10px;
        background: #f0f0f0;
        border-radius: 5px;
        font-weight: 500;
    }}
    
    .json-preview {{
        background: #f8f8f8;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-top: 10px;
        overflow-x: auto;
        color: #333;
        font-family: monospace;
        font-size: 14px;
        line-height: 1.5;
        max-height: 400px;
        overflow-y: auto;
    }}
    
    .error {{
        color: #e53e3e;
        padding: 10px;
        background: #fff5f5;
        border-radius: 5px;
        margin-top: 10px;
    }}
    </style>
    
    <div class="backend-persona">
        <div class="backend-header">
            <h2>{name} - 백엔드 데이터</h2>
            <p>상세 정보와 내부 변수 확인</p>
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">기본 정보</h3>
            <table class="basic-info-table">
                {basic_info_html}
            </table>
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">성격 요약 (Big 5)</h3>
            {big5_html}
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">유머 매트릭스 (3차원)</h3>
            {humor_html}
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">매력적 결함</h3>
            {flaws_html}
            
            <h3 class="section-title" style="margin-top: 20px;">모순적 특성</h3>
            {contradictions_html}
        </div>
        
        {prompt_html}
        
        <div class="backend-section">
            <h3 class="section-title">전체 백엔드 데이터</h3>
            {json_preview}
        </div>
    </div>
    """
    
    return html

def get_personas_list():
    """Get list of personas for the dataframe"""
    personas = list_personas()
    
    # Convert to dataframe format
    df_data = []
    for i, persona in enumerate(personas):
        df_data.append([
            persona["name"],
            persona["type"],
            persona["created_at"],
            persona["filename"]
        ])
    
    return df_data, personas

def load_selected_persona(selected_row, personas_list):
    """Load persona from the selected row in the dataframe"""
    if selected_row is None or len(selected_row) == 0:
        return None, "선택된 페르소나가 없습니다.", None, None, None
    
    try:
        # Get filepath from selected row
        selected_index = selected_row.index[0] if hasattr(selected_row, 'index') else 0
        filepath = personas_list[selected_index]["filepath"]
        
        # Load persona
        persona = load_persona(filepath)
        if not persona:
            return None, "페르소나 로딩에 실패했습니다.", None, None, None
        
        # Generate HTML views
        frontend_view, backend_view = toggle_frontend_backend_view(persona)
        frontend_html = create_frontend_view_html(frontend_view)
        backend_html = create_backend_view_html(backend_view)
        
        # Generate personality chart
        chart_image_path = generate_personality_chart(frontend_view)
        
        return persona, f"{persona['기본정보']['이름']}을(를) 로드했습니다.", frontend_html, backend_html, chart_image_path
    
    except Exception as e:
        return None, f"페르소나 로딩 중 오류 발생: {str(e)}", None, None, None

# 페르소나와 대화하는 함수 추가
def chat_with_persona(persona, user_message, chat_history=None):
    """
    페르소나와 대화하는 함수
    """
    if chat_history is None:
        chat_history = []
        
    if not user_message.strip():
        return chat_history, ""
        
    if not persona:
        # Gradio 4.x 호환 메시지 형식 (튜플)
        chat_history.append([user_message, "페르소나가 로드되지 않았습니다. 먼저 페르소나를 생성하거나 불러오세요."])
        return chat_history, ""
    
    try:
        # 페르소나 생성기에서 대화 기능 호출
        # 이전 대화 기록 변환 필요 - 리스트에서 튜플 형식으로
        converted_history = []
        for msg in chat_history:
            if isinstance(msg, list) and len(msg) == 2:
                # 리스트 형식이면 튜플로 변환
                converted_history.append((msg[0] if msg[0] else "", msg[1] if msg[1] else ""))
            elif isinstance(msg, tuple) and len(msg) == 2:
                # 이미 튜플 형식이면 그대로 사용
                converted_history.append(msg)
        
        # 페르소나 생성기에서 대화 함수 호출
        response = persona_generator.chat_with_persona(persona, user_message, converted_history)
        
        # Gradio 4.x 메시지 형식으로 추가 (리스트)
        chat_history.append([user_message, response])
        
        return chat_history, ""
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"대화 오류: {error_details}")
        chat_history.append([user_message, f"대화 중 오류가 발생했습니다: {str(e)}"])
        return chat_history, ""

# 메인 Gradio 인터페이스 구성 함수
def create_interface():
    # 현재 persona 상태 저장 - Gradio 5.x에서 변경된 방식 적용
    current_persona = gr.State(value=None)
    personas_list = gr.State(value=[])
    
    with gr.Blocks(theme=theme, css=css) as app:
        gr.Markdown("""
        # 놈팽쓰(MemoryTag): 당신 곁의 사물, 이제 친구가 되다
        이 데모는 일상 속 사물에 AI 페르소나를 부여하여 대화할 수 있게 해주는 서비스입니다.
        """)
        
        with gr.Tabs() as tabs:
            with gr.Tab("페르소나 생성", id="persona_creation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 이미지 업로드 영역
                        image_input = gr.Image(
                            type="pil", 
                            width=300, 
                            height=300, 
                            label="사물 이미지를 업로드하세요"
                        )
                        # 입력 필드들
                        with gr.Group():
                            gr.Markdown("### 맥락 정보 입력")
                            name_input = gr.Textbox(label="사물 이름 (빈칸일 경우 자동 생성)", placeholder="예: 책상 위 램프")
                            
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
                        
                        # 사용자 입력들 상태 저장 - Gradio 5.x에서 변경된 방식 적용
                        user_inputs = gr.State(value={})
                        
                        with gr.Row():
                            discover_btn = gr.Button("1. 영혼 발견하기", variant="primary")
                            create_btn = gr.Button("2. 페르소나 생성", variant="secondary")
                            
                        # 영혼 깨우기 결과 표시 영역
                        awakening_output = gr.HTML(visible=False)
                        error_output = gr.Markdown(visible=False)
                    
                    with gr.Column(scale=1):
                        # 이미지 분석 결과
                        image_analysis_output = gr.JSON(label="이미지 분석 결과", visible=False)
                        # 페르소나 기본 정보 및 특성
                        basic_info_output = gr.JSON(label="기본 정보")
                        personality_traits_output = gr.JSON(label="페르소나 특성")
                        
                        # 페르소나 저장 및 내보내기 버튼
                        with gr.Row():
                            save_btn = gr.Button("페르소나 저장", variant="primary")
                            download_btn = gr.Button("JSON으로 내보내기", variant="secondary")
                        
                        # 성향 미세조정
                        with gr.Accordion("성향 미세조정", open=False):
                            with gr.Row():
                                with gr.Column(scale=1):
                                    warmth_slider = gr.Slider(0, 100, label="온기", step=1)
                                    competence_slider = gr.Slider(0, 100, label="능력", step=1)
                                    creativity_slider = gr.Slider(0, 100, label="창의성", step=1)
                                with gr.Column(scale=1):
                                    extraversion_slider = gr.Slider(0, 100, label="외향성", step=1)
                                    humor_slider = gr.Slider(0, 100, label="유머감각", step=1)
                                    trust_slider = gr.Slider(0, 100, label="신뢰도", step=1)
                                    
                            humor_style = gr.Dropdown(
                                choices=["witty_wordsmith", "warm_humorist", "playful_trickster", "sharp_observer", "self_deprecating"],
                                label="유머 스타일",
                                value="warm_humorist"
                            )
                            apply_traits_btn = gr.Button("성향 적용하기")
                
                # 유머 스타일 시각화
                humor_chart_output = gr.Plot(label="유머 스타일 매트릭스")
                
                # 페르소나 다운로드 관련 출력
                json_output = gr.Textbox(label="JSON 데이터", visible=False)
                download_output = gr.File(label="다운로드", visible=False)
                
            with gr.Tab("세부 정보", id="persona_details"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 매력적 결함 데이터프레임
                        attractive_flaws_df_output = gr.Dataframe(
                            headers=["매력적 결함", "효과"],
                            label="매력적 결함",
                            interactive=False
                        )
                        
                        # 모순적 특성 데이터프레임
                        contradictions_df_output = gr.Dataframe(
                            headers=["모순적 특성", "효과"],
                            label="모순적 특성",
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        # 성격 차트
                        personality_chart_output = gr.Plot(label="성격 차트")
                
                # 127개 성격 변수 데이터프레임
                with gr.Accordion("127개 성격 변수 세부정보", open=False):
                    personality_variables_df_output = gr.Dataframe(
                        headers=["변수", "값", "설명"],
                        label="성격 변수 (127개)",
                        interactive=False
                    )
            
            with gr.Tab("대화하기", id="persona_chat"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 페르소나 불러오기 기능
                        gr.Markdown("### 페르소나 불러오기")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                # 저장된 페르소나 목록
                                refresh_personas_btn = gr.Button("목록 새로고침", variant="secondary")
                                persona_table = gr.Dataframe(
                                    headers=["ID", "이름", "유형", "생성 날짜"],
                                    label="저장된 페르소나",
                                    interactive=False
                                )
                                load_persona_btn = gr.Button("선택한 페르소나 불러오기", variant="primary")
                            
                            with gr.Column(scale=1):
                                # JSON 파일에서 불러오기
                                gr.Markdown("### 또는 JSON 파일에서 불러오기")
                                json_upload = gr.File(
                                    label="페르소나 JSON 파일 업로드",
                                    file_types=[".json"]
                                )
                                import_persona_btn = gr.Button("JSON에서 가져오기", variant="primary")
                                import_status = gr.Markdown("")
                        
                    with gr.Column(scale=1):
                        # 현재 로드된 페르소나 정보
                        chat_persona_info = gr.Markdown("### 페르소나를 불러와 대화를 시작하세요")
                        
                        # 대화 인터페이스
                        chatbot = gr.Chatbot(height=400, label="대화")
                        with gr.Row():
                            message_input = gr.Textbox(
                                placeholder="메시지를 입력하세요...",
                                label="메시지",
                                show_label=False, 
                                lines=2
                            )
                            send_btn = gr.Button("전송", variant="primary")
        
        # 영혼 깨우기 버튼 이벤트
        discover_btn.click(
            fn=lambda name, location, time_spent, object_type: {"name": name, "location": location, "time_spent": time_spent, "object_type": object_type},
            inputs=[name_input, location_input, time_spent_input, object_type_input],
            outputs=[user_inputs],
            queue=False
        ).then(
            fn=show_awakening_progress,
            inputs=[image_input, user_inputs],
            outputs=[current_persona, error_output, awakening_output],
            queue=True
        )
        
        # 페르소나 생성 버튼 이벤트
        create_btn.click(
            fn=lambda name, location, time_spent, object_type: {"name": name, "location": location, "time_spent": time_spent, "object_type": object_type},
            inputs=[name_input, location_input, time_spent_input, object_type_input],
            outputs=[user_inputs],
            queue=False
        ).then(
            fn=create_persona_from_image,
            inputs=[image_input, user_inputs],
            outputs=[
                current_persona, error_output, image_input, image_analysis_output,
                basic_info_output, personality_traits_output, humor_chart_output,
                attractive_flaws_df_output, contradictions_df_output, personality_variables_df_output
            ],
            queue=True
        ).then(
            fn=generate_personality_chart,
            inputs=[current_persona],
            outputs=[personality_chart_output]
        ).then(
            fn=lambda: gr.update(visible=False),
            outputs=[awakening_output]
        ).then(
            fn=lambda persona: [
                50, 50, 50, 50, 50, 50  # 기본값
            ],
            inputs=[current_persona],
            outputs=[warmth_slider, competence_slider, creativity_slider, extraversion_slider, humor_slider, trust_slider]
        )
        
        # 성향 미세조정 이벤트
        apply_traits_btn.click(
            fn=refine_persona,
            inputs=[
                current_persona, warmth_slider, competence_slider, creativity_slider, 
                extraversion_slider, humor_slider, trust_slider, humor_style
            ],
            outputs=[
                current_persona, basic_info_output, personality_traits_output, 
                humor_chart_output, personality_chart_output, personality_variables_df_output
            ]
        )
        
        # 페르소나 저장 버튼 이벤트
        save_btn.click(
            fn=save_current_persona,
            inputs=[current_persona],
            outputs=[error_output]
        )
        
        # 페르소나 JSON 내보내기 버튼 이벤트
        download_btn.click(
            fn=export_persona_json,
            inputs=[current_persona],
            outputs=[download_output, json_output]
        ).then(
            fn=lambda x: gr.update(visible=True if x else False),
            inputs=[download_output],
            outputs=[download_output]
        ).then(
            fn=lambda x: gr.update(visible=False),
            inputs=[json_output],
            outputs=[json_output]
        )
        
        # 저장된 페르소나 목록 새로고침 이벤트
        refresh_personas_btn.click(
            fn=get_personas_list,
            outputs=[persona_table, personas_list]
        )
        
        # 저장된 페르소나 불러오기 이벤트
        load_persona_btn.click(
            fn=load_selected_persona,
            inputs=[persona_table, personas_list],
            outputs=[
                current_persona, chat_persona_info, chatbot,
                basic_info_output, personality_traits_output, humor_chart_output,
                attractive_flaws_df_output, contradictions_df_output, personality_variables_df_output
            ]
        ).then(
            fn=generate_personality_chart,
            inputs=[current_persona],
            outputs=[personality_chart_output]
        ).then(
            fn=lambda persona: [50, 50, 50, 50, 50, 50],  # 기본값
            inputs=[current_persona],
            outputs=[warmth_slider, competence_slider, creativity_slider, extraversion_slider, humor_slider, trust_slider]
        ).then(
            fn=lambda: gr.update(selected="persona_creation"),
            outputs=[tabs]
        )
        
        # JSON에서 페르소나 가져오기 이벤트
        import_persona_btn.click(
            fn=import_persona_json,
            inputs=[json_upload],
            outputs=[current_persona, import_status]
        ).then(
            fn=lambda persona: update_current_persona_info(persona) if persona else (None, None, None, [], [], []),
            inputs=[current_persona],
            outputs=[
                basic_info_output, personality_traits_output, humor_chart_output,
                attractive_flaws_df_output, contradictions_df_output, personality_variables_df_output
            ]
        ).then(
            fn=generate_personality_chart,
            inputs=[current_persona],
            outputs=[personality_chart_output]
        ).then(
            fn=lambda persona: [50, 50, 50, 50, 50, 50],  # 기본값
            inputs=[current_persona],
            outputs=[warmth_slider, competence_slider, creativity_slider, extraversion_slider, humor_slider, trust_slider]
        ).then(
            fn=lambda persona: f"### 페르소나를 불러왔습니다" if persona else "### 페르소나를 불러오지 못했습니다",
            inputs=[current_persona],
            outputs=[chat_persona_info]
        ).then(
            fn=lambda: gr.update(selected="persona_creation"),
            outputs=[tabs]
        )
        
        # 메시지 전송 이벤트
        send_btn.click(
            fn=chat_with_persona,
            inputs=[current_persona, message_input, chatbot],
            outputs=[chatbot, message_input]
        )
        message_input.submit(
            fn=chat_with_persona,
            inputs=[current_persona, message_input, chatbot],
            outputs=[chatbot, message_input]
        )
        
        # 앱 로드 시 저장된 페르소나 목록 로드
        app.load(
            fn=get_personas_list,
            outputs=[persona_table, personas_list]
        )
    
    return app

# 메인 실행 부분
if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860) 