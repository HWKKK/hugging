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

# Create data directories
os.makedirs("data/personas", exist_ok=True)
os.makedirs("data/conversations", exist_ok=True)

# Initialize the persona generator
persona_generator = PersonaGenerator()

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

def create_persona_from_image(image, name, location, time_spent, object_type, progress=gr.Progress()):
    """페르소나 생성 함수 - 초기 생성만"""
    if image is None:
        return None, "이미지를 업로드해주세요.", "", {}, None, [], [], [], "", None, gr.update(visible=False)
    
    progress(0.1, desc="이미지 분석 중...")
    
    user_context = {
        "name": name,
        "location": location,
        "time_spent": time_spent,
        "object_type": object_type
    }
    
    try:
        # 이미지 유효성 검사 및 처리
        if isinstance(image, str):
            # 파일 경로인 경우
            try:
                image = Image.open(image)
            except Exception as img_error:
                return None, f"❌ 이미지 파일을 읽을 수 없습니다: {str(img_error)}", "", {}, None, [], [], [], "", None, gr.update(visible=False)
        elif not isinstance(image, Image.Image):
            return None, "❌ 올바른 이미지 형식이 아닙니다.", "", {}, None, [], [], [], "", None, gr.update(visible=False)
        
        # 이미지 형식 변환 (AVIF 등 특수 형식 처리)
        if image.format in ['AVIF', 'WEBP'] or image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        
        generator = PersonaGenerator()
        
        progress(0.3, desc="이미지 분석 중...")
        # 이미지 처리 방식 수정 - PIL Image 객체를 직접 전달
        image_analysis = generator.analyze_image(image)
        
        progress(0.5, desc="페르소나 생성 중...")
        # 프론트엔드 페르소나 생성
        frontend_persona = generator.create_frontend_persona(image_analysis, user_context)
        
        # 백엔드 페르소나 생성 (구조화된 프롬프트 포함)
        backend_persona = generator.create_backend_persona(frontend_persona, image_analysis)
        
        # 페르소나 정보 포맷팅
        persona_name = backend_persona["기본정보"]["이름"]
        persona_type = backend_persona["기본정보"]["유형"]
        
        # 성격 기반 한 문장 인사 생성
        personality_traits = backend_persona["성격특성"]
        warmth = personality_traits.get("온기", 50)
        humor = personality_traits.get("유머감각", 50)
        competence = personality_traits.get("능력", 50)
        
        # 성격에 따른 간단한 첫 인사
        if warmth >= 70 and humor >= 60:
            awakening_msg = f"🌟 **{persona_name}** - 안녕! 나는 {persona_name}이야~ 뭔가 재밌는 일 없을까? 😊"
        elif warmth >= 70:
            awakening_msg = f"🌟 **{persona_name}** - 안녕하세요! {persona_name}예요. 만나서 정말 기뻐요! 💫"
        elif humor >= 70:
            awakening_msg = f"🌟 **{persona_name}** - 어? 갑자기 의식이 생겼네! {persona_name}라고 해~ ㅋㅋ 😎"
        elif competence >= 70:
            awakening_msg = f"🌟 **{persona_name}** - 시스템 활성화 완료. {persona_name}입니다. 무엇을 도와드릴까요? 🤖"
        else:
            awakening_msg = f"🌟 **{persona_name}** - 음... 안녕? 나는 {persona_name}... 뭔가 어색하네. 😅"
        
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
            f"✅ {persona_name} 페르소나가 생성되었습니다!",  # status_output
            summary_display,  # persona_summary_display
            backend_persona["성격특성"],  # personality_traits_output (hidden)
            humor_chart,  # humor_chart_output
            flaws_df,  # attractive_flaws_output
            contradictions_df,  # contradictions_output
            variables_df,  # personality_variables_output
            awakening_msg,  # persona_awakening
            None,  # download_file (initially empty)
            gr.update(visible=True)  # adjustment_section (show)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ 페르소나 생성 중 오류 발생: {str(e)}", "", {}, None, [], [], [], "", None, gr.update(visible=False)

def adjust_persona_traits(persona, warmth, competence, humor, extraversion, humor_style):
    """페르소나 성격 특성 조정 - Gradio 5.x 호환"""
    if not persona or not isinstance(persona, dict):
        return None, "조정할 페르소나가 없습니다.", {}
    
    try:
        # 깊은 복사로 원본 보호
        import copy
        adjusted_persona = copy.deepcopy(persona)
        
        # 성격 특성 업데이트
        if "성격특성" not in adjusted_persona:
            adjusted_persona["성격특성"] = {}
            
        adjusted_persona["성격특성"]["온기"] = warmth
        adjusted_persona["성격특성"]["능력"] = competence  
        adjusted_persona["성격특성"]["유머감각"] = humor
        adjusted_persona["성격특성"]["외향성"] = extraversion
        adjusted_persona["유머스타일"] = humor_style
        
        # 조정된 정보 표시
        adjusted_info = {
            "이름": adjusted_persona.get("기본정보", {}).get("이름", "Unknown"),
            "온기": warmth,
            "능력": competence,
            "유머감각": humor, 
            "외향성": extraversion,
            "유머스타일": humor_style
        }
        
        persona_name = adjusted_persona.get("기본정보", {}).get("이름", "페르소나")
        adjustment_message = f"""
### 🎭 {persona_name}의 성격이 조정되었습니다!

💭 *"오, 뭔가 달라진 기분이야! 이런 내 모습도 괜찮네. 
이제 우리 진짜 친구가 될 수 있을 것 같아!"*

✨ **조정된 성격:**
• 온기: {warmth}/100 
• 능력: {competence}/100
• 유머감각: {humor}/100  
• 외향성: {extraversion}/100
• 유머스타일: {humor_style}
        """
        
        return adjusted_persona, adjustment_message, adjusted_info
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return persona, f"조정 중 오류 발생: {str(e)}", {}

def finalize_persona(persona):
    """페르소나 최종 확정"""
    if not persona:
        return None, "페르소나가 없습니다.", "", {}, None, [], [], [], "", None
    
    try:
        generator = PersonaGenerator()
        
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
            f"✅ {persona_name} 완성!",  # status_output
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
        return None, f"❌ 페르소나 확정 중 오류 발생: {str(e)}", "", {}, None, [], [], [], "", None

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
    """성격 차트 생성 - 영어 레이블 사용"""
    if not persona or "성격특성" not in persona:
        return None
    
    try:
        traits = persona["성격특성"]
        
        # 영어 라벨 매핑 (폰트 문제 완전 해결)
        trait_mapping = {
            "온기": "Warmth",
            "능력": "Competence", 
            "창의성": "Creativity",
            "외향성": "Extraversion",
            "유머감각": "Humor",
            "신뢰성": "Reliability",
            "공감능력": "Empathy"
        }
        
        categories = [trait_mapping.get(trait, trait) for trait in traits.keys()]
        values = list(traits.values())
        
        # 극좌표 차트 생성
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values_plot = values + [values[0]]  # Close the plot
        angles_plot = np.concatenate([angles, [angles[0]]])
        
        # 더 예쁜 색상과 스타일
        ax.plot(angles_plot, values_plot, 'o-', linewidth=3, color='#6366f1', markersize=8)
        ax.fill(angles_plot, values_plot, alpha=0.25, color='#6366f1')
        
        # 격자와 축 설정
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 각 점에 값 표시
        for angle, value in zip(angles, values):
            ax.text(angle, value + 5, f'{value}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='#2d3748')
        
        plt.title("Personality Traits Radar Chart", size=16, pad=20, fontweight='bold')
        
        return fig
    except Exception as e:
        print(f"성격 차트 생성 오류: {str(e)}")
        return None

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
    """페르소나를 JSON 파일로 내보내기"""
    if not persona:
        return None, "내보낼 페르소나가 없습니다."
    
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
        
        # JSON 파일 생성
        persona_name = persona_copy.get("기본정보", {}).get("이름", "persona")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{persona_name}_{timestamp}.json"
        
        # 임시 파일 생성
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(persona_copy, f, ensure_ascii=False, indent=2)
        
        return filepath, f"✅ JSON 파일이 생성되었습니다: {filename}"
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"JSON 내보내기 오류: {error_msg}")
        return None, f"❌ JSON 내보내기 중 오류 발생: {str(e)}"

# def get_saved_personas():
#     """저장된 페르소나 목록 가져오기 - 더 이상 사용하지 않음"""
#     return [], []

# def load_persona_from_selection(selected_row, personas_list):
#     """선택된 페르소나 로드 - 더 이상 사용하지 않음"""
#     return None, "이 기능은 더 이상 사용하지 않습니다. JSON 업로드를 사용하세요.", {}, {}, None, [], [], [], ""

def chat_with_loaded_persona(persona, user_message, chat_history=None):
    """현재 로드된 페르소나와 대화 - Gradio 5.31.0 호환"""
    if not persona:
        return chat_history or [], ""
    
    if not user_message.strip():
        return chat_history or [], ""
    
    try:
        generator = PersonaGenerator()
        
        # 대화 기록을 올바른 형태로 변환 (Gradio 5.x messages 형태)
        conversation_history = []
        if chat_history:
            for message in chat_history:
                if isinstance(message, dict) and "role" in message and "content" in message:
                    # 이미 올바른 messages 형태
                    conversation_history.append(message)
                elif isinstance(message, (list, tuple)) and len(message) >= 2:
                    # 이전 버전의 tuple 형태 처리
                    conversation_history.append({"role": "user", "content": message[0]})
                    conversation_history.append({"role": "assistant", "content": message[1]})
        
        # 페르소나와 대화
        response = generator.chat_with_persona(persona, user_message, conversation_history)
        
        # 새로운 대화를 messages 형태로 추가
        if chat_history is None:
            chat_history = []
        
        # Gradio 5.31.0 messages 형식: 각 메시지는 별도로 추가
        new_history = chat_history.copy()
        new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": response})
        
        return new_history, ""
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_response = f"죄송해요, 대화 중 오류가 발생했어요: {str(e)}"
        
        if chat_history is None:
            chat_history = []
        
        # 에러 메시지도 올바른 형식으로 추가
        new_history = chat_history.copy()
        new_history.append({"role": "user", "content": user_message})
        new_history.append({"role": "assistant", "content": error_response})
        
        return new_history, ""

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
        
        # 로드된 페르소나 인사말
        greeting = f"### 🤖 {persona_name}\n\n안녕! 나는 **{persona_name}**이야. JSON에서 다시 깨어났어! 대화해보자~ 😊"
        
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
    """성격 특성을 사용자 친화적인 형태로 포맷 (수치 없이 서술형만)"""
    if not persona or "성격특성" not in persona:
        return "페르소나가 생성되지 않았습니다."
    
    generator = PersonaGenerator()
    personality_traits = persona["성격특성"]
    descriptions = generator.get_personality_descriptions(personality_traits)
    
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
                            object_type_input = gr.Dropdown(
                                choices=["가전제품", "가구", "전자기기", "장식품", "도구", "개인용품", "기타"],
                                label="어떤 종류의 사물인가요?",
                                value="가구"
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
                            gr.Markdown("생성된 페르소나의 성격을 원하는 대로 조정해보세요!")
                            
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
                                    humor_slider = gr.Slider(
                                        minimum=0, maximum=100, value=50, step=1,
                                        label="유머감각",
                                        info="0: 진지함 ↔ 100: 유머러스"
                                    )
                                    extraversion_slider = gr.Slider(
                                        minimum=0, maximum=100, value=50, step=1,
                                        label="외향성",
                                        info="0: 내향적 ↔ 100: 외향적"
                                    )
                            
                            humor_style_radio = gr.Radio(
                                choices=["따뜻한 유머러스", "위트있는 재치꾼", "날카로운 관찰자", "자기 비하적"],
                                value="따뜻한 유머러스",
                                label="유머 스타일"
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
                                export_btn = gr.Button("📥 JSON 파일로 내보내기", variant="outline")
                            download_file = gr.File(label="다운로드", visible=False)
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
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 💬 대화")
                        # Gradio 4.44.1에서 권장하는 messages 형식 사용
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
        
        # 이벤트 핸들러
        create_btn.click(
            fn=create_persona_from_image,
            inputs=[image_input, name_input, location_input, time_spent_input, object_type_input],
            outputs=[
                current_persona, status_output, persona_summary_display, personality_traits_output,
                humor_chart_output, attractive_flaws_output, contradictions_output, 
                personality_variables_output, persona_awakening, download_file, adjustment_section
            ]
        ).then(
            # 슬라이더 값을 현재 페르소나 값으로 업데이트
            fn=lambda persona: (
                persona["성격특성"]["온기"] if persona else 50,
                persona["성격특성"]["능력"] if persona else 50,
                persona["성격특성"]["유머감각"] if persona else 50,
                persona["성격특성"]["외향성"] if persona else 50,
                persona["유머스타일"] if persona else "따뜻한 유머러스"
            ),
            inputs=[current_persona],
            outputs=[warmth_slider, competence_slider, humor_slider, extraversion_slider, humor_style_radio]
        )
        
        # 성격 조정 적용
        adjust_btn.click(
            fn=adjust_persona_traits,
            inputs=[current_persona, warmth_slider, competence_slider, humor_slider, extraversion_slider, humor_style_radio],
            outputs=[current_persona, adjustment_result, adjusted_info_output]
        )
        
        # 페르소나 최종 확정
        finalize_btn.click(
            fn=finalize_persona,
            inputs=[current_persona],
            outputs=[
                current_persona, status_output, persona_summary_display, personality_traits_output,
                humor_chart_output, attractive_flaws_output, contradictions_output, 
                personality_variables_output, persona_awakening, download_file
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
        
        export_btn.click(
            fn=export_persona_to_json,
            inputs=[current_persona],
            outputs=[download_file, export_status]
        ).then(
            fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
            inputs=[download_file],
            outputs=[download_file]
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
        
        # 예시 메시지 버튼들
        example_btn1.click(
            fn=lambda: "안녕!",
            outputs=[message_input]
        )
        
        example_btn2.click(
            fn=lambda: "너는 누구야?",
            outputs=[message_input]
        )
        
        example_btn3.click(
            fn=lambda: "뭘 좋아해?",
            outputs=[message_input]
        )
        
        # 앱 로드 시 페르소나 목록 로드 (백엔드에서 사용)
        app.load(
            fn=lambda: [],
            outputs=[personas_list]
        )
    
    return app

if __name__ == "__main__":
    app = create_main_interface()
    app.launch(server_name="0.0.0.0", server_port=7860) 