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
    """페르소나 생성 함수"""
    if image is None:
        return None, "이미지를 업로드해주세요.", {}, {}, None, [], [], []

    progress(0.1, desc="이미지 분석 중...")
    
    user_context = {
        "name": name,
        "location": location,
        "time_spent": time_spent,
        "object_type": object_type
    }
    
    try:
        generator = PersonaGenerator()
        
        progress(0.3, desc="이미지 분석 중...")
        image_analysis = generator.analyze_image(image)
        
        if object_type:
            image_analysis["object_type"] = object_type
        
        progress(0.6, desc="페르소나 생성 중...")
        frontend_persona = generator.create_frontend_persona(image_analysis, user_context)
        
        progress(0.8, desc="상세 페르소나 생성 중...")
        backend_persona = generator.create_backend_persona(frontend_persona, image_analysis)
        
        progress(1.0, desc="완료!")
        
        # 기본 정보 추출
        basic_info = {
            "이름": backend_persona.get("기본정보", {}).get("이름", "Unknown"),
            "유형": backend_persona.get("기본정보", {}).get("유형", "Unknown"),
            "설명": backend_persona.get("기본정보", {}).get("설명", "")
        }
        
        personality_traits = backend_persona.get("성격특성", {})
        humor_chart = plot_humor_matrix(backend_persona.get("유머매트릭스", {}))
        
        attractive_flaws_df = []
        contradictions_df = []
        personality_variables_df = []
        
        if "매력적결함" in backend_persona:
            flaws = backend_persona["매력적결함"]
            attractive_flaws_df = [[flaw, "매력 증가"] for flaw in flaws]
            
        if "모순적특성" in backend_persona:
            contradictions = backend_persona["모순적특성"]
            contradictions_df = [[contradiction, "복잡성 증가"] for contradiction in contradictions]
            
        if "성격변수127" in backend_persona:
            variables = backend_persona["성격변수127"]
            if isinstance(variables, dict):
                personality_variables_df = [[var_name, score, VARIABLE_DESCRIPTIONS.get(var_name, "")] 
                                          for var_name, score in variables.items()]
        
        return backend_persona, "페르소나 생성 완료!", basic_info, personality_traits, humor_chart, attractive_flaws_df, contradictions_df, personality_variables_df
        
    except Exception as e:
        print(f"페르소나 생성 오류: {str(e)}")
        return None, f"오류 발생: {str(e)}", {}, {}, None, [], [], []

def generate_personality_chart(persona):
    """성격 차트 생성"""
    if not persona or "성격특성" not in persona:
        img = Image.new('RGB', (400, 400), color='white')
        draw = PIL.ImageDraw.Draw(img)
        draw.text((150, 180), "No data", fill='black')
        img_path = os.path.join("data", "temp_chart.png")
        os.makedirs("data", exist_ok=True)
        img.save(img_path)
        return img_path
    
    traits = persona["성격특성"]
    categories = list(traits.keys())
    values = list(traits.values())
    
    # 차트 생성
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    values_plot = values + [values[0]]  # Close the plot
    angles_plot = np.concatenate([angles, [angles[0]]])
    
    ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#6366f1')
    ax.fill(angles_plot, values_plot, alpha=0.25, color='#6366f1')
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    
    plt.title("성격 특성", size=16, pad=20)
    
    timestamp = int(time.time())
    img_path = os.path.join("data", f"chart_{timestamp}.png")
    plt.savefig(img_path, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return img_path

def save_persona_to_file(persona):
    """페르소나 저장"""
    if not persona:
        return "저장할 페르소나가 없습니다."
    
    try:
        persona_copy = copy.deepcopy(persona)
        
        # 저장 불가능한 객체 제거
        for key in list(persona_copy.keys()):
            if callable(persona_copy[key]):
                persona_copy.pop(key, None)
        
        filepath = save_persona(persona_copy)
        if filepath:
            name = persona.get("기본정보", {}).get("이름", "Unknown")
            return f"{name} 페르소나가 저장되었습니다."
        else:
            return "페르소나 저장에 실패했습니다."
    except Exception as e:
        return f"저장 중 오류 발생: {str(e)}"

def get_saved_personas():
    """저장된 페르소나 목록 가져오기"""
    try:
        personas = list_personas()
        df_data = []
        for i, persona in enumerate(personas):
            df_data.append([
                i,
                persona["name"],
                persona["type"],
                persona["created_at"]
            ])
        return df_data, personas
    except Exception:
        return [], []

def load_persona_from_selection(selected_row, personas_list):
    """선택된 페르소나 로드"""
    if selected_row is None or len(selected_row) == 0:
        return None, "선택된 페르소나가 없습니다.", {}, {}, None, [], [], []
    
    try:
        selected_index = selected_row.index[0] if hasattr(selected_row, 'index') else 0
        if selected_index >= len(personas_list):
            return None, "잘못된 선택입니다.", {}, {}, None, [], [], []
            
        filepath = personas_list[selected_index]["filepath"]
        persona = load_persona(filepath)
        
        if not persona:
            return None, "페르소나 로딩에 실패했습니다.", {}, {}, None, [], [], []
        
        basic_info = {
            "이름": persona.get("기본정보", {}).get("이름", "Unknown"),
            "유형": persona.get("기본정보", {}).get("유형", "Unknown"),
            "설명": persona.get("기본정보", {}).get("설명", "")
        }
        
        personality_traits = persona.get("성격특성", {})
        humor_chart = plot_humor_matrix(persona.get("유머매트릭스", {}))
        
        attractive_flaws_df = []
        contradictions_df = []
        personality_variables_df = []
        
        if "매력적결함" in persona:
            flaws = persona["매력적결함"]
            attractive_flaws_df = [[flaw, "매력 증가"] for flaw in flaws]
            
        if "모순적특성" in persona:
            contradictions = persona["모순적특성"]
            contradictions_df = [[contradiction, "복잡성 증가"] for contradiction in contradictions]
            
        if "성격변수127" in persona:
            variables = persona["성격변수127"]
            if isinstance(variables, dict):
                personality_variables_df = [[var_name, score, VARIABLE_DESCRIPTIONS.get(var_name, "")] 
                                          for var_name, score in variables.items()]
        
        return persona, f"{persona['기본정보']['이름']}을(를) 로드했습니다.", basic_info, personality_traits, humor_chart, attractive_flaws_df, contradictions_df, personality_variables_df
    
    except Exception as e:
        return None, f"로딩 중 오류 발생: {str(e)}", {}, {}, None, [], [], []

def chat_with_loaded_persona(persona, user_message, chat_history=None):
    """페르소나와 대화"""
    if chat_history is None:
        chat_history = []
        
    if not user_message.strip():
        return chat_history, ""
        
    if not persona:
        chat_history.append([user_message, "페르소나가 로드되지 않았습니다."])
        return chat_history, ""
    
    try:
        response = persona_generator.chat_with_persona(persona, user_message, chat_history)
        chat_history.append([user_message, response])
        return chat_history, ""
    except Exception as e:
        chat_history.append([user_message, f"대화 중 오류가 발생했습니다: {str(e)}"])
        return chat_history, ""

# 메인 인터페이스 생성
def create_main_interface():
    current_persona = gr.State(value=None)
    personas_list = gr.State(value=[])
    
    with gr.Blocks(theme=theme, css=css, title="놈팽쓰(MemoryTag)") as app:
        gr.Markdown("""
        # 놈팽쓰(MemoryTag): 당신 곁의 사물, 이제 친구가 되다
        일상 속 사물에 AI 페르소나를 부여하여 대화할 수 있게 해주는 서비스입니다.
        """)
        
        with gr.Tabs() as tabs:
            # 페르소나 생성 탭
            with gr.Tab("페르소나 생성", id="creation"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                        
                        create_btn = gr.Button("페르소나 생성", variant="primary", size="lg")
                        status_output = gr.Markdown("")
                    
                    with gr.Column(scale=1):
                        basic_info_output = gr.JSON(label="기본 정보")
                        personality_traits_output = gr.JSON(label="성격 특성")
                        
                        with gr.Row():
                            save_btn = gr.Button("페르소나 저장", variant="secondary")
                            chart_btn = gr.Button("성격 차트 생성", variant="secondary")
            
            # 상세 정보 탭
            with gr.Tab("상세 정보", id="details"):
                with gr.Row():
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
                    
                    with gr.Column():
                        personality_chart_output = gr.Image(label="성격 차트")
                        humor_chart_output = gr.Plot(label="유머 매트릭스")
                
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
                        gr.Markdown("### 페르소나 불러오기")
                        refresh_btn = gr.Button("목록 새로고침", variant="secondary")
                        persona_table = gr.Dataframe(
                            headers=["ID", "이름", "유형", "생성날짜"],
                            label="저장된 페르소나",
                            interactive=False
                        )
                        load_btn = gr.Button("선택한 페르소나 불러오기", variant="primary")
                        load_status = gr.Markdown("")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 대화")
                        chatbot = gr.Chatbot(height=400, label="대화")
                        with gr.Row():
                            message_input = gr.Textbox(
                                placeholder="메시지를 입력하세요...",
                                show_label=False,
                                lines=2
                            )
                            send_btn = gr.Button("전송", variant="primary")
        
        # 이벤트 핸들러
        create_btn.click(
            fn=create_persona_from_image,
            inputs=[image_input, name_input, location_input, time_spent_input, object_type_input],
            outputs=[
                current_persona, status_output, basic_info_output, personality_traits_output,
                humor_chart_output, attractive_flaws_output, contradictions_output, personality_variables_output
            ]
        )
        
        save_btn.click(
            fn=save_persona_to_file,
            inputs=[current_persona],
            outputs=[status_output]
        )
        
        chart_btn.click(
            fn=generate_personality_chart,
            inputs=[current_persona],
            outputs=[personality_chart_output]
        )
        
        refresh_btn.click(
            fn=get_saved_personas,
            outputs=[persona_table, personas_list]
        )
        
        load_btn.click(
            fn=load_persona_from_selection,
            inputs=[persona_table, personas_list],
            outputs=[
                current_persona, load_status, basic_info_output, personality_traits_output,
                humor_chart_output, attractive_flaws_output, contradictions_output, personality_variables_output
            ]
        ).then(
            fn=generate_personality_chart,
            inputs=[current_persona],
            outputs=[personality_chart_output]
        )
        
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
        
        # 앱 로드 시 페르소나 목록 로드
        app.load(
            fn=get_saved_personas,
            outputs=[persona_table, personas_list]
        )
    
    return app

if __name__ == "__main__":
    app = create_main_interface()
    app.launch(server_name="0.0.0.0", server_port=7860) 