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

# Import modules
from modules.persona_generator import PersonaGenerator
from modules.data_manager import save_persona, load_persona, list_personas, toggle_frontend_backend_view

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
.persona-details {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    margin-top: 8px;
    background-color: #f8f9fa;
}
.collapsed-section {
    margin-top: 10px;
    margin-bottom: 10px;
}
.personality-chart {
    width: 100%;
    height: auto;
    margin-top: 15px;
}
.conversation-box {
    height: 400px;
    overflow-y: auto;
}
"""

def create_frontend_view_html(persona):
    """Create HTML representation of the frontend view of the persona"""
    if not persona:
        return "페르소나가 아직 생성되지 않았습니다."
    
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    object_type = persona.get("기본정보", {}).get("유형", "Unknown")
    description = persona.get("기본정보", {}).get("설명", "")
    
    # Personality traits
    traits_html = ""
    for trait, value in persona.get("성격특성", {}).items():
        traits_html += f"<div>{trait}: <div style='background: linear-gradient(to right, #6366f1 {value}%, #e0e0e0 {value}%); height: 12px; border-radius: 6px; margin-bottom: 8px;'></div></div>"
    
    # Flaws
    flaws = persona.get("매력적결함", [])
    flaws_html = ", ".join(flaws) if flaws else "없음"
    
    # Interests
    interests = persona.get("관심사", [])
    interests_html = ", ".join(interests) if interests else "없음"
    
    # Communication style
    communication_style = persona.get("소통방식", "")
    
    html = f"""
    <div class="persona-details">
        <h2>{name}</h2>
        <p><strong>유형:</strong> {object_type}</p>
        <p><strong>설명:</strong> {description}</p>
        
        <h3>성격 특성</h3>
        {traits_html}
        
        <h3>소통 방식</h3>
        <p>{communication_style}</p>
        
        <h3>매력적 결함</h3>
        <p>{flaws_html}</p>
        
        <h3>관심사</h3>
        <p>{interests_html}</p>
    </div>
    """
    
    return html

def create_backend_view_html(persona):
    """Create HTML representation of the backend view of the persona"""
    if not persona:
        return "페르소나가 아직 생성되지 않았습니다."
    
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    
    # Convert persona to formatted JSON
    try:
        persona_json = json.dumps(persona, ensure_ascii=False, indent=2)
        
        html = f"""
        <div class="persona-details">
            <h2>{name} - 백엔드 상세 정보</h2>
            <details>
                <summary>127개 성격 변수 및 상세 정보</summary>
                <pre style="background-color: #f5f5f5; padding: 12px; border-radius: 8px; overflow-x: auto;">{persona_json}</pre>
            </details>
        </div>
        """
        
        return html
    except Exception as e:
        return f"백엔드 정보 변환 오류: {str(e)}"

def generate_personality_chart(persona):
    """Generate a radar chart for personality traits"""
    if not persona or "성격특성" not in persona:
        # Return empty image
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, "데이터 없음", ha='center', va='center')
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    
    # Get traits
    traits = persona["성격특성"]
    
    # Create radar chart
    categories = list(traits.keys())
    values = list(traits.values())
    
    # Add the first value again to close the loop
    categories.append(categories[0])
    values.append(values[0])
    
    # Convert to radians
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw polygon
    ax.fill(angles, values, alpha=0.25, color='#6366f1')
    
    # Draw lines
    ax.plot(angles, values, 'o-', linewidth=2, color='#6366f1')
    
    # Fill labels
    ax.set_thetagrids(angles[:-1] * 180/np.pi, categories[:-1])
    ax.set_rlim(0, 100)
    
    # Add titles
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    plt.title(f"{name}의 성격 특성", size=15, color='#333333', y=1.1)
    
    # Styling
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def soul_awakening(image, object_name=None):
    """Analyze image and awaken the soul of the object"""
    if image is None:
        return (
            None, 
            gr.update(visible=True, value="이미지를 먼저 업로드해주세요."),
            None, None, None
        )
    
    # Step 1: Analyze image
    print(f"Analyzing image: {image}")
    analysis_result = persona_generator.analyze_image(image)
    
    # Create context
    user_context = {
        "name": object_name if object_name else analysis_result.get("object_type", "미확인 사물")
    }
    
    # Step 2: Create frontend persona
    frontend_persona = persona_generator.create_frontend_persona(analysis_result, user_context)
    
    # Step 3: Create backend persona
    backend_persona = persona_generator.create_backend_persona(frontend_persona, analysis_result)
    
    # Save both views to a single persona file
    filepath = save_persona(backend_persona)
    if filepath:
        backend_persona["filepath"] = filepath
    
    # Generate HTML views
    frontend_html = create_frontend_view_html(frontend_persona)
    backend_html = create_backend_view_html(backend_persona)
    
    # Generate personality chart
    chart_image = generate_personality_chart(frontend_persona)
    
    return (
        analysis_result,
        gr.update(visible=False, value=""),
        frontend_html,
        backend_html,
        chart_image
    )

def start_chat(current_persona):
    """Start a conversation with the current persona"""
    if not current_persona:
        return "페르소나를 먼저 생성하거나 불러와주세요.", [], []
    
    # Generate initial greeting
    name = current_persona.get("기본정보", {}).get("이름", "Unknown")
    try:
        initial_message = persona_generator.chat_with_persona(
            current_persona, 
            "안녕하세요!"
        )
        
        # Initialize conversation history
        conversation = [
            {"role": "assistant", "content": initial_message}
        ]
        
        # Format for chatbot
        chatbot_messages = [(None, initial_message)]
        
        return f"{name}과의 대화가 시작되었습니다.", chatbot_messages, conversation
    except Exception as e:
        return f"대화 시작 중 오류 발생: {str(e)}", [], []

def chat_with_persona(message, chatbot, conversation, current_persona):
    """Chat with the current persona"""
    if not message or not current_persona:
        return chatbot, conversation
    
    # Add user message to conversation
    conversation.append({"role": "user", "content": message})
    
    # Update chatbot display
    chatbot.append((message, None))
    
    # Get response from persona
    try:
        response = persona_generator.chat_with_persona(
            current_persona,
            message,
            conversation
        )
        
        # Add response to conversation
        conversation.append({"role": "assistant", "content": response})
        
        # Update chatbot display
        chatbot[-1] = (message, response)
        
        return chatbot, conversation
    except Exception as e:
        error_message = f"오류: {str(e)}"
        conversation.append({"role": "assistant", "content": error_message})
        chatbot[-1] = (message, error_message)
        return chatbot, conversation

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
        chart_image = generate_personality_chart(frontend_view)
        
        return persona, f"{persona['기본정보']['이름']}을(를) 로드했습니다.", frontend_html, backend_html, chart_image
    
    except Exception as e:
        return None, f"페르소나 로딩 중 오류 발생: {str(e)}", None, None, None

def save_current_persona(current_persona):
    """Save current persona to a JSON file"""
    if not current_persona:
        return "저장할 페르소나가 없습니다."
    
    try:
        filepath = save_persona(current_persona)
        if filepath:
            name = current_persona.get("기본정보", {}).get("이름", "Unknown")
            return f"{name} 페르소나가 저장되었습니다: {filepath}"
        else:
            return "페르소나 저장에 실패했습니다."
    except Exception as e:
        return f"저장 중 오류 발생: {str(e)}"

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

# Main Gradio app
with gr.Blocks(title="놈팽쓰 테스트 앱", theme=theme, css=css) as app:
    # Global state
    current_persona = gr.State(None)
    conversation_history = gr.State([])
    analysis_result_state = gr.State(None)
    personas_data = gr.State([])
    
    gr.Markdown(
    """
    # 🎭 놈팽쓰(MemoryTag) 테스트 앱
    
    사물에 영혼을 불어넣어 대화할 수 있는 페르소나 생성 테스트 앱입니다.
    
    ## 사용 방법
    1. **영혼 깨우기** 탭에서 이미지를 업로드하거나 이름을 입력하여 사물의 영혼을 깨웁니다.
    2. **대화하기** 탭에서 생성된 페르소나와 대화합니다.
    3. **페르소나 관리** 탭에서 저장된 페르소나를 관리합니다.
    """
    )
    
    with gr.Tabs() as tabs:
        # Tab 1: Soul Awakening
        with gr.Tab("영혼 깨우기"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 영혼 발견하기")
                    
                    object_image = gr.Image(
                        label="사물 이미지",
                        type="filepath"
                    )
                    
                    object_name = gr.Textbox(
                        label="사물 이름 (선택사항)",
                        placeholder="자동 감지하려면 비워두세요"
                    )
                    
                    awaken_btn = gr.Button(
                        "영혼 깨우기",
                        variant="primary"
                    )
                    
                    error_message = gr.Markdown(
                        "",
                        visible=False
                    )
                
                with gr.Column(scale=2):
                    with gr.Tabs() as result_tabs:
                        with gr.Tab("페르소나 프론트"):
                            frontend_view = gr.HTML(
                                label="프론트엔드 뷰",
                                value="사물의 영혼을 깨워주세요."
                            )
                            
                            personality_chart = gr.Image(
                                label="성격 차트",
                                show_label=False
                            )
                        
                        with gr.Tab("백엔드 상세 정보"):
                            backend_view = gr.HTML(
                                label="백엔드 뷰",
                                value="사물의 영혼을 깨워주세요."
                            )
            
            # Button row
            with gr.Row():
                save_btn = gr.Button("페르소나 저장")
                chat_btn = gr.Button("대화 시작하기")
                
                save_result = gr.Markdown("")
        
        # Tab 2: Chat
        with gr.Tab("대화하기"):
            with gr.Row():
                with gr.Column(scale=3):
                    chat_status = gr.Markdown("페르소나를 먼저 생성하거나 불러와주세요.")
                    
                    chatbot = gr.Chatbot(
                        label="대화",
                        height=400,
                        show_label=False,
                    )
                    
                    with gr.Row():
                        user_message = gr.Textbox(
                            label="메시지",
                            placeholder="메시지를 입력하세요...",
                            lines=2
                        )
                        
                        send_btn = gr.Button(
                            "전송",
                            variant="primary"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 현재 페르소나")
                    current_persona_html = gr.HTML("페르소나가 선택되지 않았습니다.")
                    
                    start_chat_btn = gr.Button("새 대화 시작")
                    clear_chat_btn = gr.Button("대화 초기화")
        
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
                        label="성격 차트"
                    )
            
            with gr.Accordion("백엔드 상세 정보", open=False):
                selected_persona_backend = gr.HTML("페르소나를 선택해주세요.")
    
    # Event handlers
    # Soul Awakening
    awaken_btn.click(
        fn=soul_awakening,
        inputs=[object_image, object_name],
        outputs=[analysis_result_state, error_message, frontend_view, backend_view, personality_chart]
    ).then(
        fn=lambda x: x,
        inputs=[frontend_view],
        outputs=[current_persona_html]
    )
    
    save_btn.click(
        fn=save_current_persona,
        inputs=[current_persona],
        outputs=[save_result]
    )
    
    chat_btn.click(
        fn=lambda: gr.Tabs.update(selected="대화하기"),
        outputs=[tabs]
    )
    
    # Chat
    start_chat_btn.click(
        fn=start_chat,
        inputs=[current_persona],
        outputs=[chat_status, chatbot, conversation_history]
    )
    
    send_btn.click(
        fn=chat_with_persona,
        inputs=[user_message, chatbot, conversation_history, current_persona],
        outputs=[chatbot, conversation_history]
    ).then(
        fn=lambda: "",
        outputs=[user_message]
    )
    
    user_message.submit(
        fn=chat_with_persona,
        inputs=[user_message, chatbot, conversation_history, current_persona],
        outputs=[chatbot, conversation_history]
    ).then(
        fn=lambda: "",
        outputs=[user_message]
    )
    
    clear_chat_btn.click(
        fn=lambda: ([], []),
        outputs=[chatbot, conversation_history]
    ).then(
        fn=lambda: "대화가 초기화되었습니다.",
        outputs=[chat_status]
    )
    
    # Persona Management
    refresh_btn.click(
        fn=get_personas_list,
        outputs=[personas_df, personas_data]
    )
    
    load_btn.click(
        fn=load_selected_persona,
        inputs=[personas_df, personas_data],
        outputs=[current_persona, load_result, selected_persona_frontend, selected_persona_backend, selected_persona_chart]
    ).then(
        fn=lambda x: x,
        inputs=[selected_persona_frontend],
        outputs=[current_persona_html]
    )
    
    # Initial load of personas list
    app.load(
        fn=get_personas_list,
        outputs=[personas_df, personas_data]
    )

if __name__ == "__main__":
    app.launch() 