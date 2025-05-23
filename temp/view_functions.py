import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL.ImageDraw
import os
import time
import random
import pandas as pd
import gradio as gr
import tempfile
import base64
from datetime import datetime

# 성격 데이터 시각화 함수
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

# 성격 차트 생성 함수
def generate_personality_chart(persona):
    """Generate a radar chart for personality traits"""
    if not persona or "성격특성" not in persona:
        # Return empty image with default PIL
        img = Image.new('RGB', (400, 400), color='white')
        draw = PIL.ImageDraw.Draw(img)
        draw.text((150, 180), "데이터 없음", fill='black')
        img_path = os.path.join("data", "temp_chart.png")
        img.save(img_path)
        return img_path
    
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
    
    # 한글 폰트 설정 - 기본적으로 사용 가능한 폰트를 먼저 시도
    # Matplotlib에서 지원하는 한글 폰트 목록
    korean_fonts = ['NanumGothic', 'NanumGothicCoding', 'NanumMyeongjo', 'Malgun Gothic', 'Gulim', 'Batang', 'Arial Unicode MS', 'DejaVu Sans']
    
    # 폰트 설정
    plt.rcParams['font.family'] = 'sans-serif'  # 기본 폰트 패밀리
    
    # 여러 폰트를 시도
    font_found = False
    for font in korean_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams.get('font.sans-serif', [])
            plt.text(0, 0, '테스트', fontfamily=font)
            font_found = True
            print(f"성공적으로 한글 폰트를 설정했습니다: {font}")
            break
        except:
            continue
    
    if not font_found:
        print("한글 지원 폰트를 찾을 수 없습니다. 영문으로 표시합니다.")
        # 영어 라벨 매핑
        english_labels = {
            "온기": "Warmth",
            "능력": "Ability",
            "신뢰성": "Trust",
            "친화성": "Friendly",
            "창의성": "Creative",
            "유머감각": "Humor",
            "외향성": "Extraversion"
        }
        categories = [english_labels.get(cat, cat) for cat in categories]
    
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
    
    # 4. 각 축 설정
    ax.set_thetagrids(angles[:-1] * 180/np.pi, categories[:-1], fontsize=12)
    
    # 제목 추가
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    plt.title(f"{name} 성격 특성", size=16, color='#374151', pad=20, fontweight='bold')
    
    # 저장
    timestamp = int(time.time())
    img_path = os.path.join("data", f"chart_{timestamp}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, format='png', bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    
    return img_path

# 페르소나 저장 함수
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
        
        # 외부 함수 호출이 필요한 부분
        from modules.data_manager import save_persona
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

# 성격 미세조정 함수
def refine_persona(persona, warmth, competence, creativity, extraversion, humor, trust, humor_style):
    """페르소나의 성격을 미세조정하는 함수"""
    if not persona:
        return persona, "페르소나가 없습니다."
    
    try:
        # 복사본 생성
        refined_persona = persona.copy()
        
        # 성격 특성 업데이트
        if "성격특성" in refined_persona:
            refined_persona["성격특성"]["온기"] = int(warmth)
            refined_persona["성격특성"]["능력"] = int(competence)
            refined_persona["성격특성"]["창의성"] = int(creativity)
            refined_persona["성격특성"]["외향성"] = int(extraversion)
            refined_persona["성격특성"]["유머감각"] = int(humor)
            refined_persona["성격특성"]["신뢰성"] = int(trust)
        
        # 유머 스타일 업데이트
        refined_persona["유머스타일"] = humor_style
        
        # 127개 성격 변수가 있으면 업데이트
        if "성격변수127" in refined_persona:
            # 온기 관련 변수 업데이트
            for var in ["W01_친절함", "W02_친근함", "W06_공감능력", "W07_포용력"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(warmth * 0.9 + random.randint(0, 20))
            
            # 능력 관련 변수 업데이트
            for var in ["C01_효율성", "C02_지능", "C05_정확성", "C09_실행력"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(competence * 0.9 + random.randint(0, 20))
            
            # 창의성 관련 변수 업데이트
            for var in ["C04_창의성", "C08_통찰력"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(creativity * 0.9 + random.randint(0, 20))
            
            # 외향성 관련 변수 업데이트
            for var in ["E01_사교성", "E02_활동성", "E03_자기주장", "E06_열정성"]:
                if var in refined_persona["성격변수127"]:
                    refined_persona["성격변수127"][var] = int(extraversion * 0.9 + random.randint(0, 20))
            
            # 유머 관련 변수 업데이트
            if "H01_유머감각" in refined_persona["성격변수127"]:
                refined_persona["성격변수127"]["H01_유머감각"] = int(humor * 0.9 + random.randint(0, 20))
            
            # 신뢰성 관련 변수 업데이트
            if "W04_신뢰성" in refined_persona["성격변수127"]:
                refined_persona["성격변수127"]["W04_신뢰성"] = int(trust * 0.9 + random.randint(0, 20))
        
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

# 페르소나 리스트 가져오기 함수
def get_personas_list():
    """Get list of personas for the dataframe"""
    from modules.data_manager import list_personas
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

# 선택한 페르소나 불러오기 함수
def load_selected_persona(selected_row, personas_list):
    """Load persona from the selected row in the dataframe"""
    if selected_row is None or len(selected_row) == 0:
        return None, "선택된 페르소나가 없습니다.", None, None, None
    
    try:
        # Get filepath from selected row
        selected_index = selected_row.index[0] if hasattr(selected_row, 'index') else 0
        filepath = personas_list[selected_index]["filepath"]
        
        # Load persona
        from modules.data_manager import load_persona, toggle_frontend_backend_view
        persona = load_persona(filepath)
        
        if not persona:
            return None, "페르소나 로딩에 실패했습니다.", None, None, None
        
        # Generate HTML views
        from temp.frontend_view import create_frontend_view_html
        from temp.backend_view import create_backend_view_html
        
        frontend_view, backend_view = toggle_frontend_backend_view(persona)
        frontend_html = create_frontend_view_html(frontend_view)
        backend_html = create_backend_view_html(backend_view)
        
        # Generate personality chart
        chart_image_path = generate_personality_chart(frontend_view)
        
        return persona, f"{persona['기본정보']['이름']}을(를) 로드했습니다.", frontend_html, backend_html, chart_image_path
    
    except Exception as e:
        return None, f"페르소나 로딩 중 오류 발생: {str(e)}", None, None, None

# 현재 페르소나 정보 표시 함수
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

# 성격 변수 데이터프레임 생성 함수
def get_personality_variables_df(persona):
    if not persona or "성격변수127" not in persona:
        return []
    
    variables = persona["성격변수127"]
    if isinstance(variables, dict):
        rows = []
        for var_name, score in variables.items():
            # 변수 설명은 앱의 메인 파일에서 정의되어 있을 것이므로 일단 빈 문자열로 처리
            description = ""
            rows.append([var_name, score, description])
        return rows
    return []

# 매력적 결함 데이터프레임 생성 함수
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

# 모순적 특성 데이터프레임 생성 함수
def get_contradictions_df(persona):
    if not persona or "모순적특성" not in persona:
        return []
    
    contradictions = persona["모순적특성"]
    effects = [
        "복잡성 +35%",
        "흥미도 +28%"
    ]
    
    return [[contradiction, effects[i] if i < len(effects) else "깊이감 증가"] for i, contradiction in enumerate(contradictions)]

def export_persona_json(persona):
    """
    페르소나를 JSON 파일로 내보내는 기능
    """
    if not persona:
        return None, "페르소나가 없습니다."
    
    try:
        # persona 객체를 JSON으로 직렬화
        persona_dict = persona.copy()
        
        # 복잡한 객체를 딕셔너리로 변환
        if "humor_matrix" in persona_dict and hasattr(persona_dict["humor_matrix"], "to_dict"):
            persona_dict["humor_matrix"] = persona_dict["humor_matrix"].to_dict()
            
        if "personality" in persona_dict and hasattr(persona_dict["personality"], "to_dict"):
            persona_dict["personality"] = persona_dict["personality"].to_dict()
        
        # 현재 시간을 파일명에 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        object_name = persona.get("name", "unknown_persona").replace(" ", "_").lower()
        filename = f"{object_name}_{timestamp}.json"
        
        # JSON 문자열로 변환
        json_str = json.dumps(persona_dict, ensure_ascii=False, indent=2)
        
        return filename, json_str
    except Exception as e:
        print(f"페르소나 내보내기 실패: {e}")
        return None, f"페르소나 내보내기 실패: {e}"

def import_persona_json(file_obj):
    """
    JSON 파일로부터 페르소나를 불러오는 기능
    """
    if not file_obj:
        return None, "업로드된 파일이 없습니다."
    
    try:
        # JSON 파일을 로드하여 딕셔너리로 변환
        content = file_obj.read().decode('utf-8')
        persona_dict = json.loads(content)
        
        # 필수 필드 확인
        required_fields = ["name", "object_type"]
        for field in required_fields:
            if field not in persona_dict:
                return None, f"유효하지 않은 페르소나 파일: {field} 필드가 없습니다."
        
        # 복잡한 객체 재구성
        if "humor_matrix" in persona_dict and isinstance(persona_dict["humor_matrix"], dict):
            persona_dict["humor_matrix"] = HumorMatrix.from_dict(persona_dict["humor_matrix"])
            
        if "personality" in persona_dict and isinstance(persona_dict["personality"], dict):
            persona_dict["personality"] = PersonalityProfile.from_dict(persona_dict["personality"])
        
        return persona_dict, f"{persona_dict['name']} 페르소나를 로드했습니다."
    except Exception as e:
        print(f"페르소나 불러오기 실패: {e}")
        return None, f"페르소나 불러오기 실패: {e}" 