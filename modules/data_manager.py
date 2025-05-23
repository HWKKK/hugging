import os
import json
import time
from datetime import datetime
import uuid

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PERSONAS_DIR = os.path.join(DATA_DIR, "personas")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")

# Create directories if they don't exist
for directory in [DATA_DIR, PERSONAS_DIR, CONVERSATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

def save_persona(persona):
    """페르소나 객체를 JSON 파일로 저장"""
    if not persona or "기본정보" not in persona:
        return None
    
    # 저장 디렉토리 확인
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    
    # 파일명 생성 (이름_타입_타임스탬프.json)
    name = persona.get("기본정보", {}).get("이름", "unknown")
    object_type = persona.get("기본정보", {}).get("유형", "unknown")
    timestamp = int(time.time())
    
    # 공백이나 특수문자 처리
    name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    object_type = object_type.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    filename = f"{name}_{object_type}_{timestamp}.json"
    filepath = os.path.join(PERSONAS_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(persona, f, ensure_ascii=False, indent=2)
        return filepath
    except Exception as e:
        print(f"페르소나 저장 오류: {str(e)}")
        return None

def load_persona(filepath):
    """JSON 파일에서 페르소나 객체 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            persona = json.load(f)
        return persona
    except Exception as e:
        print(f"페르소나 로드 오류: {str(e)}")
        return None

def list_personas():
    """저장된 모든 페르소나 목록 반환"""
    try:
        personas = []
        personas_dir = PERSONAS_DIR
        
        if not os.path.exists(personas_dir):
            os.makedirs(personas_dir, exist_ok=True)
            return personas
        
        for filename in os.listdir(personas_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(personas_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        persona = json.load(f)
                    
                    # 기본 정보 추출
                    name = persona.get("기본정보", {}).get("이름", "Unknown")
                    persona_type = persona.get("기본정보", {}).get("유형", "Unknown")
                    
                    # 생성 시간 (파일명에서 추출 또는 메타데이터에서)
                    created_at = persona.get("기본정보", {}).get("생성일시", "")
                    if not created_at:
                        # 파일명에서 타임스탬프 추출 시도
                        try:
                            timestamp = int(filename.split("_")[-1].split(".")[0])
                            created_at = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                        except:
                            created_at = "알 수 없음"
                    
                    personas.append({
                        "name": name,
                        "type": persona_type,
                        "created_at": created_at,
                        "filename": filename,
                        "filepath": filepath
                    })
                except Exception as e:
                    print(f"파일 {filename} 로드 오류: {str(e)}")
                    continue
        
        # 최신순 정렬
        personas.sort(key=lambda p: p["filepath"], reverse=True)
        return personas
    
    except Exception as e:
        print(f"페르소나 목록 조회 오류: {str(e)}")
        return []

def save_conversation(conversation_data):
    """
    Save conversation data to a JSON file
    
    Args:
        conversation_data: Dictionary containing conversation information
    
    Returns:
        File path where the conversation was saved
    """
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_name = conversation_data.get("persona", {}).get("기본정보", {}).get("이름", "unnamed")
    sanitized_name = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in persona_name)
    unique_id = str(uuid.uuid4())[:8]
    filename = f"conversation_{sanitized_name}_{timestamp}_{unique_id}.json"
    
    # Full file path
    filepath = os.path.join(CONVERSATIONS_DIR, filename)
    
    # Save to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    except Exception as e:
        print(f"Error saving conversation: {str(e)}")
        return None

def toggle_frontend_backend_view(persona):
    """페르소나 객체의 프론트엔드/백엔드 뷰 전환"""
    if not persona:
        return None, None
    
    # 원본 데이터 복사
    import copy
    frontend_persona = copy.deepcopy(persona)
    backend_persona = copy.deepcopy(persona)
    
    return frontend_persona, backend_persona 