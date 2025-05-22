import os
import json
import datetime
import uuid

# Define data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PERSONAS_DIR = os.path.join(DATA_DIR, "personas")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")

# Create directories if they don't exist
for directory in [DATA_DIR, PERSONAS_DIR, CONVERSATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

def save_persona(persona_data):
    """
    Save persona data to a JSON file
    
    Args:
        persona_data: Dictionary containing persona information
    
    Returns:
        File path where the persona was saved
    """
    # Generate filename
    name = persona_data.get("기본정보", {}).get("이름", "unnamed")
    sanitized_name = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{sanitized_name}_{timestamp}_{unique_id}.json"
    
    # Full file path
    filepath = os.path.join(PERSONAS_DIR, filename)
    
    # Save to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(persona_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    except Exception as e:
        print(f"Error saving persona: {str(e)}")
        return None

def load_persona(filepath):
    """
    Load persona data from a JSON file
    
    Args:
        filepath: Path to the persona JSON file
    
    Returns:
        Dictionary containing persona information
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            persona_data = json.load(f)
        
        return persona_data
    except Exception as e:
        print(f"Error loading persona: {str(e)}")
        return None

def list_personas():
    """
    List all available personas
    
    Returns:
        List of dictionaries with persona information
    """
    personas = []
    
    try:
        for filename in os.listdir(PERSONAS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(PERSONAS_DIR, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        persona_data = json.load(f)
                    
                    # Extract basic information
                    name = persona_data.get("기본정보", {}).get("이름", "Unknown")
                    object_type = persona_data.get("기본정보", {}).get("유형", "Unknown")
                    created_at = persona_data.get("기본정보", {}).get("생성일시", "Unknown")
                    
                    personas.append({
                        "name": name,
                        "type": object_type,
                        "created_at": created_at,
                        "filename": filename,
                        "filepath": filepath
                    })
                except Exception as e:
                    print(f"Error reading persona file {filename}: {str(e)}")
        
        # Sort by creation date (newest first)
        personas.sort(key=lambda x: x["created_at"] if x["created_at"] != "Unknown" else "", reverse=True)
        
        return personas
    except Exception as e:
        print(f"Error listing personas: {str(e)}")
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
    """
    Toggle between frontend and backend view of persona data
    
    Args:
        persona: Full persona data
    
    Returns:
        Tuple containing (frontend_view, backend_view)
    """
    # Create frontend view (simplified)
    frontend_view = {}
    
    # Basic information
    if "기본정보" in persona:
        frontend_view["기본정보"] = persona["기본정보"]
    
    # Personality traits
    if "성격특성" in persona:
        frontend_view["성격특성"] = persona["성격특성"]
    
    # Communication style
    if "소통방식" in persona:
        frontend_view["소통방식"] = persona["소통방식"]
    
    # Flaws
    if "매력적결함" in persona:
        frontend_view["매력적결함"] = persona["매력적결함"]
    
    # Interests
    if "관심사" in persona:
        frontend_view["관심사"] = persona["관심사"]
    
    # Experiences
    if "경험" in persona:
        frontend_view["경험"] = persona["경험"]
    
    # Backend view includes everything
    backend_view = persona
    
    return frontend_view, backend_view 