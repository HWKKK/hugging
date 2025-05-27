#!/usr/bin/env python3
"""
Gradio 4.x 채팅 형식 테스트 스크립트
"""

def test_chat_format():
    """채팅 데이터 형식 테스트"""
    
    # Gradio 4.x 형식: [[user, bot], [user, bot]]
    chat_history_4x = [
        ["안녕!", "안녕하세요! 반가워요!"],
        ["너는 누구야?", "저는 AI 페르소나입니다."]
    ]
    
    # Gradio 5.x 형식: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    chat_history_5x = [
        {"role": "user", "content": "안녕!"},
        {"role": "assistant", "content": "안녕하세요! 반가워요!"},
        {"role": "user", "content": "너는 누구야?"},
        {"role": "assistant", "content": "저는 AI 페르소나입니다."}
    ]
    
    print("=== Gradio 4.x 형식 테스트 ===")
    for i, chat_turn in enumerate(chat_history_4x):
        print(f"Turn {i}: {chat_turn}")
        if isinstance(chat_turn, (list, tuple)) and len(chat_turn) >= 2:
            user_msg, bot_msg = chat_turn[0], chat_turn[1]
            print(f"  User: {user_msg}")
            print(f"  Bot: {bot_msg}")
        else:
            print(f"  ERROR: Invalid format - {type(chat_turn)}")
    
    print("\n=== Gradio 5.x → 4.x 변환 테스트 ===")
    converted_4x = []
    for i in range(0, len(chat_history_5x), 2):
        if i + 1 < len(chat_history_5x):
            user_turn = chat_history_5x[i]
            bot_turn = chat_history_5x[i + 1]
            
            if (isinstance(user_turn, dict) and 'content' in user_turn and
                isinstance(bot_turn, dict) and 'content' in bot_turn):
                converted_4x.append([user_turn['content'], bot_turn['content']])
            else:
                print(f"ERROR: Invalid 5.x format at index {i}")
    
    print("Converted 4.x format:")
    for turn in converted_4x:
        print(f"  {turn}")
    
    print("\n=== 4.x → 5.x 변환 테스트 ===")
    converted_5x = []
    for chat_turn in chat_history_4x:
        if isinstance(chat_turn, (list, tuple)) and len(chat_turn) >= 2:
            user_msg, bot_msg = chat_turn[0], chat_turn[1]
            if user_msg:
                converted_5x.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                converted_5x.append({"role": "assistant", "content": str(bot_msg)})
    
    print("Converted 5.x format:")
    for turn in converted_5x:
        print(f"  {turn}")

def test_string_access():
    """문자열 인덱스 오류 테스트"""
    
    # 정상적인 딕셔너리 접근
    normal_dict = {"role": "user", "content": "hello"}
    print(f"Normal dict access: {normal_dict['role']}")
    
    # 문제가 되는 상황들
    test_cases = [
        "string_value",  # 문자열
        ["list", "value"],  # 리스트
        123,  # 숫자
        None,  # None
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i}: {type(test_case)} - {test_case}")
        try:
            # 이것이 오류를 일으킬 수 있는 부분
            if isinstance(test_case, dict) and 'role' in test_case:
                print(f"  Role: {test_case['role']}")
            else:
                print(f"  Not a valid dict with 'role' key")
                
            # 문자열에 정수가 아닌 키로 접근 시도 (오류 발생)
            if isinstance(test_case, str):
                print(f"  Attempting string access with 'role' key...")
                try:
                    result = test_case['role']  # 이것이 "string indices must be integers" 오류 발생
                except TypeError as e:
                    print(f"  ERROR: {e}")
                    
        except Exception as e:
            print(f"  Unexpected error: {e}")

if __name__ == "__main__":
    test_chat_format()
    print("\n" + "="*50 + "\n")
    test_string_access() 