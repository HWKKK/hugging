#!/usr/bin/env python3
"""
API 연결 디버깅 스크립트
대화 기능이 작동하지 않는 원인을 진단합니다.
"""

import os
import sys
sys.path.append('modules')

from persona_generator import PersonaGenerator
import google.generativeai as genai

def test_api_connections():
    """API 연결 상태 테스트"""
    
    print("🔍 API 연결 진단을 시작합니다...\n")
    
    # 1. 환경변수 확인
    print("1️⃣ 환경변수 확인:")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"   GEMINI_API_KEY: {'✅ 설정됨' if gemini_key else '❌ 없음'}")
    print(f"   OPENAI_API_KEY: {'✅ 설정됨' if openai_key else '❌ 없음'}")
    
    if not gemini_key and not openai_key:
        print("\n❌ API 키가 설정되지 않았습니다!")
        print("\n해결 방법:")
        print("1. .env 파일 생성:")
        print("   echo 'GEMINI_API_KEY=your_gemini_key_here' > .env")
        print("   echo 'OPENAI_API_KEY=your_openai_key_here' >> .env")
        print("\n2. 또는 환경변수 직접 설정:")
        print("   export GEMINI_API_KEY=your_gemini_key_here")
        print("   export OPENAI_API_KEY=your_openai_key_here")
        return False
    
    # 2. PersonaGenerator 초기화 테스트
    print("\n2️⃣ PersonaGenerator 초기화 테스트:")
    
    # Gemini 테스트
    if gemini_key:
        try:
            generator_gemini = PersonaGenerator(api_provider="gemini")
            print(f"   Gemini 초기화: ✅ 성공")
            print(f"   API 키 길이: {len(gemini_key)} 문자")
        except Exception as e:
            print(f"   Gemini 초기화: ❌ 실패 - {str(e)}")
    
    # OpenAI 테스트
    if openai_key:
        try:
            generator_openai = PersonaGenerator(api_provider="openai")
            print(f"   OpenAI 초기화: ✅ 성공")
            print(f"   API 키 길이: {len(openai_key)} 문자")
        except Exception as e:
            print(f"   OpenAI 초기화: ❌ 실패 - {str(e)}")
    
    # 3. 실제 API 호출 테스트
    print("\n3️⃣ 실제 API 호출 테스트:")
    
    if gemini_key:
        try:
            generator = PersonaGenerator(api_provider="gemini")
            test_result = generator._generate_text_with_api("안녕하세요! 간단히 인사해주세요.")
            print(f"   Gemini API 호출: ✅ 성공")
            print(f"   응답 길이: {len(test_result)} 문자")
            print(f"   응답 미리보기: {test_result[:100]}...")
        except Exception as e:
            print(f"   Gemini API 호출: ❌ 실패 - {str(e)}")
    
    if openai_key:
        try:
            generator = PersonaGenerator(api_provider="openai")
            test_result = generator._generate_text_with_api("안녕하세요! 간단히 인사해주세요.")
            print(f"   OpenAI API 호출: ✅ 성공")
            print(f"   응답 길이: {len(test_result)} 문자")
            print(f"   응답 미리보기: {test_result[:100]}...")
        except Exception as e:
            print(f"   OpenAI API 호출: ❌ 실패 - {str(e)}")
    
    # 4. 페르소나 대화 테스트
    print("\n4️⃣ 페르소나 대화 테스트:")
    
    # 간단한 테스트 페르소나 생성
    test_persona = {
        "기본정보": {"이름": "테스트봇", "유형": "테스트"},
        "성격특성": {
            "온기": 70,
            "능력": 50,
            "외향성": 60,
            "친화성": 65,
            "성실성": 55,
            "신경증": 40,
            "개방성": 60,
            "창의성": 55,
            "유머감각": 65,
            "공감능력": 70
        }
    }
    
    available_apis = []
    if gemini_key:
        available_apis.append("gemini")
    if openai_key:
        available_apis.append("openai")
    
    for api in available_apis:
        try:
            generator = PersonaGenerator(api_provider=api)
            response = generator.chat_with_persona(test_persona, "안녕하세요!")
            
            if "API 연결이 설정되지 않아" in response or "뭔가 문제가 생긴 것 같아" in response:
                print(f"   {api.upper()} 페르소나 대화: ❌ 실패 - API 오류")
                print(f"   오류 응답: {response}")
            else:
                print(f"   {api.upper()} 페르소나 대화: ✅ 성공")
                print(f"   응답: {response[:100]}...")
        except Exception as e:
            print(f"   {api.upper()} 페르소나 대화: ❌ 실패 - {str(e)}")
    
    print("\n🔍 진단 완료!")
    return True

def quick_fix_suggestions():
    """빠른 수정 제안"""
    print("\n💡 빠른 해결 방법:")
    print("1. API 키 설정 (.env 파일 생성):")
    print("   GEMINI_API_KEY=your_actual_api_key_here")
    print("   OPENAI_API_KEY=your_actual_api_key_here")
    print()
    print("2. Gemini API 키 발급: https://makersuite.google.com/app/apikey")
    print("3. OpenAI API 키 발급: https://platform.openai.com/api-keys")
    print()
    print("4. API 키 형식 확인:")
    print("   - Gemini: AIza... (약 40자)")
    print("   - OpenAI: sk-... (약 60자)")

if __name__ == "__main__":
    success = test_api_connections()
    
    if not success:
        quick_fix_suggestions()
    
    print("\n✨ 문제가 해결되지 않으면 다음 명령어로 재테스트:")
    print("   python debug_api.py") 