#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from modules.persona_generator import PersonaGenerator
import traceback
import json

def test_persona_creation():
    """페르소나 생성 테스트"""
    try:
        print("🔧 PersonaGenerator 초기화...")
        generator = PersonaGenerator()
        print("✅ 초기화 성공")
        
        # 8가지 대표 스테레오타입 테스트
        test_cases = [
            {
                "name": "열정적_엔터테이너",
                "image_analysis": {
                    'object_type': '스피커',
                    'colors': ['red', 'gold'],
                    'materials': ['plastic'],
                    'condition': '새것',
                    'personality_hints': {
                        'warmth_factor': 85,
                        'competence_factor': 45,
                        'humor_factor': 90
                    }
                },
                "user_context": {'name': '댄스킹'}
            },
            {
                "name": "차가운_완벽주의자", 
                "image_analysis": {
                    'object_type': '시계',
                    'colors': ['black', 'silver'],
                    'materials': ['metal'],
                    'condition': '완벽함',
                    'personality_hints': {
                        'warmth_factor': 25,
                        'competence_factor': 95,
                        'humor_factor': 20
                    }
                },
                "user_context": {'name': '프리시전'}
            },
            {
                "name": "따뜻한_상담사",
                "image_analysis": {
                    'object_type': '쿠션',
                    'colors': ['beige', 'pink'],
                    'materials': ['fabric'],
                    'condition': '부드러움',
                    'personality_hints': {
                        'warmth_factor': 95,
                        'competence_factor': 55,
                        'humor_factor': 30
                    }
                },
                "user_context": {'name': '허니'}
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎭 테스트 {i}: {test_case['name']}")
            try:
                persona = generator.create_frontend_persona(
                    test_case['image_analysis'], 
                    test_case['user_context']
                )
                
                print(f"  ✅ 생성 성공: {persona['기본정보']['이름']}")
                print(f"  📊 온기: {persona['성격특성']['온기']:.1f}")
                print(f"  📊 능력: {persona['성격특성']['능력']:.1f}")
                print(f"  📊 유머: {persona['성격특성']['유머감각']:.1f}")
                print(f"  💎 매력적결함: {len(persona['매력적결함'])}개")
                print(f"  🌈 모순특성: {len(persona['모순적특성'])}개")
                print(f"  📝 127변수: {len(persona.get('성격프로필', {}))}개")
                
                # 간단한 대화 테스트
                try:
                    response = generator.chat_with_persona(persona, "안녕!", [])
                    print(f"  💬 대화 성공: {len(response)}자")
                    print(f"  💬 응답 미리보기: {response[:50]}...")
                except Exception as e:
                    print(f"  ❌ 대화 오류: {str(e)}")
                    
            except Exception as e:
                print(f"  ❌ 생성 오류: {str(e)}")
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 전체 오류: {str(e)}")
        traceback.print_exc()

def test_personality_profile():
    """PersonalityProfile 클래스 테스트"""
    try:
        print("\n🧬 PersonalityProfile 테스트...")
        from modules.persona_generator import PersonalityProfile
        
        profile = PersonalityProfile()
        print(f"✅ 기본 생성: {len(profile.variables)}개 변수")
        
        # 카테고리 요약 테스트
        warmth_avg = profile.get_category_summary("W")
        print(f"✅ 온기 평균: {warmth_avg}")
        
        # 매력적 결함 생성 테스트
        flaws = profile.generate_attractive_flaws()
        print(f"✅ 매력적 결함: {len(flaws)}개")
        
        # 모순적 특성 생성 테스트
        contradictions = profile.generate_contradictions()
        print(f"✅ 모순적 특성: {len(contradictions)}개")
        
    except Exception as e:
        print(f"❌ PersonalityProfile 오류: {str(e)}")
        traceback.print_exc()

def test_humor_matrix():
    """HumorMatrix 클래스 테스트"""
    try:
        print("\n🎪 HumorMatrix 테스트...")
        from modules.persona_generator import HumorMatrix
        
        matrix = HumorMatrix()
        print(f"✅ 기본 생성: {matrix.dimensions}")
        
        # 템플릿 테스트
        matrix2 = HumorMatrix.from_template("witty_wordsmith")
        print(f"✅ 템플릿 생성: {matrix2.dimensions}")
        
        # 프롬프트 생성 테스트
        prompt = matrix.generate_humor_prompt()
        print(f"✅ 프롬프트 생성: {len(prompt)}자")
        
    except Exception as e:
        print(f"❌ HumorMatrix 오류: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 놈팽쓰 페르소나 시스템 디버깅 테스트")
    print("=" * 50)
    
    test_personality_profile()
    test_humor_matrix()
    test_persona_creation()
    
    print("\n✅ 디버깅 테스트 완료") 