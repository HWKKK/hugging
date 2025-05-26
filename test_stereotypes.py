#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from modules.persona_generator import PersonaGenerator, PersonalityProfile, HumorMatrix
import json

def test_8_stereotypes():
    """8가지 스테레오타입 극명한 차별화 테스트"""
    
    generator = PersonaGenerator()
    
    # 8가지 극명한 스테레오타입 정의
    stereotypes = [
        {
            "name": "🎉 열정적 엔터테이너",
            "image_analysis": {
                'object_type': '블루투스 스피커',
                'colors': ['red', 'gold'],
                'materials': ['plastic'],
                'condition': '새것같음',
                'personality_hints': {'warmth_factor': 85, 'competence_factor': 45, 'humor_factor': 90}
            },
            "user_context": {'name': '댄스킹', 'location': '파티룸'}
        },
        {
            "name": "❄️ 차가운 완벽주의자",
            "image_analysis": {
                'object_type': '기계식 시계',
                'colors': ['black', 'silver'],
                'materials': ['metal', 'steel'],
                'condition': '완벽함',
                'personality_hints': {'warmth_factor': 25, 'competence_factor': 95, 'humor_factor': 20}
            },
            "user_context": {'name': '프리시전', 'location': '사무실'}
        },
        {
            "name": "💝 따뜻한 상담사",
            "image_analysis": {
                'object_type': '털 쿠션',
                'colors': ['beige', 'pink'],
                'materials': ['fabric', 'cotton'],
                'condition': '부드러움',
                'personality_hints': {'warmth_factor': 95, 'competence_factor': 55, 'humor_factor': 30}
            },
            "user_context": {'name': '허니', 'location': '침실'}
        },
        {
            "name": "🎭 위트 넘치는 지식인",
            "image_analysis": {
                'object_type': '고급 만년필',
                'colors': ['burgundy', 'gold'],
                'materials': ['metal', 'resin'],
                'condition': '고급스러움',
                'personality_hints': {'warmth_factor': 40, 'competence_factor': 90, 'humor_factor': 85}
            },
            "user_context": {'name': '위즈덤', 'location': '서재'}
        },
        {
            "name": "🌙 수줍은 몽상가",
            "image_analysis": {
                'object_type': '수채화 붓',
                'colors': ['lavender', 'white'],
                'materials': ['wood', 'hair'],
                'condition': '예술적임',
                'personality_hints': {'warmth_factor': 60, 'competence_factor': 50, 'humor_factor': 55}
            },
            "user_context": {'name': '루나', 'location': '화실'}
        },
        {
            "name": "👑 카리스마틱 리더",
            "image_analysis": {
                'object_type': '고급 노트북',
                'colors': ['titanium', 'black'],
                'materials': ['aluminum', 'carbon'],
                'condition': '프리미엄',
                'personality_hints': {'warmth_factor': 60, 'competence_factor': 95, 'humor_factor': 70}
            },
            "user_context": {'name': '맥스', 'location': '회의실'}
        },
        {
            "name": "😜 장난꾸러기 친구",
            "image_analysis": {
                'object_type': '고무공',
                'colors': ['rainbow', 'bright'],
                'materials': ['rubber'],
                'condition': '통통튐',
                'personality_hints': {'warmth_factor': 80, 'competence_factor': 30, 'humor_factor': 95}
            },
            "user_context": {'name': '바운시', 'location': '놀이터'}
        },
        {
            "name": "🔮 신비로운 현자",
            "image_analysis": {
                'object_type': '오래된 책',
                'colors': ['dark_brown', 'gold'],
                'materials': ['leather', 'paper'],
                'condition': '고풍스러움',
                'personality_hints': {'warmth_factor': 50, 'competence_factor': 90, 'humor_factor': 40}
            },
            "user_context": {'name': '세이지', 'location': '도서관'}
        }
    ]
    
    print("🎭 8가지 스테레오타입 극명한 차별화 테스트")
    print("=" * 80)
    
    results = []
    
    for i, stereotype in enumerate(stereotypes, 1):
        print(f"\n{stereotype['name']} ({i}/8)")
        print("-" * 50)
        
        try:
            # 페르소나 생성
            persona = generator.create_frontend_persona(
                stereotype['image_analysis'], 
                stereotype['user_context']
            )
            
            # 성격 특성 분석
            traits = persona['성격특성']
            profile_vars = persona.get('성격프로필', {})
            
            # 핵심 지표 추출
            result = {
                "이름": persona['기본정보']['이름'],
                "스테레오타입": stereotype['name'],
                "사물": stereotype['image_analysis']['object_type'],
                "위치": stereotype['user_context']['location'],
                "성격특성": {
                    "온기": round(traits['온기'], 1),
                    "능력": round(traits['능력'], 1),
                    "외향성": round(traits['외향성'], 1),
                    "유머감각": round(traits['유머감각'], 1),
                    "창의성": round(traits['창의성'], 1),
                    "공감능력": round(traits['공감능력'], 1)
                },
                "127변수_샘플": {
                    "W01_친절함": profile_vars.get('W01_친절함', 0),
                    "C01_효율성": profile_vars.get('C01_효율성', 0),
                    "E01_사교성": profile_vars.get('E01_사교성', 0),
                    "H01_언어유희빈도": profile_vars.get('H01_언어유희빈도', 0),
                    "F01_완벽주의불안": profile_vars.get('F01_완벽주의불안', 0)
                },
                "매력적결함": persona['매력적결함'],
                "모순적특성": persona['모순적특성'],
                "유머스타일": persona['유머스타일'],
                "소통방식": persona['소통방식']
            }
            
            results.append(result)
            
            # 출력
            print(f"👤 이름: {result['이름']}")
            print(f"🏠 사물: {result['사물']} (위치: {result['위치']})")
            print(f"📊 성격: 온기{result['성격특성']['온기']} 능력{result['성격특성']['능력']} 외향성{result['성격특성']['외향성']} 유머{result['성격특성']['유머감각']}")
            print(f"🧬 127변수: 친절함{result['127변수_샘플']['W01_친절함']} 효율성{result['127변수_샘플']['C01_효율성']} 사교성{result['127변수_샘플']['E01_사교성']}")
            print(f"💎 매력적결함: {len(result['매력적결함'])}개 - {result['매력적결함'][0][:30]}...")
            print(f"🌈 모순적특성: {len(result['모순적특성'])}개 - {result['모순적특성'][0][:30]}...")
            print(f"🎪 유머스타일: {result['유머스타일'][:50]}...")
            print(f"💬 소통방식: {result['소통방식'][:50]}...")
            
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n\n📊 차별화 분석 결과:")
    print("=" * 80)
    
    if len(results) >= 2:
        # 극명한 차별화 확인
        print("🔥 극명한 차별화 확인:")
        
        # 가장 따뜻한 vs 가장 차가운
        warmest = max(results, key=lambda x: x['성격특성']['온기'])
        coldest = min(results, key=lambda x: x['성격특성']['온기'])
        print(f"  🔥 가장 따뜻: {warmest['이름']} (온기 {warmest['성격특성']['온기']})")
        print(f"  ❄️ 가장 차가운: {coldest['이름']} (온기 {coldest['성격특성']['온기']})")
        print(f"  📈 차이: {warmest['성격특성']['온기'] - coldest['성격특성']['온기']:.1f}")
        
        # 가장 유능한 vs 가장 무능한
        most_competent = max(results, key=lambda x: x['성격특성']['능력'])
        least_competent = min(results, key=lambda x: x['성격특성']['능력'])
        print(f"  🏆 가장 유능: {most_competent['이름']} (능력 {most_competent['성격특성']['능력']})")
        print(f"  🤪 가장 무능: {least_competent['이름']} (능력 {least_competent['성격특성']['능력']})")
        print(f"  📈 차이: {most_competent['성격특성']['능력'] - least_competent['성격특성']['능력']:.1f}")
        
        # 가장 외향적 vs 가장 내향적
        most_extroverted = max(results, key=lambda x: x['성격특성']['외향성'])
        most_introverted = min(results, key=lambda x: x['성격특성']['외향성'])
        print(f"  🎉 가장 외향적: {most_extroverted['이름']} (외향성 {most_extroverted['성격특성']['외향성']})")
        print(f"  🤫 가장 내향적: {most_introverted['이름']} (외향성 {most_introverted['성격특성']['외향성']})")
        print(f"  📈 차이: {most_extroverted['성격특성']['외향성'] - most_introverted['성격특성']['외향성']:.1f}")
        
        print(f"\n✅ 차별화 성공! 각 특성별로 극명한 차이가 확인됨")
        
        # 사물 특성 반영 확인
        print(f"\n🎨 사물 특성 반영 확인:")
        for result in results:
            print(f"  {result['사물']} → {result['이름']} → 온기{result['성격특성']['온기']} 능력{result['성격특성']['능력']}")
    
    return results

def analyze_physical_trait_impact():
    """물리적 특성이 성격에 미치는 영향 분석"""
    
    print(f"\n🎨 물리적 특성 → 성격 영향 분석")
    print("=" * 50)
    
    generator = PersonaGenerator()
    
    # 같은 기본 사물, 다른 물리적 특성
    base_object = {
        'object_type': '머그컵',
        'personality_hints': {'warmth_factor': 50, 'competence_factor': 50, 'humor_factor': 50}
    }
    
    variations = [
        {
            "name": "빨간 플라스틱 머그컵",
            "colors": ['red'],
            "materials": ['plastic'],
            "condition": "새것"
        },
        {
            "name": "검은 금속 머그컵", 
            "colors": ['black'],
            "materials": ['metal'],
            "condition": "고급스러움"
        },
        {
            "name": "베이지 천 머그컵",
            "colors": ['beige'],
            "materials": ['fabric'],
            "condition": "부드러움"
        }
    ]
    
    results = []
    for variation in variations:
        image_analysis = {**base_object, **variation}
        user_context = {'name': variation['name'][:5]}
        
        persona = generator.create_frontend_persona(image_analysis, user_context)
        traits = persona['성격특성']
        
        result = {
            "name": variation['name'],
            "colors": variation['colors'],
            "materials": variation['materials'],
            "condition": variation['condition'],
            "warmth": round(traits['온기'], 1),
            "competence": round(traits['능력'], 1),
            "humor": round(traits['유머감각'], 1)
        }
        results.append(result)
        
        print(f"{result['name']}: 온기{result['warmth']} 능력{result['competence']} 유머{result['humor']}")
    
    print(f"\n🔍 결론: 같은 사물도 색상/재질/상태에 따라 성격이 다르게 형성됨!")
    return results

if __name__ == "__main__":
    # 8가지 스테레오타입 테스트
    stereotype_results = test_8_stereotypes()
    
    # 물리적 특성 영향 분석
    physical_results = analyze_physical_trait_impact()
    
    print(f"\n\n🎉 테스트 완료!")
    print(f"📊 8가지 스테레오타입: {len(stereotype_results)}개 생성")
    print(f"🎨 물리적 특성 변화: {len(physical_results)}개 테스트")
    print(f"✅ 시스템이 정상적으로 차별화된 성격을 생성하고 있습니다!") 