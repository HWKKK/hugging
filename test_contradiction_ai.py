#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.persona_generator import PersonalityProfile

def test_ai_contradiction_generation():
    """AI 기반 모순적 특성 생성 테스트"""
    
    # 테스트 시나리오 1: 금속 전기포트 (온기 높음, 능력 낮음)
    print("🔥 테스트 1: 스테인리스 전기포트 (온기 90, 능력 20)")
    print("=" * 60)
    
    object_analysis_1 = {
        "object_type": "전기포트",
        "materials": ["스테인리스 스틸", "플라스틱"],
        "colors": ["은색", "검은색"],
        "size": "중간 크기",
        "condition": "양호함",
        "estimated_age": "1년 정도"
    }
    
    personality_traits_1 = {
        "온기": 90,
        "능력": 20,
        "외향성": 60,
        "유머감각": 80
    }
    
    profile1 = PersonalityProfile()
    contradictions1 = profile1.generate_contradictions(object_analysis_1, personality_traits_1)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions1, 1):
        print(f"   {i}. {contradiction}")
    print()
    
    # 테스트 시나리오 2: 플라스틱 인형 (외향성 낮음, 유머 높음)
    print("🧸 테스트 2: 플라스틱 인형 (외향성 10, 유머감각 95)")
    print("=" * 60)
    
    object_analysis_2 = {
        "object_type": "인형",
        "materials": ["플라스틱", "면직물"],
        "colors": ["분홍색", "흰색"],
        "size": "작은 크기",
        "condition": "낡음",
        "estimated_age": "오래됨"
    }
    
    personality_traits_2 = {
        "온기": 75,
        "능력": 40,
        "외향성": 10,
        "유머감각": 95
    }
    
    profile2 = PersonalityProfile()
    contradictions2 = profile2.generate_contradictions(object_analysis_2, personality_traits_2)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions2, 1):
        print(f"   {i}. {contradiction}")
    print()
    
    # 테스트 시나리오 3: 목재 연필 (능력 높음, 온기 낮음)
    print("✏️ 테스트 3: 목재 연필 (능력 95, 온기 15)")
    print("=" * 60)
    
    object_analysis_3 = {
        "object_type": "연필",
        "materials": ["목재", "흑연"],
        "colors": ["노란색", "은색"],
        "size": "작은 크기",
        "condition": "사용됨",
        "estimated_age": "몇 개월"
    }
    
    personality_traits_3 = {
        "온기": 15,
        "능력": 95,
        "외향성": 30,
        "유머감각": 40
    }
    
    profile3 = PersonalityProfile()
    contradictions3 = profile3.generate_contradictions(object_analysis_3, personality_traits_3)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions3, 1):
        print(f"   {i}. {contradiction}")
    print()
    
    # 테스트 시나리오 4: 천 소재 쿠션 (모든 값 극단)
    print("🛋️ 테스트 4: 천 소재 쿠션 (온기 100, 외향성 5)")
    print("=" * 60)
    
    object_analysis_4 = {
        "object_type": "쿠션",
        "materials": ["면직물", "솜"],
        "colors": ["베이지색"],
        "size": "중간 크기", 
        "condition": "매우 좋음",
        "estimated_age": "새것"
    }
    
    personality_traits_4 = {
        "온기": 100,
        "능력": 60,
        "외향성": 5,
        "유머감각": 85
    }
    
    profile4 = PersonalityProfile()
    contradictions4 = profile4.generate_contradictions(object_analysis_4, personality_traits_4)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions4, 1):
        print(f"   {i}. {contradiction}")
    print()
    
    print("🎯 결론:")
    print("각 테스트마다 사물의 특성과 성격 조정값이 반영된")
    print("창의적이고 말투가 드러나는 모순적 특성이 생성되었는지 확인해보세요!")

if __name__ == "__main__":
    test_ai_contradiction_generation() 