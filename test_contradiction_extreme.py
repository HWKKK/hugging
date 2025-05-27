#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.persona_generator import PersonalityProfile

def test_extreme_personality_contradictions():
    """극단적인 성격 조합 테스트 - 사용자 조정값이 확실히 드러나는지 확인"""
    
    # 테스트 1: 스테인리스 시계 (온기 100, 능력 5) - 극단 대비
    print("⏰ 테스트 1: 스테인리스 시계 (온기 100, 능력 5)")
    print("=" * 60)
    
    object_analysis_1 = {
        "object_type": "시계",
        "materials": ["스테인리스 스틸"],
        "colors": ["은색"],
        "size": "중간 크기",
        "condition": "좋음",
        "estimated_age": "새것"
    }
    
    personality_traits_1 = {
        "온기": 100,
        "능력": 5,
        "외향성": 50,
        "유머감각": 50
    }
    
    profile1 = PersonalityProfile()
    contradictions1 = profile1.generate_contradictions(object_analysis_1, personality_traits_1)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions1, 1):
        print(f"   {i}. {contradiction}")
    print("✅ 예상: 따뜻함 + 서툼이 드러나야 함")
    print()
    
    # 테스트 2: 플라스틱 로봇 (온기 5, 능력 100) - 반대 극단
    print("🤖 테스트 2: 플라스틱 로봇 (온기 5, 능력 100)")
    print("=" * 60)
    
    object_analysis_2 = {
        "object_type": "로봇",
        "materials": ["플라스틱", "금속"],
        "colors": ["흰색", "파란색"],
        "size": "중간 크기",
        "condition": "새것",
        "estimated_age": "새것"
    }
    
    personality_traits_2 = {
        "온기": 5,
        "능력": 100,
        "외향성": 50,
        "유머감각": 50
    }
    
    profile2 = PersonalityProfile()
    contradictions2 = profile2.generate_contradictions(object_analysis_2, personality_traits_2)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions2, 1):
        print(f"   {i}. {contradiction}")
    print("✅ 예상: 차가움 + 뛰어난 능력이 드러나야 함")
    print()
    
    # 테스트 3: 면 인형 (외향성 5, 유머 100) - 내향적 유머러스
    print("🧸 테스트 3: 면 인형 (외향성 5, 유머 100)")
    print("=" * 60)
    
    object_analysis_3 = {
        "object_type": "인형",
        "materials": ["면직물", "솜"],
        "colors": ["분홍색"],
        "size": "작은 크기",
        "condition": "오래됨",
        "estimated_age": "오래됨"
    }
    
    personality_traits_3 = {
        "온기": 70,
        "능력": 50,
        "외향성": 5,
        "유머감각": 100
    }
    
    profile3 = PersonalityProfile()
    contradictions3 = profile3.generate_contradictions(object_analysis_3, personality_traits_3)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions3, 1):
        print(f"   {i}. {contradiction}")
    print("✅ 예상: 조용함 + 뛰어난 유머 감각이 드러나야 함")
    print()
    
    # 테스트 4: 목재 트로피 (외향성 100, 유머 5) - 외향적이지만 유머없음
    print("🏆 테스트 4: 목재 트로피 (외향성 100, 유머 5)")
    print("=" * 60)
    
    object_analysis_4 = {
        "object_type": "트로피",
        "materials": ["목재", "금속"],
        "colors": ["금색", "갈색"],
        "size": "큰 크기",
        "condition": "좋음",
        "estimated_age": "몇 년"
    }
    
    personality_traits_4 = {
        "온기": 60,
        "능력": 80,
        "외향성": 100,
        "유머감각": 5
    }
    
    profile4 = PersonalityProfile()
    contradictions4 = profile4.generate_contradictions(object_analysis_4, personality_traits_4)
    
    print("📝 생성된 모순적 특성:")
    for i, contradiction in enumerate(contradictions4, 1):
        print(f"   {i}. {contradiction}")
    print("✅ 예상: 활발함 + 유머 부족이 드러나야 함")
    print()
    
    print("🎯 극단 테스트 결론:")
    print("사용자가 조정한 극단적인 성격 수치들이")
    print("모순적 특성에서 구체적인 말투와 행동으로 드러나는지 확인!")

if __name__ == "__main__":
    test_extreme_personality_contradictions() 