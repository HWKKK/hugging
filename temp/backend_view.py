import json

def create_backend_view_html(persona):
    """Create HTML representation of the backend view of the persona"""
    if not persona:
        return "<div class='persona-details'>페르소나가 아직 생성되지 않았습니다.</div>"
    
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    
    # 백엔드 기본 정보
    basic_info = persona.get("기본정보", {})
    basic_info_html = ""
    for key, value in basic_info.items():
        basic_info_html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
    
    # 1. 성격 변수 요약
    personality_summary = persona.get("성격요약", {})
    summary_html = ""
    
    if personality_summary:
        summary_html += "<div class='summary-container'>"
        for category, value in personality_summary.items():
            if isinstance(value, (int, float)):
                summary_html += f"""
                <div class='summary-item'>
                    <div class='summary-label'>{category}</div>
                    <div class='summary-bar-container'>
                        <div class='summary-bar' style='width: {value}%; background: linear-gradient(90deg, #10b981, #6ee7b7);'></div>
                    </div>
                    <div class='summary-value'>{value:.1f}</div>
                </div>
                """
        summary_html += "</div>"
    
    # 2. 성격 매트릭스 (5차원 빅5 시각화)
    big5_html = ""
    if "성격특성" in persona:
        # 빅5 매핑 (기존 특성에서 변환)
        big5 = {
            "외향성(Extraversion)": persona.get("성격특성", {}).get("외향성", 50),
            "친화성(Agreeableness)": persona.get("성격특성", {}).get("온기", 50),
            "성실성(Conscientiousness)": persona.get("성격특성", {}).get("신뢰성", 50),
            "신경증(Neuroticism)": 100 - persona.get("성격특성", {}).get("안정성", 50) if "안정성" in persona.get("성격특성", {}) else 50,
            "개방성(Openness)": persona.get("성격특성", {}).get("창의성", 50)
        }
        
        big5_html = "<div class='big5-matrix'>"
        for trait, value in big5.items():
            big5_html += f"""
            <div class='big5-item'>
                <div class='big5-label'>{trait}</div>
                <div class='big5-bar-container'>
                    <div class='big5-bar' style='width: {value}%;'></div>
                </div>
                <div class='big5-value'>{value}%</div>
            </div>
            """
        big5_html += "</div>"
    
    # 3. 유머 매트릭스
    humor_matrix = persona.get("유머매트릭스", {})
    humor_html = ""
    
    if humor_matrix:
        warmth_vs_wit = humor_matrix.get("warmth_vs_wit", 50)
        self_vs_observational = humor_matrix.get("self_vs_observational", 50)
        subtle_vs_expressive = humor_matrix.get("subtle_vs_expressive", 50)
        
        humor_html = f"""
        <div class='humor-matrix'>
            <div class='humor-dimension'>
                <div class='dimension-label'>따뜻함 vs 위트</div>
                <div class='dimension-bar-container'>
                    <div class='dimension-indicator' style='left: {warmth_vs_wit}%;'></div>
                    <div class='dimension-label-left'>위트</div>
                    <div class='dimension-label-right'>따뜻함</div>
                </div>
            </div>
            
            <div class='humor-dimension'>
                <div class='dimension-label'>자기참조 vs 관찰형</div>
                <div class='dimension-bar-container'>
                    <div class='dimension-indicator' style='left: {self_vs_observational}%;'></div>
                    <div class='dimension-label-left'>관찰형</div>
                    <div class='dimension-label-right'>자기참조</div>
                </div>
            </div>
            
            <div class='humor-dimension'>
                <div class='dimension-label'>미묘함 vs 표현적</div>
                <div class='dimension-bar-container'>
                    <div class='dimension-indicator' style='left: {subtle_vs_expressive}%;'></div>
                    <div class='dimension-label-left'>미묘함</div>
                    <div class='dimension-label-right'>표현적</div>
                </div>
            </div>
        </div>
        """
    
    # 4. 매력적 결함과 모순적 특성
    flaws_html = ""
    contradictions_html = ""
    
    flaws = persona.get("매력적결함", [])
    if flaws:
        flaws_html = "<ul class='flaws-list'>"
        for flaw in flaws:
            flaws_html += f"<li>{flaw}</li>"
        flaws_html += "</ul>"
    
    contradictions = persona.get("모순적특성", [])
    if contradictions:
        contradictions_html = "<ul class='contradictions-list'>"
        for contradiction in contradictions:
            contradictions_html += f"<li>{contradiction}</li>"
        contradictions_html += "</ul>"
    
    # 6. 프롬프트 템플릿 (있는 경우)
    prompt_html = ""
    if "프롬프트" in persona:
        prompt_text = persona.get("프롬프트", "")
        prompt_html = f"""
        <div class='prompt-section'>
            <h3 class='section-title'>대화 프롬프트</h3>
            <pre class='prompt-text'>{prompt_text}</pre>
        </div>
        """
    
    # 7. 완전한 백엔드 JSON (접이식)
    try:
        # 내부 상태 객체 제거 (JSON 변환 불가)
        json_persona = {k: v for k, v in persona.items() if k not in ["personality_profile", "humor_matrix"]}
        persona_json = json.dumps(json_persona, ensure_ascii=False, indent=2)
        
        json_preview = f"""
        <details class='json-details'>
            <summary>전체 백엔드 데이터 (JSON)</summary>
            <pre class='json-preview'>{persona_json}</pre>
        </details>
        """
    except Exception as e:
        json_preview = f"<div class='error'>JSON 변환 오류: {str(e)}</div>"
    
    # 8. 전체 HTML 조합
    html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    .backend-persona {{
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
        max-width: 100%;
    }}
    
    .backend-header {{
        background: linear-gradient(135deg, #059669, #34d399);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .backend-header h2 {{
        margin: 0;
        font-size: 24px;
    }}
    
    .backend-header p {{
        margin: 5px 0 0 0;
        opacity: 0.9;
    }}
    
    .backend-section {{
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #e0e0e0;
    }}
    
    .section-title {{
        font-size: 18px;
        margin: 0 0 10px 0;
        color: #444;
        border-bottom: 2px solid #10b981;
        padding-bottom: 5px;
        display: inline-block;
    }}
    
    /* 기본 정보 테이블 */
    .basic-info-table {{
        width: 100%;
        border-collapse: collapse;
    }}
    
    .basic-info-table td {{
        padding: 8px;
        border-bottom: 1px solid #e0e0e0;
    }}
    
    .basic-info-table td:first-child {{
        width: 120px;
        font-weight: 500;
    }}
    
    /* 요약 스타일 */
    .summary-container {{
        margin-top: 10px;
    }}
    
    .summary-item {{
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }}
    
    .summary-label {{
        width: 150px;
        font-weight: 500;
    }}
    
    .summary-bar-container {{
        flex-grow: 1;
        background: #e0e0e0;
        height: 10px;
        border-radius: 5px;
        margin: 0 10px;
        overflow: hidden;
    }}
    
    .summary-bar {{
        height: 100%;
        border-radius: 5px;
    }}
    
    .summary-value {{
        width: 40px;
        text-align: right;
        font-size: 14px;
    }}
    
    /* 빅5 성격 매트릭스 */
    .big5-matrix {{
        margin-top: 15px;
    }}
    
    .big5-item {{
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }}
    
    .big5-label {{
        width: 150px;
        font-weight: 500;
    }}
    
    .big5-bar-container {{
        flex-grow: 1;
        background: #e0e0e0;
        height: 12px;
        border-radius: 6px;
        margin: 0 10px;
        overflow: hidden;
    }}
    
    .big5-bar {{
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #10b981, #34d399);
    }}
    
    .big5-value {{
        width: 40px;
        text-align: right;
        font-weight: 500;
    }}
    
    /* 유머 매트릭스 스타일 */
    .humor-matrix {{
        margin-top: 15px;
    }}
    
    .humor-dimension {{
        margin-bottom: 20px;
    }}
    
    .dimension-label {{
        font-weight: 500;
        margin-bottom: 5px;
    }}
    
    .dimension-bar-container {{
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        position: relative;
        margin-top: 5px;
    }}
    
    .dimension-indicator {{
        width: 20px;
        height: 20px;
        background: #10b981;
        border-radius: 50%;
        position: absolute;
        top: 0;
        transform: translateX(-50%);
    }}
    
    .dimension-label-left, .dimension-label-right {{
        position: absolute;
        top: -20px;
        font-size: 12px;
        color: #666;
    }}
    
    .dimension-label-left {{
        left: 10px;
    }}
    
    .dimension-label-right {{
        right: 10px;
    }}
    
    /* 매력적 결함 및 모순적 특성 */
    .flaws-list, .contradictions-list {{
        margin: 0;
        padding-left: 20px;
    }}
    
    .flaws-list li, .contradictions-list li {{
        margin-bottom: 6px;
    }}
    
    /* 프롬프트 섹션 */
    .prompt-text {{
        background: #f3f4f6;
        border-radius: 6px;
        padding: 15px;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 14px;
        color: #374151;
        max-height: 400px;
        overflow-y: auto;
    }}
    
    /* JSON 미리보기 스타일 */
    .json-details {{
        margin-top: 15px;
    }}
    
    .json-details summary {{
        cursor: pointer;
        padding: 10px;
        background: #f0f0f0;
        border-radius: 5px;
        font-weight: 500;
    }}
    
    .json-preview {{
        background: #f8f8f8;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-top: 10px;
        overflow-x: auto;
        color: #333;
        font-family: monospace;
        font-size: 14px;
        line-height: 1.5;
        max-height: 400px;
        overflow-y: auto;
    }}
    
    .error {{
        color: #e53e3e;
        padding: 10px;
        background: #fff5f5;
        border-radius: 5px;
        margin-top: 10px;
    }}
    </style>
    
    <div class="backend-persona">
        <div class="backend-header">
            <h2>{name} - 백엔드 데이터</h2>
            <p>상세 정보와 내부 변수 확인</p>
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">기본 정보</h3>
            <table class="basic-info-table">
                {basic_info_html}
            </table>
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">성격 요약 (Big 5)</h3>
            {big5_html}
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">유머 매트릭스 (3차원)</h3>
            {humor_html}
        </div>
        
        <div class="backend-section">
            <h3 class="section-title">매력적 결함</h3>
            {flaws_html}
            
            <h3 class="section-title" style="margin-top: 20px;">모순적 특성</h3>
            {contradictions_html}
        </div>
        
        {prompt_html}
        
        <div class="backend-section">
            <h3 class="section-title">전체 백엔드 데이터</h3>
            {json_preview}
        </div>
    </div>
    """
    
    return html 