def create_frontend_view_html(persona):
    """Create HTML representation of the frontend view of the persona"""
    if not persona:
        return "<div class='persona-details'>페르소나가 아직 생성되지 않았습니다.</div>"
    
    name = persona.get("기본정보", {}).get("이름", "Unknown")
    object_type = persona.get("기본정보", {}).get("유형", "Unknown")
    description = persona.get("기본정보", {}).get("설명", "")
    
    # 성격 요약 가져오기
    personality_summary = persona.get("성격요약", {})
    summary_html = ""
    if personality_summary:
        summary_items = []
        for trait, value in personality_summary.items():
            if isinstance(value, (int, float)):
                trait_name = trait
                trait_value = value
                summary_items.append(f"• {trait_name}: {trait_value:.1f}%")
        
        if summary_items:
            summary_html = "<div class='summary-section'><h4>성격 요약</h4><ul>" + "".join([f"<li>{item}</li>" for item in summary_items]) + "</ul></div>"
    
    # Personality traits
    traits_html = ""
    for trait, value in persona.get("성격특성", {}).items():
        traits_html += f"""
        <div class="trait-item">
            <div class="trait-label">{trait}</div>
            <div class="trait-bar-container">
                <div class="trait-bar" style="width: {value}%; background: linear-gradient(90deg, #6366f1, #a5b4fc);"></div>
            </div>
            <div class="trait-value">{value}%</div>
        </div>
        """
    
    # Flaws - 매력적 결함
    flaws = persona.get("매력적결함", [])
    flaws_list = ""
    for flaw in flaws[:4]:  # 최대 4개만 표시
        flaws_list += f"<li>{flaw}</li>"
    
    # 소통 방식
    communication_style = persona.get("소통방식", "")
    
    # 유머 스타일
    humor_style = persona.get("유머스타일", "")
    
    # 전체 HTML 스타일과 내용
    html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    .frontend-persona {{
        font-family: 'Noto Sans KR', sans-serif;
        color: #333;
        max-width: 100%;
    }}
    
    .persona-header {{
        background: linear-gradient(135deg, #6366f1, #a5b4fc);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .persona-header h2 {{
        margin: 0;
        font-size: 24px;
    }}
    
    .persona-header p {{
        margin: 5px 0 0 0;
        opacity: 0.9;
    }}
    
    .persona-section {{
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
        border-bottom: 2px solid #6366f1;
        padding-bottom: 5px;
        display: inline-block;
    }}
    
    .trait-item {{
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }}
    
    .trait-label {{
        width: 80px;
        font-weight: 500;
    }}
    
    .trait-bar-container {{
        flex-grow: 1;
        background: #e0e0e0;
        height: 10px;
        border-radius: 5px;
        margin: 0 10px;
        overflow: hidden;
    }}
    
    .trait-bar {{
        height: 100%;
        border-radius: 5px;
    }}
    
    .trait-value {{
        width: 40px;
        text-align: right;
        font-size: 14px;
    }}
    
    .tags-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }}
    
    .flaw-tag, .contradiction-tag, .interest-tag {{
        background: #f0f4ff;
        border: 1px solid #d0d4ff;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 14px;
        display: inline-block;
    }}
    
    .flaw-tag {{
        background: #fff0f0;
        border-color: #ffd0d0;
    }}
    
    .contradiction-tag {{
        background: #f0fff4;
        border-color: #d0ffd4;
    }}
    
    /* 영혼 각성 UX 스타일 */
    .awakening-result {{
        background: #f9f9ff;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }}
    
    .speech-bubble {{
        background: #fff;
        border-radius: 18px;
        padding: 15px;
        margin-bottom: 15px;
        position: relative;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }}
    
    .speech-bubble:after {{
        content: '';
        position: absolute;
        bottom: -10px;
        left: 30px;
        border-width: 10px 10px 0;
        border-style: solid;
        border-color: #fff transparent;
    }}
    
    .persona-speech {{
        margin: 0;
        font-size: 15px;
        line-height: 1.5;
        color: #4b5563;
    }}
    
    .persona-traits-highlight {{
        background: #f0f4ff;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }}
    
    .persona-traits-highlight h4 {{
        margin-top: 0;
        margin-bottom: 10px;
        color: #4338ca;
    }}
    
    .persona-traits-highlight ul {{
        margin: 0;
        padding-left: 20px;
        color: #4b5563;
    }}
    
    .persona-traits-highlight li {{
        margin-bottom: 5px;
    }}
    
    .first-interaction {{
        margin-top: 20px;
    }}
    
    .interaction-buttons, .confirmation-buttons {{
        display: flex;
        gap: 10px;
        margin-top: 15px;
    }}
    
    .interaction-btn, .confirmation-btn {{
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.2s;
        font-family: 'Noto Sans KR', sans-serif;
    }}
    
    .interaction-btn:hover, .confirmation-btn:hover {{
        background: #e5e7eb;
    }}
    
    .confirmation-btn.primary {{
        background: #6366f1;
        color: white;
        border: 1px solid #4f46e5;
    }}
    
    .confirmation-btn.primary:hover {{
        background: #4f46e5;
    }}
    
    /* 요약 섹션 스타일 */
    .summary-section {{
        background: #f0f4ff;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }}
    
    .summary-section h4 {{
        margin-top: 0;
        margin-bottom: 10px;
        color: #4338ca;
    }}
    
    .summary-section ul {{
        margin: 0;
        padding-left: 20px;
        color: #4b5563;
    }}
    
    .summary-section li {{
        margin-bottom: 5px;
    }}
    </style>
    
    <div class="frontend-persona">
        <div class="persona-header">
            <h2>{name}</h2>
            <p><strong>{object_type}</strong> - {description}</p>
        </div>
        
        {summary_html}
        
        <div class="persona-section">
            <h3 class="section-title">성격 특성</h3>
            <div class="traits-container">
                {traits_html}
            </div>
        </div>
        
        <div class="persona-section">
            <h3 class="section-title">소통 스타일</h3>
            <p>{communication_style}</p>
            <h3 class="section-title" style="margin-top: 15px;">유머 스타일</h3>
            <p>{humor_style}</p>
        </div>
        
        <div class="persona-section">
            <h3 class="section-title">매력적 결함</h3>
            <ul class="flaws-list">
                {flaws_list}
            </ul>
        </div>
    </div>
    """
    
    return html 