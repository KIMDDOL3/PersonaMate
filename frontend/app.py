import os, json, time, io, requests, numpy as np, pandas as pd
import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

# 확실히 환경 변수를 덮어쓰도록 수정
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"), override=True)
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BACKEND = os.getenv('BACKEND_URL','http://localhost:9000')
NANUM = '/mnt/data/NanumGothic.ttf'
if os.path.exists(NANUM):
    try: pdfmetrics.registerFont(TTFont('NanumGothic', NANUM)); FONT='NanumGothic'
    except Exception: FONT='Helvetica'
else: FONT='Helvetica'

CATALOG=[
  {'id':'yt_ai_001','platform':'youtube','name':'AI Explained','tags':['ai','ml','research','news'],'url':'https://youtube.com/@ai_expl'},
  {'id':'yt_ds_001','platform':'youtube','name':'Data School','tags':['data','python','pandas','sklearn'],'url':'https://youtube.com/@dataschool'},
  {'id':'ig_photo_001','platform':'instagram','name':'Urban Lens','tags':['photography','city','street'],'url':'https://instagram.com/urbanlens'},
  {'id':'x_genai_001','platform':'x','name':'GenAI Daily','tags':['genai','llm','startup'],'url':'https://x.com/genaidaily'},
  {'id':'yt_product_001','platform':'youtube','name':'Product Bytes','tags':['product','ux','startup'],'url':'https://youtube.com/@productbytes'},
  {'id':'ig_fitness_001','platform':'instagram','name':'Daily Mobility','tags':['fitness','mobility','stretching'],'url':'https://instagram.com/daily.mobility'}
]

def openai_embed(texts):
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        r = client.embeddings.create(model='text-embedding-3-large', input=texts)
        import numpy as np
        X = np.array([d.embedding for d in r.data], dtype='float32')
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return X
    except Exception as e:
        print("OpenAI 임베딩 실패:", e)
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer('all-MiniLM-L6-v2')
        return m.encode(texts, normalize_embeddings=True)

def explain(item, hints):
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        prompt=f"사용자 관심 힌트: {hints}\n추천 후보: {item}\n왜 이 후보를 추천하는지 한국어로 2문장으로 설명."
        r = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], max_tokens=160)
        return r.choices[0].message.content
    except Exception as e:
        print("OpenAI 설명 생성 실패:", e)
        return '태그/주제 유사도가 높아 추천합니다.'

def mmr(q, X, lam=0.7, top_k=6):
    sel=[]; import numpy as np; sim_q=(q@X.T).ravel()
    while len(sel)<min(top_k,len(X)):
        if not sel: sel.append(int(np.argmax(sim_q))); continue
        sim_sel = np.max(X@X[sel].T, axis=1); score = lam*sim_q - (1-lam)*sim_sel
        for i in sel: score[i] = -1e9
        sel.append(int(np.argmax(score)))
    return sel

def radar_for_mbti(mbti:str):
    dims=['E/I','N/S','T/F','J/P']
    vals=[1.0 if mbti and mbti.upper()[0]=='E' else 0.0,
          1.0 if mbti and mbti.upper()[1]=='N' else 0.0,
          1.0 if mbti and mbti.upper()[2]=='T' else 0.0,
          1.0 if mbti and mbti.upper()[3]=='J' else 0.0]
    fig = go.Figure(data=go.Scatterpolar(r=vals+vals[:1], theta=dims+dims[:1], fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=False, margin=dict(l=20,r=20,t=20,b=20), width=450, height=350)
    return fig

def collect_from_platforms(do_yt, do_ig, do_x):
    data={}
    if do_yt:
        try:
            res=requests.get(f"{BACKEND}/youtube/subscriptions", timeout=30).json()
            data['youtube']=[s['name'] for s in res.get('subscriptions',[])]
        except Exception as e: data['youtube_error']=str(e)
    if do_ig:
        try:
            res=requests.get(f"{BACKEND}/instagram/media", timeout=30).json()
            names=[]; 
            for m in res.get('data',[])[:50]:
                cap=(m.get('caption') or '').split()
                names.extend([w.strip('#') for w in cap if w.startswith('#')])
            data['instagram']=names[:20]
        except Exception as e: data['instagram_error']=str(e)
    if do_x:
        try:
            res=requests.get(f"{BACKEND}/x/me_following", timeout=30).json()
            data['x']=[u.get('username') for u in (res.get('data') or [])]
        except Exception as e: data['x_error']=str(e)
    return data

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json

def recommend_pipeline(yt, sns, mbti, use_openai=True, top_k=6):
    chs=[s.strip() for s in (yt or '').splitlines() if s.strip()]
    acc=[s.strip() for s in (sns or '').splitlines() if s.strip()]
    # 구조화된 JSON으로 사용자 데이터 전달
    user_profile = {
        "youtube_subscriptions": chs,
        "sns_keywords": acc,
        "mbti": mbti
    }
    
    if use_openai:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = ChatPromptTemplate.from_template("""
        당신은 추천 시스템입니다.
        아래 사용자의 실제 데이터(JSON)와 MBTI를 기반으로 **새로운 사이트 15개**를 추천하세요.
        - 유튜브 채널과 웹사이트를 혼합해서 추천할 것
        - 한국 채널/사이트와 영어권 채널/사이트를 함께 추천할 것
        - 최소 5개 이상은 한국 채널/사이트여야 함
        - 반드시 JSON 배열 형식으로 출력할 것
        - 각 항목은 {{ "name": ..., "platform": "youtube" 또는 "web", "url": ..., "tags": "...", "score": 0.0~1.0, "reason": ... }} 형식일 것
        - 추천 사유는 반드시 한국어로 작성할 것
        - 비어 있는 값 없이 모든 필드를 채울 것
        - **사용자가 이미 구독한 채널/사이트는 절대 추천하지 말 것**
        - 즉, 아래 JSON의 youtube_subscriptions와 sns_keywords는 참고용 데이터이며, 추천 결과에 포함되면 안 됨

        사용자 데이터(JSON):
        {user_profile}
        """)
        chain = prompt | llm
        response = chain.invoke({"user_profile": json.dumps(user_profile, ensure_ascii=False)})
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            data = json.loads(content)
            return data[:top_k]
        except Exception as e:
            return [{"name":"추천 실패","platform":"error","url":"","tags":"","score":0.0,"reason":f"파싱 실패: {str(e)}\n원본: {response.content}"}]
    else:
        return []

def export_json(cards):
    path=f"./recommendations_{int(time.time())}.json"
    with open(path,'w',encoding='utf-8') as f: json.dump(cards, f, ensure_ascii=False, indent=2)
    return path
def export_html(cards, mbti):
    html=["<html><head><meta charset='utf-8'><title>PersonaMate Report</title></head><body>"]
    html.append(f"<h2>추천 결과 (MBTI: {mbti})</h2><ol>")
    for c in cards:
        html.append(f"<li><a href='{c['url']}' target='_blank'>{c['name']}</a> ({c['platform']}) — 점수 {c['score']:.3f}<br><small>{c['reason']}</small></li>")
    html.append("</ol></body></html>")
    path=f"./recommendations_{int(time.time())}.html"
    with open(path,'w',encoding='utf-8') as f: f.write('\n'.join(html))
    return path
def export_pdf(cards, mbti):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    cpath=f"./recommendations_{int(time.time())}.pdf"
    c=canvas.Canvas(cpath, pagesize=A4); W,H=A4
    c.setFont(FONT, 14); c.drawString(40, H-50, f"PersonaMate 추천 리포트 (MBTI: {mbti})")
    y=H-90; c.setFont(FONT, 11)
    for i,card in enumerate(cards, start=1):
        c.drawString(40, y, f"{i}. {card['name']} ({card['platform']})  점수 {card['score']:.3f}"); y-=16
        for line in [card['url'], card['reason']]:
            for chunk in [line[j:j+90] for j in range(0, len(line), 90)]:
                c.drawString(60,y,chunk); y-=14
        y-=6
        if y<80: c.showPage(); y=H-80; c.setFont(FONT,11)
    c.save(); return cpath

with gr.Blocks(title='PersonaMate Pro (OAuth + Advanced UI)') as demo:
    gr.Markdown('## PersonaMate Pro — OAuth 수집 + 고급 추천 UI')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 1) OAuth 로그인')
            btn_google=gr.Button('Google (YouTube) 로그인 열기')
            btn_instagram=gr.Button('Instagram 로그인 열기')
            btn_x=gr.Button('X (Twitter) 로그인 열기')
        with gr.Column(scale=2):
            gr.Markdown('### 2) 자동 수집')
            yt_chk=gr.Checkbox(label='YouTube 구독 목록 사용', value=True)
            ig_chk=gr.Checkbox(label='Instagram 해시태그 사용', value=False)
            x_chk=gr.Checkbox(label='X 팔로잉 사용자명 사용', value=False)
            fetch_btn=gr.Button('내 계정에서 데이터 수집')
            collected=gr.JSON(label='수집 내용 미리보기')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 3) 입력/MBTI')
            yt_text=gr.Textbox(lines=6,label='유튜브 구독 (수집/수동 혼용)')
            sns_text=gr.Textbox(lines=6,label='SNS 키워드/계정 (수집/수동 혼용)')
            mbti=gr.Dropdown(choices=['ISTJ','ISFJ','INFJ','INTJ','ISTP','ISFP','INFP','INTP','ESTP','ESFP','ENFP','ENTP','ESTJ','ESFJ','ENFJ','ENTJ'], value='ENFP', label='MBTI')
            use_openai=gr.Checkbox(label='OpenAI 임베딩 사용', value=True)
            run_btn=gr.Button('분석 & 추천 실행', variant='primary')
        with gr.Column(scale=2):
            gr.Markdown('### 4) 결과')
            chart=gr.Plot(label='MBTI 레이더')
            table=gr.Dataframe(headers=['이름','플랫폼','URL(클릭)','태그','점수','추천 사유'], row_count=6, col_count=6, wrap=True, interactive=False)
            links=gr.HTML('<i>여기에 클릭 가능한 바로가기를 생성합니다.</i>')
            save_json=gr.Button('JSON 저장')
            save_html=gr.Button('HTML 공유 페이지 저장')
            save_pdf=gr.Button('PDF 리포트 저장')
            json_file=gr.File(label='JSON 파일')
            html_file=gr.File(label='HTML 파일')
            pdf_file=gr.File(label='PDF 파일')
    import webbrowser
    def open_g():
        webbrowser.open(f"{BACKEND}/oauth/google/start")
    def open_i():
        webbrowser.open(f"{BACKEND}/oauth/instagram/start")
    def open_x():
        webbrowser.open(f"{BACKEND}/oauth/x/start")
    btn_google.click(fn=open_g, inputs=[], outputs=[])
    btn_instagram.click(fn=open_i, inputs=[], outputs=[])
    btn_x.click(fn=open_x, inputs=[], outputs=[])
    def _fetch(do_yt,do_ig,do_x):
        d=collect_from_platforms(do_yt,do_ig,do_x)
        return d, '\n'.join((d.get('youtube') or [])[:10]), '\n'.join((d.get('instagram') or [])[:10]), '\n'.join((d.get('x') or [])[:10])
    fetch_btn.click(_fetch, [yt_chk,ig_chk,x_chk], [collected, yt_text, sns_text, sns_text])
    raw=gr.State([])
    def _run(yt,sns,mbti,use_openai):
        cards = recommend_pipeline(yt, sns, mbti, use_openai=use_openai, top_k=15)
        rows = [[
            c.get('name',''),
            c.get('platform',''),
            c.get('url',''),
            c.get('tags',''),
            round(c.get('score',0),3) if 'score' in c else '',
            c.get('reason','')
        ] for c in cards]
        link_html = '<ul>' + ''.join([
            f"<li><a href='{c.get('url','')}' target='_blank'>{c.get('name','')}</a> <small>({c.get('platform','')})</small></li>"
            for c in cards
        ]) + '</ul>'
        return radar_for_mbti(mbti), rows, link_html, cards
    run_btn.click(_run, [yt_text, sns_text, mbti, use_openai], [chart, table, links, raw])
    def _save_json(cards): 
        if not cards: return None
        import time, json
        p=f"./recommendations_{int(time.time())}.json"
        open(p,'w',encoding='utf-8').write(json.dumps(cards, ensure_ascii=False, indent=2)); return p
    def _save_html(cards, mbti):
        if not cards: return None
        import time
        html=['<html><head><meta charset=\'utf-8\'><title>PersonaMate Report</title></head><body>']
        html.append(f"<h2>추천 결과 (MBTI: {mbti})</h2><ol>")
        for c in cards:
            html.append(f"<li><a href='{c['url']}' target='_blank'>{c['name']}</a> ({c['platform']}) — 점수 {c['score']:.3f}<br><small>{c['reason']}</small></li>")
        html.append('</ol></body></html>')
        p=f"./recommendations_{int(time.time())}.html"
        open(p,'w',encoding='utf-8').write('\n'.join(html)); return p
    def _save_pdf(cards, mbti):
        if not cards: return None
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        import time
        p=f"./recommendations_{int(time.time())}.pdf"
        c=canvas.Canvas(p, pagesize=A4); W,H=A4; c.setFont(FONT,14); c.drawString(40,H-50,f"PersonaMate 추천 리포트 (MBTI: {mbti})")
        y=H-90; c.setFont(FONT,11)
        for i,card in enumerate(cards, start=1):
            c.drawString(40,y,f"{i}. {card['name']} ({card['platform']})  점수 {card['score']:.3f}"); y-=16
            for line in [card['url'], card['reason']]:
                for chunk in [line[j:j+90] for j in range(0,len(line),90)]:
                    c.drawString(60,y,chunk); y-=14
            y-=6
            if y<80: c.showPage(); y=H-80; c.setFont(FONT,11)
        c.save(); return p
    save_json.click(_save_json, [raw], [json_file])
    save_html.click(_save_html, [raw, mbti], [html_file])
    save_pdf.click(_save_pdf, [raw, mbti], [pdf_file])
if __name__=='__main__':
    demo.launch(share=True)
