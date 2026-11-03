import sys
import re
import json
from pathlib import Path
import torch

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import gradio as gr
from transformers import AutoTokenizer, AutoModelForTokenClassification

# นำเข้าฟังก์ชันประมวลผลและการสกัดข้อมูลจาก Step 4 เพื่อไม่ให้เขียนโค้ดซ้ำซ้อน
from step4_interactive_app import (
    BASE_DIR,
    MODEL_DIR,
    SAMPLE_1,
    SAMPLE_2,
    SAMPLE_3,
    split_into_sentences,
    extract_entities_from_sentence,
    structure_cyber_events,
    generate_executive_narrative,
)


def launch_web_ui(model, tokenizer, id2label, device="cpu", port=7860, inbrowser=True):
    """ฟังก์ชันเปิดบริการ Web Application ด้วย Gradio Dashboard"""
    print("\nStarting Web Application (Gradio Dashboard)...")

    def web_analyze(text):
        if not text or not text.strip():
            return [], "<p>Please enter news text.</p>", "<p>No data</p>", {}

        sentences = split_into_sentences(text)
        entities_per_sent = [
            extract_entities_from_sentence(s, model, tokenizer, id2label, device=device)
            for s in sentences
        ]
        result = structure_cyber_events(entities_per_sent, text)

        # 1. จัดเตรียมข้อมูลสำหรับ HighlightedText (ใช้ word boundary \b เพื่อไม่ให้ตัดคำกลางคัน)
        valid_entities = sorted(result["all_entities"], key=lambda x: len(x['text']), reverse=True)
        all_spans = []
        for ent in valid_entities:
            phrase = ent["text"].strip()
            if len(phrase) < 2:
                continue
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                st, en = m.span()
                if not any(max(st, existing[0]) < min(en, existing[1]) for existing in all_spans):
                    all_spans.append((st, en, ent["label"]))
        all_spans.sort(key=lambda x: x[0])

        # รวมข้อความกับป้ายกำกับสำหรับ HighlightedText
        highlighted = []
        last_idx = 0
        for st, en, lbl in all_spans:
            if st > last_idx:
                highlighted.append((text[last_idx:st], None))
            highlighted.append((text[st:en], lbl))
            last_idx = en
        if last_idx < len(text):
            highlighted.append((text[last_idx:], None))

        # 2. การ์ดสรุป Event Hoppers
        cards_html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"
        for ev in result["extracted_events"]:
            args_list = "".join(f"<li><b>{a['role']}:</b> {a['text']}</li>" for a in ev["arguments"])
            cards_html += f"""
            <div style='background: #ffffff; border: 1px solid #bfdbfe; border-left: 5px solid #2563eb; border-radius: 8px; padding: 14px;'>
                <h4 style='margin: 0 0 8px 0; color: #1e40af;'>Event {ev['event_id']}: {ev['subtype']}</h4>
                <p style='margin: 0 0 6px 0;'><b>Trigger Phrase:</b> <span style='color: #dc2626;'>{ev['trigger_phrase']}</span></p>
                <ul style='margin: 0; padding-left: 20px;'>{args_list if args_list else '<li>No direct argument in same sentence</li>'}</ul>
            </div>
            """
        cards_html += "</div>"

        # 3. สรุปใจความภาษาอังกฤษ พร้อมป้ายกำกับบทบาทสไตล์เดียวกับ Highlighted Text
        narrative = generate_executive_narrative(result["extracted_events"], result["all_entities"], is_html=True)

        summary_html = f"""
        <div style='background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 18px; color: #0f172a; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='font-size: 12px; font-weight: 700; color: #1e40af; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px;'>Executive Incident Summary</div>
            <p style='margin: 0; font-size: 15px; line-height: 1.9;'>{narrative}</p>
        </div>
        """

        return highlighted, cards_html, summary_html, result

    with gr.Blocks(title="SecBERT Cyber Threat Intelligence Extractor") as demo:
        gr.HTML("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin: 0;">CASIE: Cyber Threat Intelligence Extractor (Step 5)</h2>
            <p style="margin: 6px 0 0 0; opacity: 0.9;">Fine-Tuned SecBERT for End-to-End Cyber Event & Argument Extraction</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 1. Input News")
                inp_text = gr.Textbox(lines=8, label="Cybersecurity News Text", value=SAMPLE_1)
                with gr.Row():
                    btn_run = gr.Button("Analyze News", variant="primary")
                    btn_clr = gr.Button("Clear")

                gr.Markdown("#### Sample News Presets:")
                with gr.Row():
                    b1 = gr.Button("Sample 1: LockBit", size="sm")
                    b2 = gr.Button("Sample 2: Zero-Day", size="sm")
                    b3 = gr.Button("Sample 3: Citrix", size="sm")

            with gr.Column(scale=7):
                gr.Markdown("### 2. Threat Intelligence Output")
                out_summary = gr.HTML()
                with gr.Tabs():
                    with gr.TabItem("Highlighted Text"):
                        out_hl = gr.HighlightedText(label="Entities in Context")
                    with gr.TabItem("Event Hoppers"):
                        out_cards = gr.HTML()
                    with gr.TabItem("Structured JSON"):
                        out_json = gr.JSON()

        btn_run.click(web_analyze, inputs=[inp_text], outputs=[out_hl, out_cards, out_summary, out_json])
        b1.click(lambda: SAMPLE_1, outputs=[inp_text])
        b2.click(lambda: SAMPLE_2, outputs=[inp_text])
        b3.click(lambda: SAMPLE_3, outputs=[inp_text])
        btn_clr.click(lambda: "", outputs=[inp_text])

    print(f"Web UI started! Opening browser at http://127.0.0.1:{port} ...")
    demo.launch(server_port=port, inbrowser=inbrowser)


def main():
    """ฟังก์ชันหลักสำหรับรัน Step 5 ในรูปแบบเว็บแอปโดยตรง"""
    if not MODEL_DIR.exists():
        print(f"Error: Model directory not found: {MODEL_DIR.resolve()}")
        print("   Please ensure 'secbert_wra' exists in models/")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("  [Step 5] SecBERT Cyber Threat Intelligence Extractor (Web Application)")
    print(f"  Device    : {device.upper()}")
    print(f"  Model Dir : {MODEL_DIR.name}")
    print("=" * 80)

    print("\nLoading SecBERT model and label mappings...")
    with open(MODEL_DIR / "label_mapping.json", 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    id2label = mapping["id2label"]

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()
    print("Model ready!\n")

    launch_web_ui(model, tokenizer, id2label, device=device)


if __name__ == "__main__":
    main()
