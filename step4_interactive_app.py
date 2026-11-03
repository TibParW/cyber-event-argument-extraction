import sys
import re
import json
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "secbert_wra"

# รหัสสี ANSI สำหรับตกแต่ง Terminal
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED, C_BLUE, C_YELLOW = "\033[1;31m", "\033[1;34m", "\033[1;33m"
C_GREEN, C_MAGENTA, C_CYAN = "\033[1;32m", "\033[1;35m", "\033[1;36m"

SAMPLE_1 = "On Friday, the LockBit ransomware group claimed responsibility for encrypting internal databases at Crosshouse Hospital. The attackers demanded a $2 million ransom, threatening to leak patient records and social security numbers. Security researchers discovered that the breach exploited a critical vulnerability CVE-2023-34362 in the file transfer software."
SAMPLE_2 = "A Chinese threat actor codenamed UTA0565 has been observed exploiting the recently disclosed Google Chrome-Microsoft Windows exploit chain as zero-days through fake websites. The attacks involved the chaining of two vulnerabilities in Chrome (CVE-2026-85046, CVE-2026-87491) and one impacting Windows ALPC (CVE-2026-85880) to achieve remote code execution."
SAMPLE_3 = "The U.S. CISA added two critical Citrix NetScaler ADC and Gateway flaws to its Known Exploited Vulnerabilities catalog: CVE-2026-88771 and CVE-2026-88772 allowing remote code execution. Organizations are required to apply the fixes by September 30."


# ==============================================================================
# ฟังก์ชันประมวลผลข้อความและสกัด Information Extraction (SecBERT NLP Pipeline)
# ==============================================================================
def split_into_sentences(text):
    """หั่นข่าวออกเป็นประโยค โดยไม่หั่นตัวเลขทศนิยมหรือ CVE ขาด"""
    matches = [m.group().strip() for m in re.finditer(r'[^.!?\n]+(?:[.!?\n]|$)', text)]
    return [s for s in matches if len(s) > 4] or [text.strip()]


def is_valid_entity(ent):
    """กรอง False Positives เช่น คำว่า 'a' หรือ stop words เดี่ยวๆ ที่โมเดลทายผิดเป็น Price หรือ Trigger"""
    if not ent or "text" not in ent:
        return False
    txt, lbl = ent["text"].strip().lower(), ent.get("label", "")
    if txt in ["a", "an", "the", ".", ",", "at", "to", "in", "of", "and", "that", "t", "th"]:
        return False
    if lbl in ["Price", "Payment-Amount", "Money"] and not any(ch.isdigit() or ch == '$' for ch in ent["text"]):
        return False
    return not (len(txt) <= 1 and not txt.isdigit())


def extract_entities_from_sentence(sentence, model, tokenizer, id2label, device="cpu"):
    """ส่ง 1 ประโยคเข้า SecBERT และต่อ Subwords กลับเข้าด้วยกันอย่างถูกต้อง"""
    inputs = {k: v.to(device) for k, v in tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).items()}
    with torch.no_grad():
        preds = torch.argmax(model(**inputs).logits, dim=2)[0].cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities, cur = [], None

    for tok, pred_id in zip(tokens, preds):
        if tok in tokenizer.all_special_tokens:
            continue
        label = id2label.get(str(pred_id), id2label.get(pred_id, "O"))
        clean_tok = tok.replace("##", "")

        if tok.startswith("##") and cur:
            cur["text"] += clean_tok
            continue

        if label.startswith("B-"):
            if cur and is_valid_entity(cur):
                entities.append(cur)
            cur = {"text": clean_tok, "label": label[2:], "type": "Trigger" if label[2:].startswith("Trigger-") else "Argument"}
        elif label.startswith("I-") and cur and label[2:] == cur["label"]:
            cur["text"] += " " + clean_tok
        else:
            if cur and is_valid_entity(cur):
                entities.append(cur)
            cur = None

    if cur and is_valid_entity(cur):
        entities.append(cur)
    return entities


def structure_cyber_events(entities_per_sent, original_text):
    """จัดกลุ่ม Entity ในแต่ละประโยคให้เป็น Event Hopper ตามมาตรฐาน CASIE"""
    events, all_args, all_trigs = [], [], []

    for sent_idx, entities in enumerate(entities_per_sent):
        sent_trigs = [e for e in entities if e["type"] == "Trigger"]
        sent_args = [e for e in entities if e["type"] == "Argument"]
        all_args.extend(sent_args)
        all_trigs.extend(sent_trigs)

        if sent_trigs:
            for trig in sent_trigs:
                events.append({
                    "event_id": f"E{len(events) + 1}",
                    "subtype": trig["label"].replace("Trigger-", ""),
                    "trigger_phrase": trig["text"],
                    "arguments": [{"role": a["label"], "text": a["text"]} for a in sent_args]
                })
        elif sent_args and events:
            events[-1]["arguments"].extend([{"role": a["label"], "text": a["text"]} for a in sent_args])

    return {
        "summary": {"total_events_detected": len(events), "total_arguments_detected": len(all_args)},
        "extracted_events": events,
        "all_entities": all_trigs + all_args
    }


def make_badge(text, role, is_html=False):
    """สร้างป้ายกำกับบทบาทของ Entity เช่น [ATTACKER], [VICTIM], [CVE]"""
    role_u = role.upper().replace(" ", "_")
    if not is_html:
        c = C_BLUE if "ATTACKER" in role_u else (C_YELLOW if "VICTIM" in role_u else (C_GREEN if "CVE" in role_u or "VULNERAB" in role_u else (C_CYAN if "PRICE" in role_u else C_MAGENTA)))
        return f"{c}{text} [{role_u}]{C_RESET}"

    colors = {
        "ATTACKER": ("#e0e7ff", "#3730a3", "#4f46e5"), "VICTIM": ("#ccfbf1", "#115e59", "#0d9488"),
        "CVE": ("#fee2e2", "#991b1b", "#dc2626"), "VULNERABILITY": ("#fce7f3", "#9d174d", "#db2777"),
        "VULNERABLE_SYSTEM": ("#ffedd5", "#9a3412", "#ea580c"), "PRICE": ("#fef9c3", "#854d0e", "#ca8a04"),
        "COMPROMISED-DATA": ("#fae8ff", "#86198f", "#c026d3"), "ATTACK-PATTERN": ("#ede9fe", "#5b21b6", "#7c3aed")
    }
    bg, fg, tag_bg = colors.get(role_u, ("#f1f5f9", "#334155", "#64748b"))
    return f"<span style='display:inline-flex;align-items:baseline;background:{bg};color:{fg};padding:1px 6px;border-radius:4px;font-weight:500;margin:0 2px;'>{text}<span style='margin-left:4px;font-size:10px;font-weight:700;background:{tag_bg};color:#fff;padding:1px 4px;border-radius:3px;'>{role_u}</span></span>"


def generate_executive_narrative(events, all_entities, is_html=False):
    """สังเคราะห์ข้อความสรุปใจความสำคัญ (Executive Summary) พร้อมป้ายบทบาท"""
    if not events and not all_entities:
        return "No confirmed cybersecurity incident details could be synthesized from this text."

    def clean_text(t):
        return re.sub(r'\$\s+', '$', re.sub(r'(?i)\bcve\s*-\s*(\d{4})\s*-\s*(\d{4,7})\b', r'CVE-\1-\2', t)).strip()

    def get_unique(roles, is_price=False):
        vals = []
        for e in all_entities:
            if e["label"] in roles:
                v = clean_text(e["text"])
                if v.lower() in ["a", "an", "the", "."] or (is_price and not any(ch.isdigit() or ch == '$' for ch in v)):
                    continue
                if v and v not in vals:
                    vals.append(v)
        return vals

    render = lambda items, r: ", ".join(make_badge(it, r, is_html=is_html) for it in items)
    attackers, victims = get_unique(["Attacker"]), get_unique(["Victim"])
    cves, systems = get_unique(["CVE"]), get_unique(["Vulnerable_System"])
    data, prices = get_unique(["Compromised-Data"]), get_unique(["Price", "Payment-Amount", "Money"], is_price=True)

    clauses = []
    if attackers and victims: clauses.append(f"Threat actor {render(attackers, 'Attacker')} targeted {render(victims, 'Victim')}")
    elif attackers: clauses.append(f"Threat actor {render(attackers, 'Attacker')} was detected in an active campaign")
    elif victims: clauses.append(f"Cyber attack identified affecting victim {render(victims, 'Victim')}")
    else: clauses.append("Security event activity detected")

    if cves and systems: clauses.append(f"exploiting vulnerability {render(cves, 'CVE')} in {render(systems, 'Vulnerable_System')}")
    elif cves: clauses.append(f"exploiting vulnerability {render(cves, 'CVE')}")
    elif systems: clauses.append(f"affecting vulnerable system {render(systems, 'Vulnerable_System')}")

    if data: clauses.append(f"compromising sensitive data ({render(data, 'Compromised-Data')})")
    if prices: clauses.append(f"with an extortion/ransom demand of {render(prices, 'Price')}")

    return ", ".join(clauses) + "."


# ==============================================================================
# ฟังก์ชันแสดงผลบน Command Line (CLI Dashboard)
# ==============================================================================
def print_cli_dashboard(result, original_text):
    """แสดงรายงานฉบับกระชับบน Terminal (เน้น Executive Summary และภาพรวม)"""
    print("\n" + "=" * 80)
    print(f" {C_BOLD}CASIE CYBER THREAT INTELLIGENCE REPORT (Incident Analysis){C_RESET}")
    print("=" * 80 + f"\n\n{C_BOLD}[Original News Text]:{C_RESET}\n  \"{original_text.strip()}\"\n" + "-" * 80)

    events, all_ents = result["extracted_events"], result["all_entities"]
    print(f"{C_BOLD}[Executive Summary]:{C_RESET}\n  {generate_executive_narrative(events, all_ents, is_html=False)}\n" + "-" * 80)

    st_counts = {}
    for ev in events: st_counts[ev["subtype"]] = st_counts.get(ev["subtype"], 0) + 1
    st_str = ", ".join(f"{k}: {v}" for k, v in st_counts.items()) if st_counts else "None"
    n_trig = sum(1 for e in all_ents if e.get('type') == 'Trigger')

    print(f"{C_BOLD}[Intelligence Overview]:{C_RESET}")
    print(f"  - Extracted Events    : {len(events)} event{'s' if len(events) != 1 else ''} ({st_str})")
    print(f"  - Extracted Entities  : {len(all_ents)} items ({n_trig} triggers, {len(all_ents) - n_trig} arguments)")
    print("=" * 80)


def print_detailed_event_hoppers(result, original_text=None):
    """แสดงรายละเอียดทั้งหมดของ Event Hoppers, Entities และ JSON เมื่อกดปุ่มลัด 'e'"""
    events, all_ents = result["extracted_events"], result["all_entities"]
    print("\n" + "=" * 80 + f"\n {C_BOLD}CASIE DETAILED EVENT HOPPERS & ENTITIES (Expanded View){C_RESET}\n" + "=" * 80)

    print(f"\n{C_BOLD}[Extracted Event Hoppers ({len(events)} events)]:{C_RESET}")
    for ev in events:
        print(f"\n  - {C_BOLD}[EVENT {ev['event_id']}]{C_RESET}: Subtype -> {C_RED}{ev['subtype']}{C_RESET} (Trigger: \"{C_RED}{ev['trigger_phrase']}{C_RESET}\")\n    Arguments & Roles:")
        for arg in ev["arguments"]:
            r = arg["role"]
            c = C_BLUE if "Attacker" in r else (C_YELLOW if "Victim" in r else (C_GREEN if "CVE" in r or "Tool" in r else C_CYAN))
            print(f"      - {c}{r:<22}{C_RESET} : {arg['text']}")
        if not ev["arguments"]:
            print("      - No direct argument in same sentence")

    print("\n" + "-" * 80 + f"\n{C_BOLD}[Extracted Entities ({len(all_ents)} items)]:{C_RESET}\n  {'Index':<6} {'Text':<35} {'Role / Label'}\n  " + "-" * 70)
    for idx, ent in enumerate(all_ents, 1):
        lbl = ent['label']
        is_t = ent['type'] == 'Trigger'
        c = C_RED if is_t else (C_BLUE if 'Attacker' in lbl else (C_YELLOW if 'Victim' in lbl else C_GREEN))
        print(f"  {idx:<6} {c}{ent['text']:<35}{C_RESET} {'Trigger' if is_t else 'Role'} ({lbl})")

    print("\n" + "-" * 80 + f"\n{C_BOLD}[Structured JSON Output (CASIE Schema)]:{C_RESET}")
    print(json.dumps(result, indent=2, ensure_ascii=False) + "\n" + "=" * 80)


# ==============================================================================
# ฟังก์ชันช่วยเหลือการรับข้อความ (Clipboard & Multi-line Console Input)
# ==============================================================================
def get_clipboard_text():
    """ดึงข้อความจาก Windows Clipboard หากมี"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return text.strip() if text else ""
    except Exception:
        return ""


def drain_stdin_burst(initial_line=""):
    """กวาดรวบข้อความทุกบรรทัดที่ถูก Paste เข้ามาพร้อมกันใน console buffer"""
    lines = [initial_line] if initial_line else []
    try:
        import msvcrt
        while msvcrt.kbhit():
            line = sys.stdin.readline()
            if not line: break
            lines.append(line.rstrip('\r\n'))
    except Exception: pass
    return "\n".join(lines).strip()


def read_multiline_input():
    """รับข้อความหลายบรรทัดใน Terminal โดยไม่หลุดเมื่อเจอการเว้นบรรทัด"""
    print("\n" + "-" * 80 + f"\n{C_BOLD}พิมพ์หรือวางข้อความข่าวลงใน Terminal (Manual Multi-line Paste Mode):{C_RESET}")
    print("  - รองรับข้อความขนาดยาวและหลายย่อหน้า (กด Ctrl+V ได้เลย)")
    print("  - เมื่อวางเสร็จ ให้กด Enter แล้วพิมพ์ 'END' ในบรรทัดใหม่ (หรือกด Enter ว่าง 2 ครั้ง)\n" + "-" * 80)

    lines, empty_count = [], 0
    while True:
        try: line = input()
        except (EOFError, KeyboardInterrupt): break
        s = line.strip()
        if s.upper() in ["END", "DONE", "EOF", ":W"]: break

        is_burst = False
        try:
            import msvcrt
            is_burst = msvcrt.kbhit()
        except Exception: pass

        if not s:
            if is_burst:
                if lines: lines.append("")
            else:
                empty_count += 1
                if lines and empty_count >= 2:
                    while lines and lines[-1] == "": lines.pop()
                    break
                if lines: lines.append("")
        else:
            empty_count = 0
            lines.append(line)

    return "\n".join(lines).strip()


# ==============================================================================
# วงจรหลัก: Interactive Terminal CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Step 4: SecBERT Interactive Inference & Deployment")
    parser.add_argument("--web", action="store_true", help="เปิดโหมด Web Application (Step 5) ทันที")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"Error: Model directory not found: {MODEL_DIR.resolve()}\n   Please ensure 'secbert_wra' exists in models/")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80 + f"\n  {C_BOLD}[Step 4] SecBERT Cyber Threat Intelligence Extraction (Interactive CLI){C_RESET}")
    print(f"  Device    : {device.upper()}\n  Model Dir : {MODEL_DIR.name}\n" + "=" * 80)

    print("\nLoading SecBERT model and label mappings...")
    with open(MODEL_DIR / "label_mapping.json", 'r', encoding='utf-8') as f:
        id2label = json.load(f)["id2label"]

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()
    print("Model ready!\n")

    if args.web:
        from step5_web_app import launch_web_ui
        launch_web_ui(model, tokenizer, id2label, device)
        return

    last_analysis = {"result": None, "news_text": ""}

    while True:
        print("\n" + "=" * 80 + f"\n{C_BOLD}Select Input Mode (Interactive Menu):{C_RESET}\n" + "=" * 80)
        print(f"  {C_GREEN}[1]{C_RESET} ป้อนหรือวางข้อความข่าว (รองรับ Clipboard อัตโนมัติ & หลายย่อหน้า)")
        print(f"  {C_GREEN}[2]{C_RESET} โหลดข่าวจากไฟล์ข้อความ (.txt)")
        print(f"  {C_RED}[q]{C_RESET} ออกจากโปรแกรม\n" + "-" * 80)

        raw_choice = input(f"{C_BOLD}เลือกตัวเลือก (1/2/q): {C_RESET}").strip()
        if not raw_choice: continue
        if raw_choice.lower() in ['q', 'exit', 'quit']:
            print(f"\n{C_BOLD}Exiting program. Goodbye!{C_RESET}\n")
            break

        if raw_choice.lower() in ['w', 'web']:
            from step5_web_app import launch_web_ui
            launch_web_ui(model, tokenizer, id2label, device)
            break
        if raw_choice.lower() in ['e', 'event', 'events', 'hopper', 'hoppers']:
            if not last_analysis["result"]: print(f"\n{C_YELLOW}No incident analyzed yet. Please analyze news via [1] or [2] first.{C_RESET}")
            else: print_detailed_event_hoppers(last_analysis["result"], last_analysis["news_text"])
            continue
        if raw_choice.lower() in ['j', 'json']:
            if not last_analysis["result"]: print(f"\n{C_YELLOW}No incident analyzed yet. Please analyze news via [1] or [2] first.{C_RESET}")
            else: print("\n" + "=" * 80 + f"\n{C_BOLD}[Structured JSON Output (CASIE Schema)]:{C_RESET}\n" + json.dumps(last_analysis["result"], indent=2, ensure_ascii=False) + "\n" + "=" * 80)
            continue

        news_text = ""
        if raw_choice.lower() in ['s1', 'sample1']: news_text = SAMPLE_1
        elif raw_choice.lower() in ['s2', 'sample2']: news_text = SAMPLE_2
        elif raw_choice.lower() in ['s3', 'sample3']: news_text = SAMPLE_3
        elif raw_choice.lower() in ['clip', 'clipboard']:
            news_text = get_clipboard_text()
            if not news_text:
                print("Warning: Clipboard is empty!")
                continue
            print(f"Loaded text from Clipboard ({len(news_text):,} characters)!")

        elif raw_choice == '1':
            clip_content = get_clipboard_text()
            if clip_content and len(clip_content) >= 15:
                preview = clip_content.replace("\n", " ").strip()[:90]
                print("\n" + "-" * 80 + f"\n{C_BOLD}[Option 1] ป้อนเนื้อหาข่าว (Input News Text){C_RESET}\n" + "-" * 80)
                print(f"ตรวจพบข้อความใน Clipboard:\n  \"{C_CYAN}{preview}...{C_RESET}\" ({len(clip_content):,} characters)")
                print(f"\n  {C_GREEN}[Enter]{C_RESET} ใช้วิเคราะห์ข้อความจาก Clipboard นี้ทันที (แนะนำ)\n  {C_YELLOW}[p]{C_RESET}     วางหรือพิมพ์ข้อความใหม่ด้วยตนเองใน Terminal\n  {C_RED}[b]{C_RESET}     ย้อนกลับไปเมนูหลัก\n" + "-" * 80)
                sub = input(f"{C_BOLD}เลือก (Enter/p/b): {C_RESET}").strip().lower()
                if sub in ['b', 'back']: continue
                elif sub in ['', 'y']: news_text = clip_content
                else: news_text = read_multiline_input()
            else:
                news_text = read_multiline_input()

            if not news_text:
                print("Warning: No input detected, please try again.")
                continue

        elif raw_choice == '2':
            p = Path(input("\nEnter path to .txt file: ").strip().strip('"\''))
            if p.exists() and p.is_file():
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    news_text = f.read().strip()
                print(f"File loaded successfully ({len(news_text):,} characters)!")
            else:
                print(f"File not found: {p.resolve()}")
                continue
        else:
            burst = drain_stdin_burst(initial_line=raw_choice)
            if len(burst) > 15:
                print(f"\nDetected direct news paste ({len(burst):,} characters)...")
                news_text = burst
            else:
                print("Invalid choice. Please try again.")
                continue

        print("\nAnalyzing and extracting Cyber Threat Intelligence with SecBERT...")
        sentences = split_into_sentences(news_text)
        entities_per_sent = [extract_entities_from_sentence(s, model, tokenizer, id2label, device=device) for s in sentences]
        result = structure_cyber_events(entities_per_sent, news_text)

        last_analysis["result"] = result
        last_analysis["news_text"] = news_text
        print_cli_dashboard(result, news_text)


if __name__ == "__main__":
    main()
