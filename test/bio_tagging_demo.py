import sys
import json
import re
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def split_sentences_with_offsets(text):
    """
    1. หั่นเนื้อข่าวออกเป็นแต่ละประโยค พร้อมจำพิกัด [sent_start : sent_end]
    """
    sentences = []
    # ตัดประโยคด้วยเครื่องหมาย . ! ? หรือขึ้นบรรทัดใหม่
    for match in re.finditer(r'[^.!?\n]+[.!?\n]?', text):
        s = match.group()
        if not s.strip():
            continue
        start, end = match.span()
        l_strip = len(s) - len(s.lstrip())
        r_strip = len(s) - len(s.rstrip())
        sent_start = start + l_strip
        sent_end = end - r_strip
        sent_text = text[sent_start:sent_end]
        if len(sent_text) > 5:
            sentences.append((sent_start, sent_end, sent_text))
    return sentences


def tokenize_sentence(sent_text):
    """
    2. ตัดคำในประโยค (Word Tokenization) พร้อมเก็บพิกัดของแต่ละคำเทียบกับประโยค
    """
    tokens = []
    spans = []
    for match in re.finditer(r'\w+|[^\w\s]', sent_text):
        tokens.append(match.group())
        spans.append(match.span())
    return tokens, spans


def convert_to_bio(sent_start, sent_end, sent_text, all_spans):
    """
    3. นำ Spans ทั้งหมดมาทาบเทียบเข้ากับคำในประโยค เพื่อแปะป้าย B- / I- / O
    """
    tokens, tok_spans = tokenize_sentence(sent_text)
    tags = ["O"] * len(tokens)

    # กรองเฉพาะ Spans ที่ตกอยู่ในประโยคนี้
    for sp in all_spans:
        if sp["start"] >= sent_start and sp["end"] <= sent_end:
            # คำนวณพิกัดสัมพัทธ์ในประโยค
            rel_start = sp["start"] - sent_start
            rel_end = sp["end"] - sent_start
            label = sp["label"]

            first = True
            for idx, (t_st, t_en) in enumerate(tok_spans):
                # ตรวจสอบว่าคำนี้อยู่ในช่วงพิกัดของ Span หรือไม่
                if t_st >= rel_start and t_en <= rel_end:
                    if tags[idx] == "O" or (sp.get("category") == "trigger" and not tags[idx].startswith("B-Trigger")):
                        if first:
                            tags[idx] = f"B-{label}"  # จุดเริ่มต้นคำ
                            first = False
                        else:
                            tags[idx] = f"I-{label}"  # คำที่ต่อเนื่องกัน

    return tokens, tags


def main():
    print("=" * 75)
    print("🌟 ขั้นตอนถัดไป: Sentence Splitting & BIO Tagging Conversion")
    print("=" * 75)

    # 1. โหลดข้อมูลจริงจากไฟล์ 10001.json
    local_path = Path(__file__).resolve().parent.parent / "data" / "annotation" / "10001.json"
    nlp2_path = Path("d:/Kasetsart/NLP/NLP2_pp/CASIE_repo/data/annotation/10001.json")
    file_path = local_path if local_path.exists() else nlp2_path
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from merge_spans_demo import extract_and_merge_spans
    content, merged_spans = extract_and_merge_spans(data)

    # 2. หั่นเนื้อข่าวเป็นประโยค
    sentences = split_sentences_with_offsets(content)
    print(f"บทความนี้ถูกแบ่งออกเป็น {len(sentences)} ประโยค\n")

    # 3. ลองยกตัวอย่างประโยคแรกสุดมาดูการแปะป้าย BIO
    sent_start, sent_end, sent_text = sentences[0]
    tokens, tags = convert_to_bio(sent_start, sent_end, sent_text, merged_spans)

    print("📝 ตัวอย่างประโยคที่ 1:")
    print(f"\"{sent_text}\"\n")
    print("พิกัดประโยคในบทความเดิม: [", sent_start, ":", sent_end, "]")
    print("-" * 75)
    print(f"{'ลำดับ':<6} {'คำ (Token)':<25} {'ป้ายกำกับ (BIO Tag)':<30}")
    print("-" * 75)

    for i, (tok, tag) in enumerate(zip(tokens, tags), 1):
        # ไฮไลต์ถ้าไม่ใช่ 'O'
        highlight = f"<-- 🎯 {tag}" if tag != "O" else ""
        print(f"{i:<6} {tok:<25} {tag:<30} {highlight}")

    print("=" * 75)
    print("✅ ตอนนี้ข้อมูลของเราพร้อมแล้วในรูปแบบ Token-Level BIO Tagging!")
    print("   ซึ่งเป็นมาตรฐานที่โมเดล Transformer (BERT / SecBERT) นำไปเทรนได้ทันที")


if __name__ == "__main__":
    main()
