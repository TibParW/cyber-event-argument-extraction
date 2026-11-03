"""
step1_data_preparation.py
สคริปต์ประมวลผลข้อมูล 1,000 ข่าวจริง จาก CASIE Dataset
พร้อมสกัดฟีเจอร์ POS Tag และ General NER Tag ตามที่อาจารย์แนะนำ

กระบวนการ:
  1. โหลดรายชื่อไฟล์ตาม Official Split ของอาจารย์โดยตรง:
     - test__testlista.txt -> Test Set (100 ข่าว)
     - train+validate__annlistver3k_traina.txt -> แบ่ง Train 800 ข่าว (80%) และ Val 100 ข่าว (10%)
  2. รวมเฉลย Nugget (Trigger) และ Argument (Roles) เข้าด้วยกัน (Merge Spans)
  3. หั่นประโยคและใช้ spaCy สกัดพร้อมกัน:
     - tokens (คำ)
     - pos_tags (Universal POS Tag เช่น NOUN, VERB, PROPN, NUM)
     - ner_gen_tags (General NER Tag เช่น ORG, PERSON, MONEY, DATE)
     - ner_tags (ป้ายกำกับเฉลย CASIE BIO 63 คลาส)
  4. กรอง Class Imbalance:
     - Train/Val: เก็บประโยคที่มี Entity 100% และสุ่มประโยคว่าง 40%
     - Test: เก็บ 100% ทุกประโยคเพื่อสะท้อนโลกความเป็นจริง
  5. บันทึกผลลัพธ์เป็น train.json, val.json, test.json และตาราง Mapping ทุกฟีเจอร์ลง dataset/
"""
import sys
import json
import re
import random
from pathlib import Path
from collections import Counter
import spacy

# ป้องกันปัญหา UnicodeEncodeError บน Windows คอนโซล เมื่อมีอิโมจิ
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent
local_annotation = BASE_DIR / "data" / "annotation"
if local_annotation.exists() and len(list(local_annotation.glob("*.json"))) > 0:
    ANNOTATION_DIR = local_annotation
else:
    ANNOTATION_DIR = Path("d:/Kasetsart/NLP/NLP2_pp/CASIE_repo/data/annotation")

OUTPUT_DIR = BASE_DIR / "dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def split_sentences_with_offsets(text):
    """หั่นเนื้อข่าวออกเป็นประโยค พร้อมจำพิกัดตัวอักษร [start : end]"""
    sentences = []
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
        if len(sent_text) > 4:
            sentences.append((sent_start, sent_end, sent_text))
    return sentences


def parse_single_file(filepath, nlp):
    """
    อ่านไฟล์ JSON ข่าว 1 ไฟล์:
    - ดึง Event Nuggets และ Arguments (Roles)
    - ใช้ spaCy ตัดคำพร้อมดึง POS Tag และ General NER Tag
    - ทาบพิกัดแปะป้าย BIO ให้อัตโนมัติ
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    content = data.get('content', '')
    all_spans = []

    # 1. ดึง Nugget (Trigger) และ Argument (Roles)
    hoppers = data.get('cyberevent', {}).get('hopper', [])
    for hopper in hoppers:
        for ev in hopper.get('events', []):
            nug = ev.get('nugget', {})
            subtype = ev.get('subtype', 'Unknown')
            if 'startOffset' in nug and 'endOffset' in nug:
                all_spans.append({
                    'start': nug['startOffset'],
                    'end': nug['endOffset'],
                    'label': f"Trigger-{subtype}",
                    'category': 'trigger'
                })

            for arg in ev.get('argument', []):
                role = arg.get('role', {}).get('type', 'Unknown')
                if 'startOffset' in arg and 'endOffset' in arg:
                    all_spans.append({
                        'start': arg['startOffset'],
                        'end': arg['endOffset'],
                        'label': role,
                        'category': 'argument'
                    })

    # จัดเรียง Spans ตามตำแหน่งเริ่มต้น
    all_spans.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))

    # 2. หั่นประโยคและใช้ spaCy สกัดข้อมูล
    sentences = split_sentences_with_offsets(content)
    doc_samples = []

    for sent_start, sent_end, sent_text in sentences:
        doc = nlp(sent_text)

        tokens = []
        pos_tags = []
        ner_gen_tags = []
        tok_spans = []

        for t in doc:
            if t.pos_ == 'SPACE' or not t.text.strip():
                continue
            tokens.append(t.text)
            pos_tags.append(t.pos_)
            ner_gen_tags.append(t.ent_type_ if t.ent_type_ else 'O')
            tok_spans.append((t.idx, t.idx + len(t.text)))

        if not tokens:
            continue

        # กำหนดป้ายกำกับเริ่มต้นเป็น 'O'
        tags = ['O'] * len(tokens)

        # นำ Spans มาทาบเทียบพิกัดกับคำในประโยค
        for sp in all_spans:
            if sp['start'] >= sent_start and sp['end'] <= sent_end:
                rel_start = sp['start'] - sent_start
                rel_end = sp['end'] - sent_start
                label = sp['label']

                first = True
                for idx, (t_st, t_en) in enumerate(tok_spans):
                    if t_st >= rel_start and t_en <= rel_end:
                        if tags[idx] == 'O' or (sp['category'] == 'trigger' and not tags[idx].startswith('B-Trigger')):
                            if first:
                                tags[idx] = f"B-{label}"
                                first = False
                            else:
                                tags[idx] = f"I-{label}"

        doc_samples.append({
            'tokens': tokens,
            'pos_tags': pos_tags,
            'ner_gen_tags': ner_gen_tags,
            'ner_tags': tags,
            'sentence': sent_text
        })

    return doc_samples


def main():
    print("=" * 80)
    print("🚀 [Step 1] เริ่มต้นการเตรียมชุดข้อมูลใหม่ (CASIE + POS Tag + General NER)")
    print("   อิงตาม Official Split ของอาจารย์โดยตรง (test__testlista.txt)")
    print("=" * 80)

    # โหลด spaCy pipeline (ปิด parser เพื่อความเร็วสูงสุด)
    print("\n📦 กำลังโหลดโมเดลภาษาอังกฤษ spaCy (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    # ค้นหาไฟล์ Split ทั้งใน data/splits/ และ root ของ LPLLL
    test_split_candidate = BASE_DIR / "data" / "splits" / "test__testlista.txt"
    test_split_path = test_split_candidate if test_split_candidate.exists() else BASE_DIR / "test__testlista.txt"

    train_val_split_candidate = BASE_DIR / "data" / "splits" / "train+validate__annlistver3k_traina.txt"
    train_val_split_path = train_val_split_candidate if train_val_split_candidate.exists() else BASE_DIR / "train+validate__annlistver3k_traina.txt"

    if not test_split_path.exists() or not train_val_split_path.exists():
        raise FileNotFoundError("❌ ไม่พบไฟล์ split ของอาจารย์ในโฟลเดอร์ LPLLL")

    with open(test_split_path, 'r', encoding='utf-8') as f:
        test_ids = [line.strip() for line in f if line.strip()]

    with open(train_val_split_path, 'r', encoding='utf-8') as f:
        tv_ids = [line.strip() for line in f if line.strip()]

    test_files = [ANNOTATION_DIR / f"{fid}.json" for fid in test_ids if (ANNOTATION_DIR / f"{fid}.json").exists()]
    valid_tv_files = [ANNOTATION_DIR / f"{fid}.json" for fid in tv_ids if (ANNOTATION_DIR / f"{fid}.json").exists()]

    # แบ่ง Train 800 ไฟล์ (80%) และ Val 100 ไฟล์ (10%) โดยใช้ Seed คงที่
    rng = random.Random(42)
    shuffled_tv = valid_tv_files.copy()
    rng.shuffle(shuffled_tv)

    train_files = shuffled_tv[:800]
    val_files = shuffled_tv[800:]

    print(f"\n📊 สัดส่วนการแบ่งข้อมูลตาม Benchmark ของอาจารย์ (Official Split):")
    print(f"   - ชุด Train:      {len(train_files)} ข่าว (80%)")
    print(f"   - ชุด Validation: {len(val_files)} ข่าว (10%)")
    print(f"   - ชุด Test:       {len(test_files)} ข่าว (10%) [test__testlista.txt]")
    print(f"   - รวมทั้งหมด:     {len(train_files) + len(val_files) + len(test_files)} ข่าวครบถ้วน\n")

    def process_file_list(files, split_name, keep_ratio=0.4):
        print(f"⏳ กำลังประมวลผลชุด {split_name} ({len(files)} ข่าว)...")
        samples = []
        rand_inst = random.Random(42)

        for idx, fpath in enumerate(files, 1):
            doc_samples = parse_single_file(fpath, nlp)
            for s in doc_samples:
                has_entity = any(t != 'O' for t in s['ner_tags'])
                keep_sample = (
                    has_entity 
                    or keep_ratio is None 
                    or keep_ratio >= 1.0 
                    or rand_inst.random() < keep_ratio
                )
                if keep_sample:
                    samples.append(s)

            if idx % 200 == 0 or idx == len(files):
                print(f"   - ทำเสร็จแล้ว {idx}/{len(files)} ข่าว...")

        print(f"   -> ได้ทั้งหมด {len(samples):,} ประโยค\n")
        return samples

    # รันประมวลผลทั้ง 3 ชุด
    train_data = process_file_list(train_files, "Train", keep_ratio=0.4)
    val_data = process_file_list(val_files, "Validation", keep_ratio=0.4)
    test_data = process_file_list(test_files, "Test", keep_ratio=None)

    # --------------------------------------------------------------------------
    # สร้างตาราง Mapping สำหรับทั้ง 3 Features
    # --------------------------------------------------------------------------
    print("🏷️  กำลังสร้างพจนานุกรม Mapping สำหรับทุก Features...")

    # 1. CASIE BIO Labels
    all_casie_tags = set(['O'])
    for d in [train_data, val_data, test_data]:
        for s in d:
            all_casie_tags.update(s['ner_tags'])

    sorted_entities = sorted(list(set(t.split('-', 1)[1] for t in all_casie_tags if t != 'O')))
    casie_tag_list = ['O']
    for ent in sorted_entities:
        casie_tag_list.append(f"B-{ent}")
        casie_tag_list.append(f"I-{ent}")
    casie_label2id = {tag: i for i, tag in enumerate(casie_tag_list)}
    casie_id2label = {i: tag for i, tag in enumerate(casie_tag_list)}

    # 2. POS Tags
    all_pos = set(['PAD', 'UNK', 'O'])
    for d in [train_data, val_data, test_data]:
        for s in d:
            all_pos.update(s['pos_tags'])
    pos_list = sorted(list(all_pos))
    pos2id = {p: i for i, p in enumerate(pos_list)}
    id2pos = {i: p for i, p in enumerate(pos_list)}

    # 3. General NER Tags
    all_ner_gen = set(['PAD', 'UNK', 'O'])
    for d in [train_data, val_data, test_data]:
        for s in d:
            all_ner_gen.update(s['ner_gen_tags'])
    ner_gen_list = sorted(list(all_ner_gen))
    ner_gen2id = {n: i for i, n in enumerate(ner_gen_list)}
    id2ner_gen = {i: n for i, n in enumerate(ner_gen_list)}

    print(f"   - CASIE BIO Classes: {len(casie_tag_list)} คลาส")
    print(f"   - POS Tag Types:     {len(pos_list)} ประเภท: {pos_list}")
    print(f"   - General NER Types: {len(ner_gen_list)} ประเภท: {ner_gen_list}")

    # บันทึกเป็นไฟล์ JSON
    def save_split_json(data, filename):
        formatted = []
        for it in data:
            formatted.append({
                'tokens': it['tokens'],
                'pos_tags': [pos2id.get(p, pos2id['UNK']) for p in it['pos_tags']],
                'pos_tags_str': it['pos_tags'],
                'ner_gen_tags': [ner_gen2id.get(n, ner_gen2id['UNK']) for n in it['ner_gen_tags']],
                'ner_gen_tags_str': it['ner_gen_tags'],
                'ner_tags': [casie_label2id[t] for t in it['ner_tags']],
                'ner_tags_str': it['ner_tags'],
                'sentence': it['sentence']
            })
        out_path = OUTPUT_DIR / filename
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, ensure_ascii=False)
        print(f"💾 บันทึกไฟล์: {out_path} ({len(formatted):,} ประโยค)")

    print("\n" + "=" * 80)
    print("📁 กำลังบันทึกชุดข้อมูลลงโฟลเดอร์ dataset/:")
    save_split_json(train_data, "train.json")
    save_split_json(val_data, "val.json")
    save_split_json(test_data, "test.json")

    # บันทึกไฟล์ Mapping
    with open(OUTPUT_DIR / "label_mapping.json", 'w', encoding='utf-8') as f:
        json.dump({'label2id': casie_label2id, 'id2label': casie_id2label, 'tag_list': casie_tag_list}, f, indent=2)

    with open(OUTPUT_DIR / "pos_mapping.json", 'w', encoding='utf-8') as f:
        json.dump({'pos2id': pos2id, 'id2pos': id2pos, 'pos_list': pos_list}, f, indent=2)

    with open(OUTPUT_DIR / "ner_gen_mapping.json", 'w', encoding='utf-8') as f:
        json.dump({'ner_gen2id': ner_gen2id, 'id2ner_gen': id2ner_gen, 'ner_gen_list': ner_gen_list}, f, indent=2)

    with open(OUTPUT_DIR / "feature_mappings.json", 'w', encoding='utf-8') as f:
        json.dump({
            'casie': {'label2id': casie_label2id, 'id2label': casie_id2label, 'tag_list': casie_tag_list},
            'pos': {'pos2id': pos2id, 'id2pos': id2pos, 'pos_list': pos_list},
            'ner_gen': {'ner_gen2id': ner_gen2id, 'id2ner_gen': id2ner_gen, 'ner_gen_list': ner_gen_list}
        }, f, indent=2)

    print(f"💾 บันทึกตาราง Mapping ทั้งหมดลง {OUTPUT_DIR} เรียบร้อยแล้ว!")
    print("=" * 80)
    print("🎉 เสร็จสิ้น Step 1: ชุดข้อมูลใหม่พร้อมสำหรับสถาปัตยกรรมโมเดลและฟีเจอร์เสริมทุกรูปแบบ!")
    print("=" * 80)


if __name__ == "__main__":
    main()
