"""
subword_alignment_demo.py
สาธิตปัญหา Subword Tokenization Mismatch และวิธีแก้ด้วย Subword Alignment (-100)
ตำแหน่งไฟล์: D:\\Kasetsart\\NLP\\LPLLL\\subword_alignment_demo.py
"""

import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from transformers import AutoTokenizer

def demo_subword_alignment():
    print("=" * 80)
    print("🌟 ขั้นตอนที่ 3: Subword Tokenization & Label Alignment")
    print("=" * 80)

    # 1. โหลด Tokenizer ของ SecBERT (หรือ bert-base-cased)
    model_name = "jackaduma/SecBERT"
    print(f"กำลังโหลด Tokenizer: {model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        # หากต่อเน็ตช้าหรือออฟไลน์ ให้ fallback เป็น bert-base-cased ชั่วคราว
        print("Fallback เป็น bert-base-cased...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # 2. ข้อมูลตั้งต้นระดับคำ (Word Level) จากขั้นตอนที่ 2
    words = ["Crosshouse", "Hospital", "accessed", "by", "LockBit", "."]
    ner_tags = ["B-Victim", "I-Victim", "B-Trigger-Databreach", "O", "B-Attacker", "O"]

    print("\n📝 ข้อมูลตั้งต้นระดับคำ (ก่อนเข้า Tokenizer):")
    print("  Words:   ", words, f"(จำนวนคำ: {len(words)})")
    print("  Labels:  ", ner_tags, f"(จำนวนป้าย: {len(ner_tags)})")

    # 3. นำคำเข้า Tokenizer (is_split_into_words=True)
    tokenized_inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=False
    )

    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
    word_ids = tokenized_inputs.word_ids()

    print("\n⚠️ ปัญหาเกิดขึ้นทันทีหลังเข้า Tokenizer:")
    print(f"  Tokenizer แตกคำย่อยและเพิ่ม Special Tokens กลายเป็น: {len(tokens)} โทเคน!")
    print("  Tokens:  ", tokens)
    print("  (จะเห็นว่า 'Crosshouse' โดนหั่นเป็น 'Cross' + '##house' และ 'LockBit' โดนหั่นเป็น 'Lock' + '##Bit')")
    print(f"  ❌ ตอนนี้จำนวน Tokens ({len(tokens)}) != จำนวน Labels ({len(ner_tags)}) ส่งเข้าโมเดลตรง ๆ ไม่ได้!")

    # 4. วิธีแก้: ทำ Subword Alignment ด้วย word_ids() และใส่ -100
    aligned_labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # เป็น Special Token เช่น [CLS], [SEP]
            aligned_labels.append("-100 (Ignore)")
        elif word_idx != previous_word_idx:
            # เป็น Subword ชิ้นแรกของคำ -> รับป้ายจริงไปเลย
            aligned_labels.append(ner_tags[word_idx])
        else:
            # เป็น Subword ชิ้นต่อมา (เช่น ##house, ##Bit) -> แปะ -100
            aligned_labels.append("-100 (Ignore)")
        previous_word_idx = word_idx

    # 5. แสดงผลลัพธ์การจับคู่เปรียบเทียบ
    print("\n" + "=" * 80)
    print("✅ ผลลัพธ์หลังทำ Subword Alignment (พร้อมส่งเข้า PyTorch Trainer):")
    print("=" * 80)
    print(f"{'ลำดับ':<6} {'Token ย่อย':<18} {'word_id':<12} {'Aligned Label':<22} {'เหตุผล'}")
    print("-" * 80)

    for idx, (tok, w_id, lbl) in enumerate(zip(tokens, word_ids, aligned_labels), 1):
        w_id_str = str(w_id) if w_id is not None else "None"
        reason = ""
        if lbl == "-100 (Ignore)":
            if w_id is None:
                reason = "Special token -> ไม่คิด Loss"
            else:
                reason = "เศษคำย่อย -> ไม่คิด Loss ซ้ำ"
        else:
            reason = "คำแรก -> ใช้สอนโมเดลคำนวณ Loss"

        print(f"{idx:<6} {tok:<18} {w_id_str:<12} {lbl:<22} {reason}")

    print("=" * 80)
    print("💡 ทำไมต้องเป็น -100?")
    print("   เพราะใน PyTorch มีค่า default คือ ignore_index = -100")
    print("   CrossEntropyLoss จะข้ามโทเคนเหล่านี้ไปเลย ไม่นำมาคิด Loss และไม่ตัดคะแนน!")

if __name__ == "__main__":
    demo_subword_alignment()
