"""
evaluation_demo.py
สาธิตขั้นตอนการตรวจข้อสอบ คิดคะแนน และคำนวณเปอร์เซ็นต์ (Precision, Recall, F1)
ตำแหน่งไฟล์: D:\\Kasetsart\\NLP\\LPLLL\\evaluation_demo.py
"""

import sys
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def demo_evaluation_calculation():
    print("=" * 80)
    print("🔍 ขั้นตอนการตรวจข้อสอบและคิดคะแนน (Evaluation Process)")
    print("=" * 80)

    # 1. ข้อมูลที่เราส่งไปสอบ: คือประโยคจาก test.json
    sentence = ["LockBit", "demanded", "$2M", "ransom", "from", "Crosshouse", "Hospital", "."]

    # 2. กระดาษเฉลยของอาจารย์ (Ground Truth จาก test.json)
    true_labels = [
        "B-Attacker",          # LockBit
        "O",                   # demanded
        "B-Money",             # $2M
        "B-Trigger-Ransom",    # ransom
        "O",                   # from
        "B-Victim",            # Crosshouse
        "I-Victim",            # Hospital
        "O"                    # .
    ]

    # 3. สิ่งที่โมเดลเดาออกมา (Model Predictions)
    # สมมุติโมเดลเดาถูกเกือบหมด แต่ทาย $2M พลาด และทาย Crosshouse ผิด
    pred_labels = [
        "B-Attacker",          # LockBit  -> ทายถูก!
        "O",                   # demanded -> ทายถูก!
        "O",                   # $2M      -> โมเดลมองข้าม (ทายหลุด)
        "B-Trigger-Ransom",    # ransom   -> ทายถูก!
        "O",                   # from     -> ทายถูก!
        "B-Victim",            # Crosshouse -> ทายถูก
        "I-Victim",            # Hospital   -> ทายถูก
        "O"                    # .        -> ทายถูก!
    ]

    print("📄 ประโยคที่ใช้ทดสอบ (จาก test.json):")
    print("   ", " ".join(sentence))
    print("\n" + "-" * 80)
    print(f"{'คำ (Token)':<15} {'เฉลยจริง (Ground Truth)':<25} {'โมเดลทาย (Prediction)':<25} {'ผลการตรวจ'}")
    print("-" * 80)

    for word, truth, pred in zip(sentence, true_labels, pred_labels):
        if truth == pred:
            if truth != "O":
                status = "✅ ถูกต้อง (+1 คะแนน Entity)"
            else:
                status = "⚪ คำทั่วไป (ถูก)"
        else:
            status = "❌ ทายผิด/หลุด"

        print(f"{word:<15} {truth:<25} {pred:<25} {status}")

    print("=" * 80)
    print("🧮 วิธีคิดคะแนนออกมาเป็นเปอร์เซ็นต์ (%) โดยไลบรารี seqeval:")
    print("=" * 80)

    # ในระดับ Entity:
    # เฉลยจริงมี 4 Entities: [LockBit (Attacker), $2M (Money), ransom (Trigger), Crosshouse Hospital (Victim)]
    # สิ่งที่โมเดลทายมา 3 Entities: [LockBit (Attacker), ransom (Trigger), Crosshouse Hospital (Victim)]

    total_truth_entities = 4     # เฉลยจริงมี 4 ก้อน
    total_pred_entities = 3      # โมเดลชี้เป้ามา 3 ก้อน
    correct_entities = 3         # ทายถูกตรงกันเป๊ะ 3 ก้อน (LockBit, ransom, Crosshouse Hospital)

    # 1. Precision = (ทายถูก / โมเดลชี้เป้าทั้งหมด) * 100
    precision = (correct_entities / total_pred_entities) * 100

    # 2. Recall = (ทายถูก / เฉลยจริงทั้งหมด) * 100
    recall = (correct_entities / total_truth_entities) * 100

    # 3. F1-Score = 2 * (P * R) / (P + R)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"1. Precision (ความแม่นยำ):")
    print(f"   สูตร: (ทายถูก {correct_entities} / โมเดลชี้มาทั้งหมด {total_pred_entities}) × 100")
    print(f"   คะแนน = {precision:.2f}%  (หมายถึง: ในบรรดาที่โมเดลบอกว่าใช่ มันทายแม่น 100% ไม่ชี้มั่วเลย)\n")

    print(f"2. Recall (ความครอบคลุม):")
    print(f"   สูตร: (ทายถูก {correct_entities} / เฉลยจริงที่มีทั้งหมด {total_truth_entities}) × 100")
    print(f"   คะแนน = {recall:.2f}%  (หมายถึง: เฉลยมี 4 ตัว จับมาได้ 3 ตัว หลุดรอดไป 1 ตัว คือ $2M)\n")

    print(f"3. F1-Score (คะแนนรวมความเก่งจริง):")
    print(f"   สูตร: 2 × (P × R) / (P + R)")
    print(f"   คะแนน = {f1:.2f}%\n")

    print("=" * 80)
    print("💻 จุดที่เราดูผลคะแนนจริงในโปรแกรม:")
    print("   ตอนรัน: trainer.evaluate(test_dataset)")
    print("   ไลบรารีจะคำนวณแบบนี้กับประโยคทั้งหมด 1,081 ประโยคใน test.json")
    print("   แล้วพิมพ์สรุปตารางตัวเลข % ทั้งหมดออกมาบนหน้าจอ Terminal ทันที!")

if __name__ == "__main__":
    demo_evaluation_calculation()
