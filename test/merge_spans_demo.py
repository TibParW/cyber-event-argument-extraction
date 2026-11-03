import sys
import json
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def extract_and_merge_spans(json_data):
    """
    ฟังก์ชันสำหรับดึง nugget และ argument ออกมาจาก CASIE JSON
    แล้วนำมารวมเป็น Spans รายการเดียวกัน พร้อมเรียงลำดับตามพิกัดตัวอักษร
    """
    content = json_data.get("content", "")
    spans = []

    hoppers = json_data.get("cyberevent", {}).get("hopper", [])
    
    # วนลูปผ่านทุก hopper และทุก event ในข่าว
    for hopper in hoppers:
        for ev in hopper.get("events", []):
            
            # -------------------------------------------------------------
            # 1. ดึงข้อมูล Event Nugget (Trigger - คำบ่งบอกการโจมตี)
            # -------------------------------------------------------------
            nug = ev.get("nugget", {})
            subtype = ev.get("subtype", "Unknown")
            
            if "startOffset" in nug and "endOffset" in nug:
                start = nug["startOffset"]
                end = nug["endOffset"]
                spans.append({
                    "start": start,
                    "end": end,
                    "text": content[start:end],          # ข้อความจริงตามพิกัด
                    "label": f"Trigger-{subtype}",       # ป้ายกำกับ (เช่น Trigger-Databreach)
                    "category": "trigger"                # ชนิด: trigger
                })

            # -------------------------------------------------------------
            # 2. ดึงข้อมูล Event Arguments (Roles - บทบาทของผู้เกี่ยวข้อง)
            # -------------------------------------------------------------
            for arg in ev.get("argument", []):
                role = arg.get("role", {}).get("type", "Unknown")
                if "startOffset" in arg and "endOffset" in arg:
                    start = arg["startOffset"]
                    end = arg["endOffset"]
                    spans.append({
                        "start": start,
                        "end": end,
                        "text": content[start:end],      # ข้อความจริงตามพิกัด
                        "label": role,                   # ป้ายกำกับ (เช่น Attacker, Victim)
                        "category": "argument"           # ชนิด: argument
                    })

    # -----------------------------------------------------------------
    # 3. จัดเรียงลำดับ Spans (Sorting & Conflict Resolution)
    #    - เรียงตามจุดเริ่มต้น (start) ก่อน
    #    - หากเริ่มพร้อมกัน: เอาอันที่ยาวกว่าขึ้นก่อน (-(end - start))
    # -----------------------------------------------------------------
    spans.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    # -----------------------------------------------------------------
    # 4. ลบ Spans ที่ซ้ำกัน 100% (Deduplication)
    # -----------------------------------------------------------------
    unique_spans = []
    seen = set()
    for sp in spans:
        key = (sp["start"], sp["end"], sp["label"])
        if key not in seen:
            seen.add(key)
            unique_spans.append(sp)

    return content, unique_spans


def main():
    print("=" * 70)
    print("🌟 ตัวอย่างการทำงานของ Merge Spans จาก CASIE Dataset")
    print("=" * 70)

    # ทดสอบกับไฟล์จริงของอาจารย์ (ไฟล์ 10001.json)
    local_path = Path(__file__).resolve().parent.parent / "data" / "annotation" / "10001.json"
    nlp2_path = Path("d:/Kasetsart/NLP/NLP2_pp/CASIE_repo/data/annotation/10001.json")
    real_file_path = local_path if local_path.exists() else nlp2_path

    if real_file_path.exists():
        print(f"กำลังโหลดข้อมูลจริงจาก: {real_file_path.name}\n")
        with open(real_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # ข้อมูลตัวอย่างสมมติ (หากไม่พบไฟล์)
        print("ใช้ข้อมูล Mock สำหรับทดสอบ...")
        data = {
            "content": "On Friday, LockBit demanded $2 million ransom from Crosshouse Hospital.",
            "cyberevent": {
                "hopper": [
                    {
                        "events": [
                            {
                                "subtype": "Ransom",
                                "nugget": {"startOffset": 38, "endOffset": 44, "text": "ransom"},
                                "argument": [
                                    {"startOffset": 11, "endOffset": 18, "role": {"type": "Attacker"}},
                                    {"startOffset": 28, "endOffset": 37, "role": {"type": "Money"}},
                                    {"startOffset": 50, "endOffset": 69, "role": {"type": "Victim"}}
                                ]
                            }
                        ]
                    }
                ]
            }
        }

    content, merged_spans = extract_and_merge_spans(data)

    print("📄 ตัวอย่างเนื้อข่าว (content บางส่วน):")
    print(f"\"{content[:160]}...\"\n")

    print(f"🎯 รวม Spans ทั้งหมดได้: {len(merged_spans)} จุด")
    print("-" * 70)
    print(f"{'ลำดับ':<5} {'พิกัด [Start : End]':<22} {'ข้อความจริง (Text)':<25} {'ป้ายกำกับ (Label)'}")
    print("-" * 70)

    for i, sp in enumerate(merged_spans[:10], 1):
        coord = f"[{sp['start']:>4} : {sp['end']:<4}]"
        txt = sp["text"]
        if len(txt) > 22:
            txt = txt[:19] + "..."
        print(f"{i:<5} {coord:<22} {txt:<25} {sp['label']}")

    if len(merged_spans) > 10:
        print(f"... และยังมีอีก {len(merged_spans) - 10} รายการ")

    print("=" * 70)
    print("✅ สรุป: ตอนนี้เราได้ 'เส้นเวลา (Timeline)' ของทุก Entity ที่เรียงตามลำดับแล้ว")
    print("   พร้อมที่จะนำไปทาบกับแต่ละประโยค เพื่อแปลงเป็น BIO Tagging ในขั้นตอนถัดไป!")


if __name__ == "__main__":
    main()
