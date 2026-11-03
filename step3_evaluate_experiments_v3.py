import sys
import json
import argparse
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, Trainer

try:
    from step2_train_experiments_v3 import (
        BertWithExtraFeaturesForTokenClassification,
        ExtraFeatureDataCollator,
        tokenize_and_align_labels,
        extract_bio_spans,
        build_valid_transition_matrix,
        viterbi_bio_decode
    )
except ImportError:
    from step2_train_experiments import (
        BertWithExtraFeaturesForTokenClassification,
        ExtraFeatureDataCollator,
        tokenize_and_align_labels,
        extract_bio_spans,
        build_valid_transition_matrix,
        viterbi_bio_decode
    )

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"


# ==============================================================================
# Dual Metrics Builder (Strict Exact Match + NIST TAC Overlap Match)
# ==============================================================================
def compute_span_metrics(true_predictions, true_labels):
    """
    คำนวณทั้ง Strict F1 และ Relaxed Overlap F1 จากคู่คำตอบที่ถอดรหัสแล้ว
    """
    n_strict_tp = 0
    n_relaxed_tp = 0
    total_preds = 0
    total_labs = 0
    n_correct_tok = 0
    n_total_tok = 0

    for p_seq, l_seq in zip(true_predictions, true_labels):
        p_spans = extract_bio_spans(p_seq)
        l_spans = extract_bio_spans(l_seq)
        total_preds += len(p_spans)
        total_labs += len(l_spans)

        # 1. Strict Set Match (ต้องตรง 100% ทั้ง Type, Start, End)
        n_strict_tp += len(set(p_spans).intersection(set(l_spans)))

        # 2. Relaxed Overlap Match (เกณฑ์ NIST TAC / เปเปอร์อาจารย์: Type ตรง และขอบเขตเหลื่อมกัน)
        matched_l = set()
        for p_type, p_st, p_en in p_spans:
            for l_idx, (l_type, l_st, l_en) in enumerate(l_spans):
                if l_idx not in matched_l and p_type == l_type:
                    if max(p_st, l_st) <= min(p_en, l_en):
                        n_relaxed_tp += 1
                        matched_l.add(l_idx)
                        break

        for p_t, l_t in zip(p_seq, l_seq):
            if p_t == l_t:
                n_correct_tok += 1
            n_total_tok += 1

    # Strict Metrics
    s_p = n_strict_tp / total_preds if total_preds > 0 else 0.0
    s_r = n_strict_tp / total_labs if total_labs > 0 else 0.0
    s_f1 = (2 * s_p * s_r / (s_p + s_r)) if (s_p + s_r) > 0 else 0.0

    # Relaxed Overlap Metrics (AAAI 2020 Benchmark)
    r_p = n_relaxed_tp / total_preds if total_preds > 0 else 0.0
    r_r = n_relaxed_tp / total_labs if total_labs > 0 else 0.0
    r_f1 = (2 * r_p * r_r / (r_p + r_r)) if (r_p + r_r) > 0 else 0.0

    acc = n_correct_tok / n_total_tok if n_total_tok > 0 else 0.0

    return {
        "strict_precision": round(s_p * 100, 2),
        "strict_recall": round(s_r * 100, 2),
        "strict_f1": round(s_f1 * 100, 2),
        "relaxed_precision": round(r_p * 100, 2),
        "relaxed_recall": round(r_r * 100, 2),
        "relaxed_f1": round(r_f1 * 100, 2),
        "accuracy": round(acc * 100, 2),
    }


def load_custom_model(model_dir, cfg, device):
    """โหลดโครงสร้างและน้ำหนักโมเดลกลับขึ้นมาทดสอบ"""
    model = BertWithExtraFeaturesForTokenClassification(
        model_name=cfg["model_name"],
        num_labels=cfg["num_labels"],
        num_pos=cfg.get("num_pos", 0),
        num_ner_gen=cfg.get("num_ner_gen", 0),
        o_weight=cfg.get("o_weight", 0.35)
    )

    weights_path = Path(model_dir) / "custom_model_weights.bin"
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
    elif (Path(model_dir) / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(Path(model_dir) / "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
    elif (Path(model_dir) / "pytorch_model.bin").exists():
        model.load_state_dict(torch.load(Path(model_dir) / "pytorch_model.bin", map_location=device), strict=False)

    model.to(device)
    model.eval()
    return model


# ==============================================================================
# ฟังก์ชันประเมินผลโมเดลเดี่ยว (Single Model Evaluation)
# ==============================================================================
def evaluate_single_model(model_dir, test_raw, tag_list, label2id, device, use_viterbi=True):
    model_dir = Path(model_dir)
    config_file = model_dir / "experiment_config.json"
    if not config_file.exists():
        print(f"⚠️ ข้าม {model_dir.name} เนื่องจากไม่พบ experiment_config.json")
        return None, None

    with open(config_file, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = load_custom_model(model_dir, cfg, device)

    test_hf = Dataset.from_dict({
        'tokens': [it['tokens'] for it in test_raw],
        'ner_tags': [it['ner_tags'] for it in test_raw],
        'pos_tags': [it['pos_tags'] for it in test_raw],
        'ner_gen_tags': [it['ner_gen_tags'] for it in test_raw]
    })

    test_tokenized = test_hf.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_seq_length=cfg.get("max_seq_length", 256)),
        batched=True,
        remove_columns=test_hf.column_names
    )

    data_collator = ExtraFeatureDataCollator(tokenizer)
    trainer = Trainer(model=model, data_collator=data_collator)

    preds_output = trainer.predict(test_tokenized)
    logits = preds_output.predictions  # [N, SeqLen, NumLabels]
    labels = preds_output.label_ids     # [N, SeqLen]

    valid_trans = build_valid_transition_matrix(label2id) if use_viterbi else None

    true_predictions = []
    true_labels = []

    for sent_idx in range(len(logits)):
        sent_logits = logits[sent_idx]
        sent_labs = labels[sent_idx]

        valid_mask = (sent_labs != -100)
        if not np.any(valid_mask):
            continue

        valid_logits = sent_logits[valid_mask]
        valid_truth = sent_labs[valid_mask]

        if use_viterbi:
            pred_path = viterbi_bio_decode(valid_logits, valid_trans, label2id)
        else:
            pred_path = np.argmax(valid_logits, axis=-1)

        true_predictions.append([tag_list[idx] for idx in pred_path])
        true_labels.append([tag_list[idx] for idx in valid_truth])

    metrics = compute_span_metrics(true_predictions, true_labels)

    result_row = {
        "Experiment": model_dir.name,
        "Model": cfg["model_name"].split("/")[-1],
        "Features": cfg["feature_mode"].upper(),
        "Class Weight 'O'": cfg.get("o_weight", 0.35),
        "Viterbi Decoding": "YES" if use_viterbi else "NO",
        "Strict F1 (%)": metrics["strict_f1"],
        "Relaxed F1 (%) [TAC/NIST]": metrics["relaxed_f1"],
        "Relaxed Recall (%)": metrics["relaxed_recall"],
        "Relaxed Precision (%)": metrics["relaxed_precision"],
        "Accuracy (%)": metrics["accuracy"],
    }

    return result_row, (logits, labels)


# ==============================================================================
# เทคนิคที่ 4: Multi-Model Ensemble Averaging
# ==============================================================================
def evaluate_ensemble(models_to_ensemble, test_raw, tag_list, label2id, device, use_viterbi=True):
    """
    นำ Probability distribution ของแต่ละโมเดลมาเฉลี่ยร่วมกัน (Logits/Softmax Averaging)
    แล้วถอดรหัสด้วย Viterbi Constrained BIO Decoder
    """
    print("\n" + "=" * 80)
    print("🤝 [เทคนิคที่ 4: Multi-Model Ensemble Averaging]")
    print(f"   รวมพลังโมเดลทดสอบทั้งหมด {len(models_to_ensemble)} โมเดล:")
    for m in models_to_ensemble:
        print(f"     - {m.name}")
    print("=" * 80)

    valid_models = []
    all_models_word_probs = []
    ground_truth_word_labels = None

    for m_dir in models_to_ensemble:
        res, (logits, labels) = evaluate_single_model(m_dir, test_raw, tag_list, label2id, device, use_viterbi=False)
        if res is None or res.get("Strict F1 (%)", 0) == 0.0 or np.isnan(logits).any():
            print(f"   ⚠️ ไม่นำ {m_dir.name} มารวมใน Ensemble (เนื่องจากค่าน้ำหนักผิดปกติหรือ F1 เป็น 0)")
            continue
        valid_models.append(m_dir.name)

        model_sent_word_probs = []
        model_sent_truth = []
        for sent_idx in range(len(logits)):
            sent_logits = logits[sent_idx]
            sent_labs = labels[sent_idx]
            valid_mask = (sent_labs != -100)
            if not np.any(valid_mask):
                continue
            valid_logits = sent_logits[valid_mask]
            v_probs = torch.softmax(torch.tensor(valid_logits), dim=-1).numpy()
            model_sent_word_probs.append(v_probs)
            if ground_truth_word_labels is None:
                model_sent_truth.append(sent_labs[valid_mask])

        all_models_word_probs.append(model_sent_word_probs)
        if ground_truth_word_labels is None:
            ground_truth_word_labels = model_sent_truth

    if len(all_models_word_probs) == 0:
        print("⚠️ ไม่พบโมเดลที่สมบูรณ์สำหรับทำ Ensemble")
        return None

    print(f"   ✅ รวมพลังเฉพาะโมเดลที่สมบูรณ์ {len(valid_models)} โมเดล: {', '.join(valid_models)}")

    # คำนวณความน่าจะเป็นเฉลี่ยระดับคำ (Word-level Ensemble Averaging)
    valid_trans = build_valid_transition_matrix(label2id) if use_viterbi else None

    true_predictions = []
    true_labels = []

    num_sentences = len(ground_truth_word_labels)
    for sent_idx in range(num_sentences):
        # เฉลี่ยความน่าจะเป็นของทุกโมเดลในประโยคที่ sent_idx
        sent_word_probs = np.mean([model_probs[sent_idx] for model_probs in all_models_word_probs], axis=0)
        valid_truth = ground_truth_word_labels[sent_idx]

        valid_logits = np.log(sent_word_probs + 1e-12)

        if use_viterbi:
            pred_path = viterbi_bio_decode(valid_logits, valid_trans, label2id)
        else:
            pred_path = np.argmax(valid_logits, axis=-1)

        true_predictions.append([tag_list[idx] for idx in pred_path])
        true_labels.append([tag_list[idx] for idx in valid_truth])

    metrics = compute_span_metrics(true_predictions, true_labels)

    ensemble_row = {
        "Experiment": "👑 Ensemble (Top Models Combined)",
        "Model": "ENSEMBLE (Multi-Backbone)",
        "Features": "COMBINED",
        "Class Weight 'O'": "OPTIMIZED",
        "Viterbi Decoding": "YES" if use_viterbi else "NO",
        "Strict F1 (%)": metrics["strict_f1"],
        "Relaxed F1 (%) [TAC/NIST]": metrics["relaxed_f1"],
        "Relaxed Recall (%)": metrics["relaxed_recall"],
        "Relaxed Precision (%)": metrics["relaxed_precision"],
        "Accuracy (%)": metrics["accuracy"],
    }
    return ensemble_row


# ==============================================================================
# Main Function
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Step 3: Evaluation V3 Ultra Edition")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="โฟลเดอร์โมเดลที่ต้องการวัดผลเดี่ยว")
    parser.add_argument("--all", action="store_true",
                        help="ตรวจข้อสอบทุกโมเดลใน models/ และสร้างตารางสรุปผลเปรียบเทียบ")
    parser.add_argument("--ensemble", action="store_true",
                        help="เปิดใช้งานโหมด Multi-Model Ensemble Averaging")
    parser.add_argument("--no_viterbi", action="store_true",
                        help="ปิดการใช้งาน Viterbi BIO decoding")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("🎯 [Step 3 - V3 Ultra Edition] การประเมินผลบน Test Set (ข้อสอบจริง 100 ข่าวของอาจารย์)")
    print("   ตรวจทั้ง 2 เกณฑ์: Strict F1 (Exact Match) และ Relaxed F1 (TAC/NIST Overlap)")
    print(f"   Viterbi BIO Decoding: {'เปิดใช้งาน (Enforce Valid Spans)' if not args.no_viterbi else 'ปิด'}")
    print(f"   อุปกรณ์ (Device):     {device.upper()}")
    print("=" * 80)

    with open(DATA_DIR / "feature_mappings.json", 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    label2id = mappings['casie']['label2id']
    tag_list = mappings['casie']['tag_list']

    with open(DATA_DIR / "test.json", 'r', encoding='utf-8') as f:
        test_raw = json.load(f)

    print(f"📄 โหลดข้อสอบ Test Set: {len(test_raw):,} ประโยค")

    all_results = []
    models_found = []

    if args.all or args.ensemble:
        models_found = sorted([p for p in MODELS_DIR.iterdir() if p.is_dir() and (p / "experiment_config.json").exists()])
        print(f"🔍 พบโมเดลที่ฝึกสอนเสร็จแล้ว {len(models_found)} โมเดลใน models/:")
        for m in models_found:
            print(f"   - {m.name}")

        for m_dir in models_found:
            print(f"\n📊 กำลังประเมินผล: {m_dir.name}...")
            res, _ = evaluate_single_model(m_dir, test_raw, tag_list, label2id, device, use_viterbi=(not args.no_viterbi))
            if res:
                all_results.append(res)
                print(f"   Strict F1: {res['Strict F1 (%)']}% | Relaxed F1 [NIST TAC]: {res['Relaxed F1 (%) [TAC/NIST]']}%")

        # ตรวจสอบว่าเปิดโหมด Ensemble หรือไม่
        if args.ensemble and len(models_found) >= 2:
            ens_res = evaluate_ensemble(models_found, test_raw, tag_list, label2id, device, use_viterbi=(not args.no_viterbi))
            if ens_res is not None:
                all_results.append(ens_res)
                print(f"\n👑 ผลลัพธ์ ENSEMBLE:")
                print(f"   Strict F1: {ens_res['Strict F1 (%)']}% | Relaxed F1 [NIST TAC]: {ens_res['Relaxed F1 (%) [TAC/NIST]']}%")

    elif args.model_dir:
        res, _ = evaluate_single_model(Path(args.model_dir), test_raw, tag_list, label2id, device, use_viterbi=(not args.no_viterbi))
        if res:
            all_results.append(res)

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = MODELS_DIR / "master_ablation_results_v3.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print("\n" + "=" * 90)
        print("🏆 ตารางสรุปผลการทดลองเปรียบเทียบโมเดล V3 (Master Ablation Comparison Table)")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)
        print(f"💾 บันทึกตาราง Master เรียบร้อยที่: {csv_path.resolve()}\n")


if __name__ == "__main__":
    main()
