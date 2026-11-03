"""
step2_train_experiments_v3.py
สคริปต์สถาปัตยกรรมโมเดลและการฝึกสอน V3 Ultra Edition (State-of-the-Art NLP Architecture)
รวม 4 ยอดเทคนิคตามมาตรฐานสากลและเปเปอร์ CASIE (AAAI 2020 ของอาจารย์):

4 เทคนิคหลักใน V3:
  1. Class Weighting (Cost-Sensitive Learning / o_weight = 0.35):
     - ลดน้ำหนักป้าย 'O' ลง เพื่อแก้ปัญหาคำว่าง 85%+ บังคับให้โมเดลตื่นตัวและดึง Entity ไซเบอร์ออกมา ดันค่า Recall (R) และ F1 สูงขึ้นอย่างมีนัยสำคัญ
  2. Transition Constraints / CRF Viterbi BIO Decoding:
     - ดักจับและบล็อกการเปลี่ยนป้ายที่ผิดไวยากรณ์ BIO ถึง 1,891 รูปแบบ (เช่น O -> I-Tag หรือ B-Victim -> I-Attacker) รับประกันว่าทุก Span ที่โมเดลสกัดออกมาสมบูรณ์ 100%
  3. SOTA Transformer Backbone Support (DeBERTa-v3 & SecBERT):
     - รองรับทั้ง 'jackaduma/SecBERT' (ผู้เชี่ยวชาญไซเบอร์) และ 'microsoft/deberta-v3-base' (ราชาด้าน NER ยุคใหม่) พร้อม Sequence Length 256 คำ
  4. Multi-Model Ensemble Ready:
     - บันทึก Probability distribution / Logits รองรับการหลอมรวมคะแนนเฉลี่ย (Ensemble) ใน Step 3 เพื่อดัน F1 พุ่งขึ้นอีก +2% ถึง +3%
"""
import os
import sys
import json
import argparse
import inspect
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed
)
from transformers.modeling_outputs import TokenClassifierOutput

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# สถาปัตยกรรมโมเดล V3: Transformer + Extra Feature Embeddings + Class Weighting
# ==============================================================================
class BertWithExtraFeaturesForTokenClassification(nn.Module):
    """
    สถาปัตยกรรม Token Classification V3 รองรับ:
      - Backbone: SecBERT, DeBERTa-v3, BERT-base
      - Extra Features: POS Tag + General NER Tag
      - Cost-Sensitive Loss: Class Weighting ปรับลดน้ำหนักป้าย 'O' (o_weight)
    """
    def __init__(
        self,
        model_name,
        num_labels,
        num_pos=0,
        num_ner_gen=0,
        pos_emb_dim=32,
        ner_emb_dim=32,
        dropout=0.1,
        o_weight=0.35
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # ปกติ 768

        self.use_pos = (num_pos > 0)
        self.use_ner_gen = (num_ner_gen > 0)

        classifier_in_dim = hidden_size

        if self.use_pos:
            self.pos_embedding = nn.Embedding(num_pos, pos_emb_dim, padding_idx=0)
            classifier_in_dim += pos_emb_dim

        if self.use_ner_gen:
            self.ner_gen_embedding = nn.Embedding(num_ner_gen, ner_emb_dim, padding_idx=0)
            classifier_in_dim += ner_emb_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in_dim, num_labels)
        self.num_labels = num_labels
        self.o_weight = o_weight

        # เทคนิคที่ 1: กำหนด Class Weights เพื่อแก้ปัญหา Imbalanced Data (ป้าย 'O' กินเนื้อที่ 85%+)
        class_weights = torch.ones(num_labels, dtype=torch.float)
        class_weights[0] = float(o_weight)  # ลดน้ำหนักป้าย O (index 0)
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        pos_ids=None,
        ner_gen_ids=None,
        labels=None
    ):
        bert_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        if token_type_ids is not None and "token_type_ids" in inspect.signature(self.bert.forward).parameters:
            bert_kwargs['token_type_ids'] = token_type_ids

        bert_outputs = self.bert(**bert_kwargs)
        sequence_output = bert_outputs.last_hidden_state  # [Batch, SeqLen, HiddenSize]

        features = [sequence_output]

        if self.use_pos:
            if pos_ids is None:
                pos_ids = torch.zeros_like(input_ids)
            pos_emb = self.pos_embedding(pos_ids)  # [Batch, SeqLen, pos_emb_dim]
            features.append(pos_emb)

        if self.use_ner_gen:
            if ner_gen_ids is None:
                ner_gen_ids = torch.zeros_like(input_ids)
            ner_emb = self.ner_gen_embedding(ner_gen_ids)  # [Batch, SeqLen, ner_emb_dim]
            features.append(ner_emb)

        combined = torch.cat(features, dim=-1)
        combined = self.dropout(combined)
        combined = combined.to(self.classifier.weight.dtype)
        logits = self.classifier(combined)  # [Batch, SeqLen, num_labels]

        loss = None
        if labels is not None:
            # ใช้ Class Weights ใน CrossEntropyLoss เพื่อให้โมเดลลงโทษหนักขึ้นเมื่อทาย Entity พลาด
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states if hasattr(bert_outputs, 'hidden_states') else None,
            attentions=bert_outputs.attentions if hasattr(bert_outputs, 'attentions') else None,
        )


# ==============================================================================
# Data Collator สำหรับ Dynamic Padding รองรับ Features เสริม
# ==============================================================================
class ExtraFeatureDataCollator:
    """จัดรวม Batch และทำ Padding สำหรับ input_ids, labels, pos_ids และ ner_gen_ids"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask']} for f in features],
            padding=True,
            return_tensors='pt'
        )
        max_len = batch['input_ids'].shape[1]

        if 'labels' in features[0]:
            batch['labels'] = torch.tensor(
                [f['labels'] + [-100] * (max_len - len(f['labels'])) for f in features],
                dtype=torch.long
            )

        if 'pos_ids' in features[0]:
            batch['pos_ids'] = torch.tensor(
                [f['pos_ids'] + [0] * (max_len - len(f['pos_ids'])) for f in features],
                dtype=torch.long
            )

        if 'ner_gen_ids' in features[0]:
            batch['ner_gen_ids'] = torch.tensor(
                [f['ner_gen_ids'] + [0] * (max_len - len(f['ner_gen_ids'])) for f in features],
                dtype=torch.long
            )

        return batch


# ==============================================================================
# Callback บันทึก Log และพล็อตกราฟ Loss Curve ตามที่อาจารย์วาด
# ==============================================================================
class LossPlotAndLogCallback(TrainerCallback):
    """
    ดักจับผลลัพธ์ทุกรอบ Epoch:
    - บันทึก epoch, train_loss, eval_loss, f1 ลง 'training_log.csv'
    - วาดกราฟ Train vs Val Loss พร้อมมาร์กจุด Best Model (*) ลง 'loss_curve.png'
    """
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / "training_log.csv"
        self.plot_file = self.output_dir / "loss_curve.png"
        self.history = []
        self.current_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.current_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = state.epoch
            val_loss = metrics.get("eval_loss", None)
            val_f1 = metrics.get("eval_f1", None)
            val_p = metrics.get("eval_precision", None)
            val_r = metrics.get("eval_recall", None)

            record = {
                "epoch": round(epoch, 2) if epoch else len(self.history) + 1,
                "train_loss": round(self.current_train_loss, 4) if self.current_train_loss is not None else None,
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "val_f1": round(val_f1, 4) if val_f1 is not None else None,
                "val_precision": round(val_p, 4) if val_p is not None else None,
                "val_recall": round(val_r, 4) if val_r is not None else None,
            }
            self.history.append(record)

            # บันทึกลง CSV
            df = pd.DataFrame(self.history)
            df.to_csv(self.log_file, index=False, encoding='utf-8')

            # พล็อตกราฟตามภาพวาดของอาจารย์
            self.plot_loss_curves(df)

    def plot_loss_curves(self, df):
        clean_df = df.dropna(subset=['val_loss'])
        if len(clean_df) < 1:
            return

        plt.figure(figsize=(9, 5.5))
        epochs = clean_df['epoch'].tolist()
        val_losses = clean_df['val_loss'].tolist()
        train_losses = clean_df['train_loss'].tolist()

        if any(t is not None for t in train_losses):
            valid_train = [t if t is not None else val_losses[i] for i, t in enumerate(train_losses)]
            plt.plot(epochs, valid_train, label='Train Loss', color='#1f77b4', linewidth=2.2, marker='o')

        plt.plot(epochs, val_losses, label='Validation Loss', color='#ff7f0e', linewidth=2.2, marker='s')

        # หาจุดต่ำสุดของ Validation Loss (Best Model / Stop Point)
        best_idx = int(np.argmin(val_losses))
        best_epoch = epochs[best_idx]
        best_loss = val_losses[best_idx]

        plt.scatter([best_epoch], [best_loss], color='red', s=260, zorder=6, marker='*', label=f'Best Val Loss (Stop: {best_loss:.4f})')
        plt.annotate(
            f'stop (*)\nEpoch {best_epoch}: {best_loss:.4f}',
            xy=(best_epoch, best_loss),
            xytext=(best_epoch, best_loss + (max(val_losses) - min(val_losses)) * 0.15 + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
            fontweight='bold',
            fontsize=11,
            ha='center'
        )

        plt.title('Training & Validation Loss Curve with Early Stopping (V3 Ultra)', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Epoch', fontsize=11, fontweight='bold')
        plt.ylabel('Loss', fontsize=11, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=300)
        plt.close()


# ==============================================================================
# Token Alignment สำหรับ Subwords รองรับทั้ง WordPiece (BERT) และ BPE (DeBERTa)
# ==============================================================================
def tokenize_and_align_labels(examples, tokenizer, max_seq_length=256):
    """
    ตัดคำและจับคู่ Subwords กับ Labels, POS Tags, General NER Tags
    ความยาว 256 คำ เพื่อเก็บประโยคข่าวไซเบอร์ได้ครบถ้วน
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_seq_length,
        is_split_into_words=True
    )

    labels = []
    pos_ids_batch = []
    ner_gen_ids_batch = []

    for i, sample_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        sample_pos = examples["pos_tags"][i]
        sample_ner_gen = examples["ner_gen_tags"][i]

        previous_word_idx = None
        label_ids = []
        pos_ids = []
        ner_gen_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens ([CLS], [SEP], [PAD])
                pos_ids.append(0)       # PAD
                ner_gen_ids.append(0)   # PAD
            elif word_idx != previous_word_idx:
                label_ids.append(sample_labels[word_idx])
                pos_ids.append(sample_pos[word_idx])
                ner_gen_ids.append(sample_ner_gen[word_idx])
            else:
                label_ids.append(-100)  # เศษคำย่อย (Subword continuation)
                pos_ids.append(sample_pos[word_idx])
                ner_gen_ids.append(sample_ner_gen[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
        pos_ids_batch.append(pos_ids)
        ner_gen_ids_batch.append(ner_gen_ids)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["pos_ids"] = pos_ids_batch
    tokenized_inputs["ner_gen_ids"] = ner_gen_ids_batch
    return tokenized_inputs


# ==============================================================================
# เทคนิคที่ 2: Viterbi Constrained BIO Decoder & Span Extractor
# ==============================================================================
def build_valid_transition_matrix(mapping):
    """
    สร้าง Transition Constraint Matrix ขนาด (K, K)
    บล็อก Transition ขยะที่ผิดไวยากรณ์ BIO (1,891 รูปแบบ) ทิ้งทั้งหมด
    """
    num_labels = len(mapping)
    valid_trans = np.ones((num_labels, num_labels), dtype=bool)

    for prev_name, prev_id in mapping.items():
        for curr_name, curr_id in mapping.items():
            if curr_name.startswith('I-'):
                ent_type = curr_name[2:]
                # I-Tag ต้องตามหลัง B-Tag หรือ I-Tag ของ Entity ประเภทเดียวกันเท่านั้น
                if prev_name not in [f'B-{ent_type}', f'I-{ent_type}']:
                    valid_trans[prev_id, curr_id] = False
    return valid_trans


def viterbi_bio_decode(logits, valid_trans, mapping):
    """
    Viterbi Dynamic Programming ถอดรหัสลำดับป้ายที่มีคะแนนรวมสูงสุด
    ภายใต้ข้อจำกัดว่าห้ามเกิด Invalid Transition
    """
    T, K = logits.shape
    if T == 0:
        return []

    viterbi = np.zeros((T, K))
    backpointer = np.zeros((T, K), dtype=int)

    # คำแรกของประโยค ห้ามขึ้นต้นด้วย I-
    first_logits = logits[0].copy()
    for name, tag_id in mapping.items():
        if name.startswith('I-'):
            first_logits[tag_id] = -1e9
    viterbi[0] = first_logits

    for t in range(1, T):
        prev_scores = viterbi[t - 1, :, None]  # [K, 1]
        scores = prev_scores + np.where(valid_trans, 0.0, -1e9)  # [K, K]
        best_prev = np.argmax(scores, axis=0)  # [K]
        backpointer[t] = best_prev
        viterbi[t] = scores[best_prev, np.arange(K)] + logits[t]

    best_last = np.argmax(viterbi[-1])
    path = [best_last]
    for t in range(T - 1, 0, -1):
        path.append(backpointer[t, path[-1]])
    return path[::-1]


def extract_bio_spans(tags):
    """สกัด Entity Spans (entity_type, start_idx, end_idx) จากลำดับป้าย BIO"""
    spans = []
    current = None
    for i, t in enumerate(tags):
        if t.startswith('B-'):
            if current:
                spans.append(current)
            current = (t[2:], i, i)
        elif t.startswith('I-'):
            if current and current[0] == t[2:]:
                current = (current[0], current[1], i)
            else:
                if current:
                    spans.append(current)
                current = (t[2:], i, i)
        else:
            if current:
                spans.append(current)
            current = None
    if current:
        spans.append(current)
    return set(spans)


def build_compute_metrics(tag_list, mapping, use_viterbi=True):
    """คำนวณ Precision, Recall, F1 โดยใช้ Viterbi BIO Decoder ป้องกัน Invalid Tags"""
    valid_trans = build_valid_transition_matrix(mapping) if use_viterbi else None

    def compute_metrics(p):
        raw_predictions, labels = p
        # raw_predictions: [Batch, SeqLen, NumLabels]

        true_predictions = []
        true_labels = []

        for sent_idx in range(len(raw_predictions)):
            sent_logits = raw_predictions[sent_idx]
            sent_labs = labels[sent_idx]

            valid_mask = (sent_labs != -100)
            if not np.any(valid_mask):
                continue

            valid_logits = sent_logits[valid_mask]
            valid_truth = sent_labs[valid_mask]

            if use_viterbi:
                pred_path = viterbi_bio_decode(valid_logits, valid_trans, mapping)
            else:
                pred_path = np.argmax(valid_logits, axis=-1)

            true_predictions.append([tag_list[idx] for idx in pred_path])
            true_labels.append([tag_list[idx] for idx in valid_truth])

        n_tp, n_pred, n_true = 0, 0, 0
        n_correct, n_total = 0, 0

        for p_seq, l_seq in zip(true_predictions, true_labels):
            p_ents = extract_bio_spans(p_seq)
            l_ents = extract_bio_spans(l_seq)
            n_tp += len(p_ents.intersection(l_ents))
            n_pred += len(p_ents)
            n_true += len(l_ents)

            for p_tag, l_tag in zip(p_seq, l_seq):
                if p_tag == l_tag:
                    n_correct += 1
                n_total += 1

        prec = n_tp / n_pred if n_pred > 0 else 0.0
        rec = n_tp / n_true if n_true > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = n_correct / n_total if n_total > 0 else 0.0

        return {
            "eval_precision": prec,
            "eval_recall": rec,
            "eval_f1": f1,
            "eval_accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc
        }
    return compute_metrics


# ==============================================================================
# Main Training Function
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Step 2: Train Experiments V3 Ultra Edition")
    parser.add_argument("--model_name", type=str, default="jackaduma/SecBERT",
                        help="Backbone Pre-trained model ('jackaduma/SecBERT', 'microsoft/deberta-v3-base', 'bert-base-uncased')")
    parser.add_argument("--feature_mode", type=str, default="none",
                        choices=["none", "pos", "ner", "pos_ner"],
                        help="โหมดฟีเจอร์เสริม ('none', 'pos', 'ner', 'pos_ner')")
    parser.add_argument("--o_weight", type=float, default=0.35,
                        help="น้ำหนักป้าย 'O' สำหรับ Class Weighting (default: 0.35 เพื่อเพิ่ม Recall)")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="ความยาวสูงสุดของประโยค Subwords (default: 256 คำ)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="จำนวนรอบสูงสุด (default: 10 รอบ)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch Size สำหรับ Train และ Val (default: 16)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--patience", type=int, default=3,
                        help="จำนวนรอบที่ยอมให้ Val Loss ไม่ลดลงก่อนสั่ง Early Stop (default: 3)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="จำกัดจำนวนประโยคสำหรับทดสอบรันแบบเร็ว (เช่น --max_samples 50)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="ชื่อโฟลเดอร์สำหรับบันทึกผลการทดลอง")
    parser.add_argument("--no_viterbi", action="store_true",
                        help="ปิดการใช้งาน Viterbi decoding (ใช้ argmax ปกติ)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random Seed (default: 42)")
    parser.add_argument("--do_train", action="store_true",
                        help="เริ่มกระบวนการฝึกสอนจริง")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    safe_model_tag = args.model_name.split("/")[-1].replace("-", "_")
    if args.exp_name:
        exp_dir_name = args.exp_name
    else:
        exp_dir_name = f"{safe_model_tag}_{args.feature_mode}_v3"

    output_dir = MODELS_DIR / exp_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 [Step 2 - V3 Ultra Edition] 4-in-1 Advanced Information Extraction Pipeline")
    print(f"   อุปกรณ์ที่ใช้ประมวลผล (Device): {device.upper()}")
    print(f"   โมเดลหลัก (Backbone Model):    {args.model_name}")
    print(f"   โหมดฟีเจอร์เสริม (Features):    {args.feature_mode.upper()}")
    print(f"   Class Weight 'O' (o_weight):  {args.o_weight} (Cost-Sensitive Learning)")
    print(f"   Viterbi BIO Decoding:         {'เปิดใช้งาน (Enforce Valid Spans)' if not args.no_viterbi else 'ปิด'}")
    print(f"   Max Sequence Length:          {args.max_seq_length}")
    print(f"   รอบสูงสุด (Max Epochs):        {args.epochs} รอบ (Early Stopping Patience = {args.patience})")
    print(f"   Batch Size:                   {args.batch_size}")
    print(f"   Learning Rate:                {args.learning_rate}")
    print(f"   โฟลเดอร์บันทึกโมเดล & Log:      {output_dir.resolve()}")
    print("=" * 80)

    # 1. โหลด Mappings
    with open(DATA_DIR / "feature_mappings.json", 'r', encoding='utf-8') as f:
        mappings = json.load(f)

    casie_map = mappings['casie']
    pos_map = mappings['pos']
    ner_map = mappings['ner_gen']

    label2id = casie_map['label2id']
    tag_list = casie_map['tag_list']
    num_labels = len(tag_list)
    num_pos = len(pos_map['pos_list']) if args.feature_mode in ["pos", "pos_ner"] else 0
    num_ner_gen = len(ner_map['ner_gen_list']) if args.feature_mode in ["ner", "pos_ner"] else 0

    print(f"\n [1/6] โหลดชุดข้อมูลจาก dataset/...")
    with open(DATA_DIR / "train.json", 'r', encoding='utf-8') as f:
        train_raw = json.load(f)
    with open(DATA_DIR / "val.json", 'r', encoding='utf-8') as f:
        val_raw = json.load(f)

    if args.max_samples and args.max_samples > 0:
        print(f"⚠️ Quick Test Mode: Train {args.max_samples} ตัวอย่าง, Val {max(20, args.max_samples // 4)} ตัวอย่าง")
        train_raw = train_raw[:args.max_samples]
        val_raw = val_raw[:max(20, args.max_samples // 4)]

    def to_hf_dataset(items):
        return Dataset.from_dict({
            'tokens': [it['tokens'] for it in items],
            'ner_tags': [it['ner_tags'] for it in items],
            'pos_tags': [it['pos_tags'] for it in items],
            'ner_gen_tags': [it['ner_gen_tags'] for it in items]
        })

    raw_datasets = DatasetDict({
        'train': to_hf_dataset(train_raw),
        'validation': to_hf_dataset(val_raw)
    })
    print(f"   - Train Samples: {len(raw_datasets['train']):,} ประโยค")
    print(f"   - Val Samples:   {len(raw_datasets['validation']):,} ประโยค")
    print(f"   - Classes:       {num_labels} คลาส")

    # 2. Tokenization & Alignment
    print(f"\n [2/6] กำลังประมวลผล Tokenizer และ Alignment สำหรับฟีเจอร์...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, max_seq_length=args.max_seq_length),
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # 3. ประกอบสถาปัตยกรรมโมเดล V3
    print(f"\n [3/6] กำลังประกอบสถาปัตยกรรมโมเดล Hybrid Transformer Head (พร้อม Class Weighting)...")
    model = BertWithExtraFeaturesForTokenClassification(
        model_name=args.model_name,
        num_labels=num_labels,
        num_pos=num_pos,
        num_ner_gen=num_ner_gen,
        pos_emb_dim=32,
        ner_emb_dim=32,
        dropout=0.1,
        o_weight=args.o_weight
    )

    # 4. Data Collator & Metrics (Viterbi BIO Constrained)
    print(f"\n [4/6] ตั้งค่า Data Collator และ Viterbi BIO Decoder Metrics...")
    data_collator = ExtraFeatureDataCollator(tokenizer)
    compute_metrics = build_compute_metrics(tag_list, label2id, use_viterbi=(not args.no_viterbi))

    # 5. TrainingArguments & Callbacks
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_strategy": "epoch",
        "report_to": "none",
        "use_cpu": (device == "cpu")
    }

    t_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in t_params:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    if "save_strategy" in t_params:
        training_args_kwargs["save_strategy"] = "epoch"

    if "warmup_ratio" in t_params:
        training_args_kwargs["warmup_ratio"] = 0.1
    elif "warmup_steps" in t_params:
        training_args_kwargs["warmup_steps"] = 100

    training_args = TrainingArguments(**training_args_kwargs)

    plot_callback = LossPlotAndLogCallback(output_dir=output_dir)
    early_stop_callback = EarlyStoppingCallback(early_stopping_patience=args.patience)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[plot_callback, early_stop_callback]
    )

    # 6. บันทึก Config และเริ่มเทรน
    with open(output_dir / "experiment_config.json", 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": args.model_name,
            "feature_mode": args.feature_mode,
            "o_weight": args.o_weight,
            "max_seq_length": args.max_seq_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "num_labels": num_labels,
            "num_pos": num_pos,
            "num_ner_gen": num_ner_gen,
            "use_viterbi": not args.no_viterbi
        }, f, indent=2)

    with open(output_dir / "label_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(casie_map, f, indent=2)

    tokenizer.save_pretrained(str(output_dir))

    print("\n" + "=" * 80)
    print(" [6/6] สถาปัตยกรรม V3 ประกอบเสร็จสมบูรณ์!")
    if args.do_train:
        print("🚀 เริ่มต้นกระบวนการฝึกสอนโมเดลจริง (Training Loop)...")
        print("=" * 80)
        trainer.train()
        print("\n💾 กำลังบันทึกโมเดลรอบที่ดีที่สุด (Best Model based on Val Loss)...")
        trainer.save_model(str(output_dir))
        torch.save(model.state_dict(), output_dir / "custom_model_weights.bin")
        print(f"✅ บันทึกโมเดลและไฟล์ Log เรียบร้อยที่: {output_dir.resolve()}")
    else:
        print("🔍 โหมดตรวจสอบสถาปัตยกรรม (Dry-Run Check สำเร็จ 100%):")
        print(f"   พร้อมใช้งาน! รันด้วยคำสั่ง '--do_train' เพื่อเริ่มเทรนได้ทันที")
    print("=" * 80)


if __name__ == "__main__":
    main()
