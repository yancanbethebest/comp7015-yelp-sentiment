# -*- coding: utf-8 -*-
"""命令行推理 / Command-line inference.

三个用处 / three use cases:

1. 快速抽查一句话 / sanity-check a single sentence::

       python -m models.lstm_gru_random.predict \\
           --checkpoint runs/member2_bilstm_random_full_seed42/best_model.pt \\
           --text "The food was great but the service was slow."

2. 批量打分一个 CSV（成员五做错误分析时用），输出带预测与概率的新 CSV::

       python -m models.lstm_gru_random.predict \\
           --checkpoint runs/member2_bilstm_random_full_seed42/best_model.pt \\
           --input-csv processed_data/val.csv \\
           --output-csv runs/member2_bilstm_random_full_seed42/val_predictions.csv

3. 直接 import 复用（成员四/五）::

       from models.lstm_gru_random import load_checkpoint, predict_texts
       model, vocab, payload = load_checkpoint("runs/<run>/best_model.pt")
       rows = predict_texts(model, vocab, ["terrible food"], max_len=payload["max_len"])

注意 / note:
    输出概率**未做温度校准**，不应作为可靠置信度使用；`max_len` 必须与训练时一致。
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch

if __package__:  # 作为包导入 / imported as a package
    from .model import load_checkpoint, predict_texts
else:            # 直接以脚本方式运行 / executed as a plain script
    from model import load_checkpoint, predict_texts  # type: ignore


def score_frame(model, vocab, payload, frame: pd.DataFrame, text_column: str = "text",
                batch_size: int = 64) -> pd.DataFrame:
    """用**已加载**的模型给 DataFrame 的每一行打分（避免重复读 checkpoint）。

    Args:
        model: 已加载的模型。
        vocab: 配套词表。
        payload: checkpoint 内容（用到其中的 ``max_len``）。
        frame: 待打分数据，至少包含 ``text_column``。
        text_column: 文本列名。
        batch_size: 推理批大小。

    Returns:
        在原有列基础上追加 ``prediction`` / ``sentiment`` / ``prob_positive`` 三列。
    """
    device = next(model.parameters()).device
    texts = frame[text_column].astype(str).tolist()
    rows = predict_texts(model, vocab, texts, max_len=int(payload["max_len"]),
                         device=device, batch_size=batch_size)

    output = frame.copy()
    output["prediction"] = [row["label"] for row in rows]
    output["sentiment"] = [row["sentiment"] for row in rows]
    output["prob_positive"] = [row["prob_positive"] for row in rows]
    return output


def predict_dataframe(checkpoint: Path, frame: pd.DataFrame, text_column: str = "text",
                      batch_size: int = 64) -> pd.DataFrame:
    """便捷版本：自己加载 checkpoint 再打分 / convenience wrapper around :func:`score_frame`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, payload = load_checkpoint(checkpoint, device=device)
    return score_frame(model, vocab, payload, frame,
                       text_column=text_column, batch_size=batch_size)


def build_arg_parser() -> argparse.ArgumentParser:
    """构造参数解析器 / build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m models.lstm_gru_random.predict",
        description="用已训练好的 BiLSTM/BiGRU checkpoint 做情感预测。"
                    "Predict sentiment with a trained checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例 / examples:\n"
            "  python -m models.lstm_gru_random.predict \\\n"
            "      --checkpoint runs/member2_bilstm_random_full_seed42/best_model.pt \\\n"
            "      --text \"Great food, friendly staff\" --text \"Cold food and rude waiter\"\n\n"
            "  python -m models.lstm_gru_random.predict \\\n"
            "      --checkpoint runs/member2_bilstm_random_full_seed42/best_model.pt \\\n"
            "      --input-csv processed_data/val.csv \\\n"
            "      --output-csv runs/member2_bilstm_random_full_seed42/val_predictions.csv\n"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="模型权重路径（best_model.pt 或 last_full.pt）")
    parser.add_argument("--text", action="append", default=None,
                        help="待预测的句子，可重复传入多个")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="批量打分：输入 CSV（需含 text 列）")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="批量打分的输出 CSV 路径")
    parser.add_argument("--text-column", type=str, default="text",
                        help="CSV 中的文本列名（默认 text）")
    parser.add_argument("--batch-size", type=int, default=64, help="推理批大小（默认 64）")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口 / CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.text and not args.input_csv:
        parser.error("至少要给 --text 或 --input-csv 之一 / need --text or --input-csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, payload = load_checkpoint(args.checkpoint, device=device)
    print(f"checkpoint : {args.checkpoint}")
    print(f"model      : {payload['model']['rnn_type'].upper()}  "
          f"epoch={payload['epoch']}  max_len={payload['max_len']}  "
          f"vocab={payload['vocab_size']:,}  best_val_f1={float(payload.get('best_score') or 0.0):.4f}")
    print(f"device     : {device}")

    if args.text:
        rows = predict_texts(model, vocab, list(args.text),
                             max_len=int(payload["max_len"]), device=device,
                             batch_size=args.batch_size)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))

    if args.input_csv:
        frame = pd.read_csv(Path(args.input_csv), encoding="utf-8")
        scored = score_frame(model, vocab, payload, frame,
                             text_column=args.text_column, batch_size=args.batch_size)
        output_path = Path(args.output_csv) if args.output_csv else \
            Path(args.input_csv).with_name(Path(args.input_csv).stem + "_predictions.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False, encoding="utf-8")
        print(f"已写出 / written: {output_path}  ({len(scored):,} rows)")
        if "label" in scored.columns:
            accuracy = float((scored["prediction"] == scored["label"]).mean())
            print(f"与真实标签比较 accuracy = {accuracy:.4f}（仅用于自查，正式指标请用 train.py）")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
