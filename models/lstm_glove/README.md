# lstm_glove

本目录包含完整模型交付文件：

| 文件 | 用途 |
|---|---|
| `best_model.pt` | 按验证 Macro-F1 选择的最佳 checkpoint（第 10 轮），通过 Git LFS 存储 |
| `model.py` | 模型架构，以及 `load_checkpoint()` 加载函数 |
| `train.py` | 读取本目录配置的训练代码 |
| `config.yaml` | 训练参数与已发布实验的实际轮次记录 |
| `metrics.json` | 已发布模型的完整训练/验证指标 |

在仓库根目录运行：

```bash
git lfs install
git lfs pull
python -m pip install -r requirements-training.txt
python models/lstm_glove/train.py --config models/lstm_glove/config.yaml
```

命令行参数可覆盖 YAML 的 `training` 配置，例如 `--epochs 5`。默认结果写到 `runs/lstm_glove/`，不会覆盖已发布的最佳权重。数据路径相对当前工作目录，因此请从仓库根目录运行。默认不评估测试集。

`published_run` 是历史实验说明，不控制训练；LSTM 的上限 100 轮与实际手动停止轮数须区分。checkpoint 不包含优化器状态，不支持精确断点续训。

重新训练需要 `glove.6B.100d.txt`（可用 `--embedding-file` 指定路径）；加载已有 checkpoint 不需要原始 GloVe 文件。

加载示例（在本模型目录中执行）：

```python
from model import load_checkpoint
model, checkpoint = load_checkpoint("best_model.pt", device="cpu")
```

若 GitHub 下载 ZIP 中的 `.pt` 只是几行文字，则那是 LFS 指针，请使用 `git lfs pull` 获取真实权重。
