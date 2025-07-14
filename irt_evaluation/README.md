# IRT-based Knowledge Editing Evaluation System

項目反応理論（IRT）を用いた継続的知識編集評価システム

## 概要

このシステムは、大規模言語モデル（LLM）に対する継続的知識編集実験の結果を項目反応理論（IRT）の枠組みで分析するための包括的ツールです。ROME、MEMIT、MENDなどの知識編集手法の性能を定量的に評価し、研究論文に使用可能な図表とレポートを自動生成します。

## 主な機能

### 🔍 データ処理・変換
- 実験ログ（JSON/CSV）の自動読み込み・検証
- IRT分析用データ形式への変換
- 人物-項目行列の生成
- 欠損値・外れ値の処理

### 📊 IRT分析
- 1PL（Raschモデル）、2PL、3PLモデルの推定
- EM/MCMC/MLE推定手法のサポート
- モデル比較・選択（AIC/BIC基準）
- 項目特性曲線（ICC）の生成

### 📈 可視化
- 項目特性曲線（ICC）プロット
- パラメータ分布図
- 人物-項目マップ（Wright map）
- 手法別性能比較図
- 実験条件別分析図
- 研究用ダッシュボード

### 📋 レポート生成
- HTML/PDF/LaTeX形式のレポート
- 統計的解釈・推奨事項の自動生成
- 論文投稿用の表・図の作成
- 研究結果の要約

## システム構成

```
irt_evaluation/
├── main.py                 # メインエントリーポイント
├── config.yaml            # システム設定ファイル
├── log_loader.py          # 実験ログ読み込み
├── data_converter.py      # IRT用データ変換
├── fit_irt.py             # IRTモデル推定
├── visualizer.py          # 図表生成
├── reporter.py            # レポート作成
└── README.md              # このファイル
```

## インストール・依存関係

### 必須ライブラリ
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install pyyaml jinja2 pathlib
```

### オプションライブラリ
```bash
# IRT専用ライブラリ（推奨）
pip install pyirt

# PDF生成用
pip install weasyprint

# 統計解析用
pip install scipy statsmodels
```

## 使用方法

### 1. 基本的な使用例

```bash
# 実験結果ディレクトリを指定して完全分析を実行
python main.py --input results/ --config config.yaml

# 単一実験ファイルを分析
python main.py --input experiment.json --model-type 2PL

# 複数IRTモデルの比較
python main.py --input results/ --compare-models 1PL 2PL 3PL
```

### 2. 設定ファイルのカスタマイズ

`config.yaml`で以下を設定可能：

```yaml
# IRT分析設定
irt_model:
  model_type: '2PL'           # 1PL, 2PL, 3PL
  estimation_method: 'EM'     # EM, MCMC, MLE
  max_iterations: 1000

# 可視化設定
visualization:
  figure:
    size: [10, 8]
    dpi: 300
    format: 'png'
  
# レポート設定
reporting:
  format: 'html'              # html, pdf, latex
```

### 3. 段階的実行

```bash
# データ変換のみ
python main.py --input results/ --step data_conversion

# 可視化のみ（既存のIRT結果から）
python main.py --input-irt irt_data.csv --results irt_results.json --step visualization

# レポート生成のみ
python main.py --input-irt irt_data.csv --results irt_results.json --step reporting
```

## 入力データ形式

### 実験ログ（JSON形式）

`run_knowledge_editing_new_order_sampling.py`の出力形式に対応：

```json
{
  "method": "ROME",
  "model_name": "gpt2-xl",
  "condition": "A",
  "individual_results": [
    {
      "sample_index": 1,
      "edits": [
        {
          "edit_order": 1,
          "triple": {
            "subject": "Person A",
            "relation": "Skills",
            "object": "Programming",
            "candidates": ["Programming", "Design", "Writing", "Teaching", "Research"]
          },
          "post_edit_probabilities": {
            "probabilities": [0.85, 0.05, 0.04, 0.03, 0.03]
          }
        }
      ]
    }
  ]
}
```

### CSV形式

以下の列を含むCSV形式にも対応：

| 列名 | 説明 |
|------|------|
| method | 編集手法（ROME, MEMIT, MEND等） |
| model_name | モデル名（gpt2-xl, gpt-j-6b等） |
| condition | 実験条件（A, B, C） |
| sample_index | サンプル番号 |
| edit_order | 編集順序 |
| subject | 被編集対象 |
| relation | 関係性 |
| object | 対象オブジェクト |
| target_probability | 目標確率 |
| target_rank | 目標ランク |
| is_correct | 正解フラグ |

## 出力

### ディレクトリ構成

```
output/
├── irt_results/           # IRT分析結果
│   ├── raw_data.csv
│   ├── irt_data.csv
│   ├── irt_results.json
│   └── model_comparison.json
├── figures/               # 生成図表
│   ├── icc_plots.png
│   ├── parameter_distributions.png
│   ├── method_performance.png
│   └── summary_dashboard.png
├── reports/               # 分析レポート
│   └── irt_analysis_report.html
├── tables/                # 論文用表
│   ├── model_comparison.csv
│   ├── method_performance.csv
│   └── condition_analysis.csv
└── logs/                  # ログファイル
    └── irt_evaluation.log
```

### 主要出力ファイル

1. **IRT分析結果** (`irt_results.json`)
   - θ (人物能力): 手法・モデル組合せの能力
   - β (項目難易度): 各編集タスクの難易度
   - α (識別力): 各項目の識別力
   - 適合度統計量（AIC, BIC）

2. **可視化図表**
   - 項目特性曲線（ICC）
   - パラメータ分布図
   - 手法比較図
   - 実験条件別分析図

3. **分析レポート** (HTML/PDF)
   - 実行要約
   - 統計的結果
   - 解釈・考察
   - 研究推奨事項

## IRTモデルの解釈

### パラメータの意味

- **θ (シータ)**: 人物能力
  - (method, model)の組合せごとの編集能力
  - 高いほど編集成功率が高い

- **β (ベータ)**: 項目難易度  
  - 各編集タスクの困難さ
  - 高いほど編集が困難

- **α (アルファ)**: 識別力
  - 能力差を識別する項目の能力
  - 高いほど能力差を明確に区別

### 分析結果の活用

1. **手法比較**: θの平均値で手法の有効性を比較
2. **条件分析**: βの分布で実験条件の困難度を評価
3. **改善点特定**: 低いθや高いβを持つ項目の詳細分析

## 実験設計への応用

### 継続的知識編集の評価軸

- **Condition A**: 異なる被験体への逐次編集
- **Condition B**: 同一被験体への複数関係編集  
- **Condition C**: 同一（被験体、関係）の対象再編集

### 関係タイプ

- **共有関係**: 複数対象を許可（累積的）
  - 例: Skills, Hobbies, Languages
- **排他関係**: 単一対象のみ（上書き的）
  - 例: Job, Residence, Health Status

## トラブルシューティング

### よくある問題

1. **モデル収束しない**
   ```bash
   # 反復回数を増やす
   --config で max_iterations を調整
   ```

2. **メモリ不足**
   ```bash
   # データのサンプリング
   python main.py --input results/ --sample-rate 0.5
   ```

3. **依存ライブラリエラー**
   ```bash
   # 必要最小限で実行
   pip install pandas numpy matplotlib pyyaml
   ```

### ログ確認

```bash
# 詳細ログの確認
tail -f output/logs/irt_evaluation.log

# デバッグモード
python main.py --input results/ --verbose
```

## 研究例・論文応用

### 統計的検定

```python
# 手法間の有意差検定
from scipy import stats
t_stat, p_value = stats.ttest_ind(theta_ROME, theta_MEMIT)
```

### 効果量計算

```python
# Cohen's d
def cohens_d(group1, group2):
    pooled_std = np.sqrt(((group1.var() + group2.var()) / 2))
    return (group1.mean() - group2.mean()) / pooled_std
```

### 論文用表の作成

システムが自動生成するCSV表をLaTeXに変換：

```bash
# 表をLaTeX形式で出力
python main.py --input results/ --report-format latex
```

## 拡張・カスタマイズ

### 新しい評価メトリクスの追加

1. `data_converter.py`でスコア計算ロジックを追加
2. `visualizer.py`で新しい図表を実装
3. `reporter.py`で解釈を追加

### 外部IRTライブラリとの連携

```python
# config.yamlで設定
integration:
  optional_libraries:
    pyirt: true
    stan: true    # PyStan使用
    jags: false   # JAGS使用
```

## ライセンス・引用

このシステムを研究で使用する場合は、以下の形式で引用してください：

```bibtex
@software{irt_knowledge_editing_2024,
  title={IRT-based Knowledge Editing Evaluation System},
  author={Knowledge Editing Research Team},
  year={2024},
  url={https://github.com/your-repo/irt-evaluation}
}
```

## サポート・コントリビューション

- Issues: バグ報告・機能要求
- Pull Requests: コード貢献
- Discussions: 使用方法に関する質問

## 更新履歴

- v1.0.0: 初期リリース
  - 基本的なIRT分析機能
  - HTML/PDFレポート生成
  - 知識編集実験対応