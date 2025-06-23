# 継続知識編集 (CKE) フレームワーク

EasyEditフレームワークをベースとした、大規模言語モデルにおける継続知識編集の包括的評価フレームワークです。

## 概要

このリポジトリは、LLMにおける継続知識編集（CKE）の包括的評価フレームワークを実装し、逐次的な知識編集操作の複雑さと効果に焦点を当てています。

## 主要機能

- **共有vs排他的関係**: 知識挿入・修正の精密なセマンティック制御
- **マルチ条件評価**: 3つの異なる評価シナリオ (A, B, C)
- **逐次編集パイプライン**: EasyEdit統合による包括的な編集手法サポート
- **包括的分析**: 確率ランキングと干渉パターンの検出
- **モック実行**: GPU不要での完全なパイプライン検証

## クイックスタート

### 基本的な使用方法（GPU不要）
```bash
# リポジトリのクローン
git clone https://github.com/ishigaki0302/continual-knowledge-editing.git
cd continual-knowledge-editing

# 基本依存関係のインストール
pip install numpy matplotlib

# デモの実行
python3 demo_experiment.py

# モックLLMを使用した実験実行
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5
```

### 完全セットアップ（GPU必要）
```bash
# Docker セットアップ
docker build -t cke-framework:latest .
docker run -it --ipc=host -p 8501:8501 --gpus all -v $(pwd):/app/CKE --name cke-container cke-framework:latest

# ローカルセットアップ
conda create -n CKE python=3.9.7
conda activate CKE
pip install -r requirements.txt

# 実際のモデルでの実行
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --real-model
```

## プロジェクト構成

```
├── src/
│   ├── continual_editing/     # コアCKEフレームワーク
│   │   ├── conditions/        # 評価条件の実装
│   │   ├── dataset_builder.py # データセット構築
│   │   ├── evaluation_framework.py # 評価フレームワーク
│   │   └── relation_types.py  # 関係タイプ定義
│   ├── experiments/           # 実験実装
│   │   ├── data_sampler.py    # データサンプリング
│   │   ├── rome_experiments.py # ROME実験
│   │   ├── evaluation_metrics.py # 評価指標
│   │   └── visualization.py   # 結果可視化
│   └── utils/                 # ユーティリティ
│       ├── easyedit_wrapper.py # EasyEdit統合ラッパー
│       ├── mock_llm.py        # モックLLM（テスト用）
│       └── data_utils.py      # データ処理
├── easyedit_base/             # EasyEdit統合
│   ├── easyeditor/           # EasyEditコアモジュール
│   ├── hparams/              # 全手法のハイパーパラメータ設定
│   └── examples/             # 参考実装例
├── datasets/                  # 評価データセット
│   └── temp_ckndata.json     # 実験用データ
├── results/                   # 実験結果出力
├── run_ckn_experiment.py     # メイン実験実行スクリプト
└── demo_experiment.py        # デモスクリプト
```

## 実験フレームワーク

### データ構成
`temp_ckndata.json`を使用した実験データ：
- **5つの主語**: 石垣龍馬、鈴木順大、岩瀬駿、平本伶弥、関口雅人
- **共有関係**: Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces（蓄積的）
- **排他関係**: Health Status, Job, Residence, CurrentLocation, AgeGroup（上書き）

### 実験条件

#### 条件A: 異なる主語での逐次編集
- 各編集で異なる主語を使用
- 主語間の干渉パターンを評価

#### 条件B: 同一主語での複数関係編集  
- 同じ主語に対して異なる関係を編集
- 関係間の相互作用を分析

#### 条件C: 同一(主語、関係)での再編集
- **共有関係**: 蓄積的動作（新しい知識が追加される）
- **排他関係**: 上書き動作（古い知識が置き換えられる）

### 利用可能な知識編集手法
- **ROME**: Rank-One Model Editing
- **MEMIT**: Mass Editing Memory in a Transformer  
- **MEND**: Model Editor Networks using Gradient Decomposition
- **FT**: Fine-Tuning
- **IKE**: In-Context Knowledge Editing
- **KN**: Knowledge Neurons
- **SERAC**: Semi-parametric Editing with a Retrieval-Augmented Counterfactual Model

### 対応モデル
- GPT-J-6B, GPT-2-XL
- LLaMA (7B, 3.2-3B), LLaMA-3 (8B)
- Qwen (7B, 2.5-7B), ChatGLM (2-6B, 4-9B)
- Mistral-7B, Baichuan-7B, InternLM-7B
- マルチモーダル: BLIP-2, MiniGPT-4, LLaVA, Qwen2-VL

## 使用例

### 基本的な実験実行
```bash
# ROME手法でGPT-J-6Bを使用（5つの編集）
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5

# MEMIT手法でGPT-2-XLを使用（3つの編集）
python3 run_ckn_experiment.py --method MEMIT --model gpt2-xl --num-edits 3

# 実際のモデルを使用（GPU必要）
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --real-model

# カスタム出力ファイル指定
python3 run_ckn_experiment.py --method MEND --model llama-7b --output my_results.json
```

### デモの実行
```bash
# フレームワークの機能を確認
python3 demo_experiment.py
```

## 実験結果

実験結果は`results/`ディレクトリにJSON形式で保存されます：
- 各編集の効果性（Efficacy）と局所性（Locality）
- 条件別の知識保持率
- 干渉パターンの分析
- 条件間の比較統計

## モック vs 実際の実験

### モックモード（デフォルト）
- GPU不要
- パイプライン全体のテスト
- 開発・デバッグ用

### 実際のモード（`--real-model`）
- GPU必要
- 実際のEasyEditモデルを使用
- 本格的な実験用

## 開発ワークフロー

1. **セットアップ**: 環境構築とGPUサポート
2. **実装**: `src/utils/easyedit_wrapper.py`を使用してEasyEdit機能にアクセス
3. **テスト**: `run_ckn_experiment.py`を使用して各条件を独立して検証
4. **評価**: 全条件での包括的実験実行
5. **分析**: 可視化と統計比較の生成

## 今後の研究拡張

- **追加KE手法**: より多くの編集手法の比較
- **モデル多様性**: LLaMA-3、Mistralでの実験  
- **高度な分析**: 隠れ状態追跡、編集順序効果
- **明示的vs暗示的**: 編集アプローチの比較分析

## ライセンス

MIT License

## 貢献

研究目的のリポジトリとして、実験の厳密性を重視しています。EasyEditのベースアーキテクチャとの互換性を維持しながら機能を拡張し、再現可能な実験と包括的なログに重点を置いています。
