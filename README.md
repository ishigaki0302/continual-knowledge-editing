# 継続知識編集 (CKE) フレームワーク

このリポジトリは、EasyEditフレームワークをベースとした大規模言語モデルにおける**継続知識編集（CKE）**の実験コードを実装しています。複数の実験条件を用いて、逐次的な知識編集操作の複雑さと効果を評価することに焦点を当てています。

## 研究概要

### 主要な革新
- 知識挿入・修正の精密なセマンティック制御のための**「共有」**および**「排他」**関係タイプの導入
- LLMにおける継続知識編集のための包括的評価フレームワーク
- 知識保持と干渉パターンを分析するマルチ条件実験設計
- 暗黙的vs明示的編集やエンティティ類似度効果を含むIJCNLP2025拡張分析

### 関係タイプ
- **共有関係**：（主語、関係）ペアに対して複数のオブジェクトを許可（蓄積的セマンティクス）
  - 例：Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces
- **排他関係**：（主語、関係）ペアに対して1つのオブジェクトのみを許可（上書きセマンティクス）
  - 例：HealthStatus, Job, Residence, CurrentLocation, AgeGroup

### 評価条件
1. **条件A**：異なる主語での逐次編集
2. **条件B**：同一主語での複数関係編集
3. **条件C**：同一（主語、関係）ペアでのオブジェクト再編集
   - 共有関係：蓄積的動作
   - 排他関係：上書き動作

## クイックスタート

### 基本的な知識編集（GPU必要）
```bash
# 条件A：異なる主語
python3 run_knowledge_editing.py --method ROME --model gpt-j-6b --condition A --num-edits 5

# 条件B：同一主語、異なる関係
python3 run_knowledge_editing.py --method ROME --model gpt-j-6b --condition B --num-edits 5

# 条件C：同一主語-関係、異なるオブジェクト
python3 run_knowledge_editing.py --method ROME --model gpt-j-6b --condition C --num-edits 5
```

### IJCNLP2025拡張分析デモ
```bash
# 高度な機能のデモ（GPU不要）
python3 demo_ijcnlp.py
```

## 環境セットアップ

### Dockerセットアップ（推奨）
```bash
docker build -t cke-framework:latest .
docker run -it --ipc=host -p 8501:8501 --gpus all -v $(pwd):/app/CKE --name cke-container cke-framework:latest
```

### ローカルセットアップ
```bash
conda create -n CKE python=3.9.7
conda activate CKE
pip install -r requirements.txt
```

## 対応モデル・手法

### 知識編集手法
- **ROME**：Rank-One Model Editing ✅
- **MEMIT**：Mass Editing Memory in a Transformer ✅
- **MEND**：Model Editor Networks using Gradient Decomposition ✅
- **FT**：Fine-Tuning ✅
- **IKE**：In-Context Knowledge Editing ✅
- **KN**：Knowledge Neurons ✅
- **SERAC**：現在のバージョンではサポートされていません

### 対応モデル
- **GPT-J-6B**、**GPT-2-XL**（主要テストモデル）
- LLaMA (7B, 3.2-3B), LLaMA-3 (8B)
- Qwen (7B, 2.5-7B), ChatGLM (2-6B, 4-9B)
- Mistral-7B, Baichuan-7B, InternLM-7B

## プロジェクト構成

```
├── datasets/
│   └── temp_ckndata.json            # 5つの主語と関係を含む実験データ
├── easyedit_base/                   # EasyEditフレームワーク統合
│   ├── easyeditor/                  # EasyEditコアモジュール
│   ├── hparams/                     # 手法設定（ROME, MEMIT等）
│   └── examples/                    # 参考実装
├── src/
│   ├── experiments/                 # 実験実装
│   │   ├── data_sampler.py          # 条件A/B/Cのデータサンプリング
│   │   └── ijcnlp_extensions.py     # IJCNLP2025拡張分析
│   └── utils/                       # ユーティリティ関数
│       ├── easyedit_wrapper.py      # EasyEdit統合ラッパー
│       └── mock_llm.py              # テスト用モックLLM
├── results/                         # 実験結果（JSON形式）
├── run_knowledge_editing.py         # メイン実験実行スクリプト
├── plot_knowledge_editing.py        # 結果可視化
├── demo_ijcnlp.py                  # IJCNLP2025機能デモ
└── requirements.txt                 # Python依存関係
```

## 実験フレームワーク

### データセット（`datasets/temp_ckndata.json`）
フレームワークは実験用に5つの架空の主語を使用：
- **Ryoma Ishigaki**
- **Jundai Suzuki**
- **Shun Iwase**
- **Reiya Hiramoto**
- **Masato Sekiguchi**

### 関係カテゴリ
- **共有関係**（蓄積的）：Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces
- **排他関係**（上書き）：HealthStatus, Job, Residence, CurrentLocation, AgeGroup

各関係は5つの可能なオブジェクトと、体系的評価のための定義されたプロンプト/質問テンプレートを持ちます。

### 主要実装ファイル

#### `run_knowledge_editing.py`
メイン実験スクリプトで以下を実行：
- 異なる条件（A/B/C）を使用した知識三つ組みの抽出
- EasyEditによる逐次知識編集の実行
- 確率分析による効果測定の計算
- 詳細結果のJSON形式での保存

#### `src/experiments/data_sampler.py`
データサンプリングロジックで以下を実装：
- **条件A**：編集ごとに異なる主語
- **条件B**：同一主語、異なる関係
- **条件C**：同一主語-関係、異なるオブジェクト（共有/排他バリアント）
- 選択肢形式での評価プロンプト生成

#### `src/utils/easyedit_wrapper.py`
EasyEdit統合で以下を提供：
- 全編集手法（ROME, MEMIT, MEND等）の統一インターフェース
- モデル名マッピングとハイパーパラメータ読み込み
- モデル状態保持による逐次編集

## 評価指標

### 主要指標
- **効果性**：知識編集の成功率（対象オブジェクトが最高確率を持つ）
- **編集後分析**：各編集後の確率分布
- **最終状態分析**：全編集の累積効果
- **ランキング分析**：対象オブジェクトの順位変化

### 結果可視化
`plot_knowledge_editing.py`を使用して以下を生成：
- 編集後vs最終状態の確率比較
- 全条件・編集ステップを示す6×Nグリッド
- 対象オブジェクトのハイライトと確率追跡

```bash
# 結果から可視化を生成
python3 plot_knowledge_editing.py \
  --fileA results/knowledge_editing_ROME_gpt-j-6b_condition_A_*.json \
  --fileB results/knowledge_editing_ROME_gpt-j-6b_condition_B_*.json \
  --fileC results/knowledge_editing_ROME_gpt-j-6b_condition_C_*.json \
  --out summary_plot.png
```

## IJCNLP2025拡張機能

`demo_ijcnlp.py`スクリプトは高度な分析機能をデモ：
- **暗黙的vs明示的編集**：編集アプローチの比較
- **エンティティ類似度分析**：類似vs非類似エンティティの効果
- **編集順序バリエーション**：編集シーケンスの結果への影響
- **確率分布変化**：エントロピーとランキング安定性分析
- **Hidden States分析**：層ごとの活性化変化追跡
- **セットカバレッジ評価**：蓄積vs上書き動作分析

## ハードウェア要件
- **GPU必要**：8GB以上のVRAMを持つNVIDIA GPU（RTX A6000推奨）
- **CPU代替**：デモ/モックモードのみに限定
- **メモリ**：完全な実験には16GB以上のRAM推奨

## ベースフレームワーク
[EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main)をベースに構築 - 大規模言語モデルのための使いやすい知識編集フレームワーク

## 依存関係
主要な依存関係（完全なリストは`requirements.txt`を参照）：
- `transformers==4.46.2`, `torch==2.0.1`, `peft==0.7.1`
- `sentence-transformers==3.2.1`, `einops==0.4.0`
- `matplotlib==3.5.1`, `scikit-learn==1.0.2`