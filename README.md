# 継続知識編集 (CKE) フレームワーク with IJCNLP2025拡張

EasyEditフレームワークをベースとした、大規模言語モデルにおける継続知識編集の包括的評価フレームワークです。IJCNLP2025向けの高度な分析機能を搭載しています。

## 概要

このリポジトリは、LLMにおける継続知識編集（CKE）の包括的評価フレームワークを実装し、逐次的な知識編集操作の複雑さと効果に焦点を当てています。2025年6月に大幅な機能拡張を行い、IJCNLP2025向けの高度な研究機能を追加しました。

## 主要機能

### 基本機能
- **共有vs排他的Relation**: 知識挿入・修正の精密なセマンティック制御
- **マルチ条件評価**: 3つの異なる評価シナリオ (A, B, C)
- **逐次編集パイプライン**: EasyEdit統合による包括的な編集手法サポート
- **包括的分析**: 確率ランキングと干渉パターンの検出
- **モック実行**: GPU不要での完全なパイプライン検証

### IJCNLP2025拡張機能 🆕
- **暗黙的vs明示的編集**: 編集アプローチの効果比較分析
- **エンティティ類似度分析**: 類似・非類似エンティティでの干渉パターン研究
- **編集順序効果**: 編集順序が結果に与える影響の体系的調査
- **確率分布変化分析**: 編集による予測確率の変化とランキング追跡
- **Hidden States分析**: モデル内部状態の層別変化解析
- **セットカバレッジ評価**: 共有/排他関係での知識保持度合いの定量評価

## クイックスタート

### 基本的な使用方法（GPU不要）
```bash
# リポジトリのクローン
git clone https://github.com/ishigaki0302/continual-knowledge-editing.git
cd continual-knowledge-editing

# 基本依存関係のインストール
pip install numpy matplotlib

# IJCNLP2025拡張機能のデモ実行 🆕
python3 demo_ijcnlp.py

# 基本CKE実験の実行
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5

# IJCNLP2025包括実験の実行 🆕
python3 run_ijcnlp_experiment.py --method ROME --num-edits 5
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
│   ├── experiments/           # 実験実装
│   │   ├── data_sampler.py    # データサンプリング（条件A、B、C）
│   │   └── ijcnlp_extensions.py # IJCNLP2025拡張機能 🆕
│   └── utils/                 # ユーティリティ
│       ├── easyedit_wrapper.py # EasyEdit統合ラッパー
│       └── mock_llm.py        # モックLLM（テスト用）
├── easyedit_base/             # EasyEdit統合
│   ├── easyeditor/           # EasyEditコアモジュール
│   ├── hparams/              # 全手法のハイパーパラメータ設定
│   └── examples/             # 参考実装例
├── datasets/                  # 評価データセット
│   └── temp_ckndata.json     # 実験用データ
├── results/                   # 実験結果 🆕
│   └── ijcnlp_experiment_*.json # IJCNLP実験結果
├── run_ckn_experiment.py     # 基本CKE実験実行スクリプト
├── run_ijcnlp_experiment.py  # IJCNLP2025実験実行スクリプト 🆕
└── demo_ijcnlp.py            # IJCNLP2025デモスクリプト 🆕
```

## 実験フレームワーク

### データ構成
`temp_ckndata.json`を使用した実験データ：
- **5つのSubject**: Aさん、Bさん、Cさん、Dさん、Eさん
- **共有Relation**: Skills, Hobbies, LearnedLanguages, ReadBooks, VisitedPlaces（蓄積的）
- **排他Relation**: Health Status, Job, Residence, CurrentLocation, AgeGroup（上書き）

### 実験条件

#### 条件A: 異なるSubjectでの逐次編集
- 各編集で異なるSubjectを使用
- Subject間の干渉パターンを評価

#### 条件B: 同一Subjectでの複数Relation編集  
- 同じSubjectに対して異なるRelationを編集
- Relation間の相互作用を分析

#### 条件C: 同一(Subject、Relation)での再編集
- **共有Relation**: 蓄積的動作（新しい知識が追加される）
- **排他Relation**: 上書き動作（古い知識が置き換えられる）

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
# IJCNLP2025拡張機能のデモ 🆕
python3 demo_ijcnlp.py

# 基本CKE機能のテスト
python3 run_ckn_experiment.py --method ROME --num-edits 3
```

## 実験スクリプトの説明

### 🧪 `run_ckn_experiment.py` - 基本CKE実験実行スクリプト
**継続的知識編集の基本実験**を実行します。条件A、B、Cの完全な評価を提供します。

**主な機能：**
- 条件A、B、Cの全実験を自動実行
- コマンドライン引数で手法・モデル・編集数を指定可能
- モック/実機両対応（`--real-model`フラグで切り替え）
- JSON形式での詳細結果出力
- 包括的な評価指標の算出

**使用例：**
```bash
# 基本実行（モック、5編集）
python3 run_ckn_experiment.py --method ROME --model gpt-j-6b --num-edits 5

# 実機実行（GPU必要）
python3 run_ckn_experiment.py --method MEMIT --model gpt2-xl --real-model

# カスタム出力
python3 run_ckn_experiment.py --method MEND --output my_results.json
```

### 🚀 `run_ijcnlp_experiment.py` - IJCNLP2025拡張実験スクリプト 🆕
**IJCNLP2025向けの高度な知識編集分析**を実行します。最新の研究機能を提供します。

**主な機能：**
- 暗黙的vs明示的編集の比較分析
- エンティティ類似度に基づく干渉分析
- 編集順序効果の体系的調査
- 確率分布変化とランキング追跡
- Hidden states の層別変化解析
- セットカバレッジ評価

**使用例：**
```bash
# 包括実験（全分析を実行）
python3 run_ijcnlp_experiment.py --method ROME --num-edits 5

# 特定の分析のみ実行
python3 run_ijcnlp_experiment.py --experiment-type implicit-explicit
python3 run_ijcnlp_experiment.py --experiment-type similarity
python3 run_ijcnlp_experiment.py --experiment-type order --real-model

# 異なる手法での比較
python3 run_ijcnlp_experiment.py --method MEMIT --experiment-type comprehensive
```

### 🎮 `demo_ijcnlp.py` - IJCNLP2025拡張デモスクリプト 🆕
**IJCNLP2025拡張機能の理解とテストのためのデモスクリプト**です。全機能を段階的に紹介します。

**主な機能：**
- 暗黙的vs明示的編集のデモ
- エンティティ類似度分析の例示
- 編集順序バリエーションの表示
- 確率分布変化の可視化
- Hidden states分析の実例
- セットカバレッジ評価の説明

**内容：**
- 各拡張機能の具体例表示
- モック実行による安全なテスト
- 研究者向けの詳細な出力


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

## プロジェクト開発履歴

### Phase 1: 基本フレームワーク開発 (2025年6月初期)
- **プロジェクト初期化**: CLAUDE.md作成、基本構造設計
- **EasyEdit統合**: フレームワーククローン、統合、設定
- **CKE基本機能**: 条件A、B、C実装、データサンプリング
- **モック実行**: GPU不要でのパイプライン検証
- **基本実験**: `run_ckn_experiment.py`、`demo_experiment.py`

### Phase 2: IJCNLP2025拡張開発 (2025年6月中期) 🆕
- **高度分析機能**: 暗黙的vs明示的編集比較
- **エンティティ類似度**: 類似度マトリックス、干渉パターン分析
- **編集順序効果**: 順序バリエーション、感度分析
- **確率分布分析**: ランキング変化、安定性評価
- **Hidden States**: 層別変化、局所vs全体効果
- **セットカバレッジ**: 共有/排他関係での知識保持評価

### Phase 3: 統合・最適化 (2025年6月後期) 🆕
- **包括実験システム**: `run_ijcnlp_experiment.py`実装
- **デモシステム**: `demo_ijcnlp.py`による機能紹介
- **バグ修正**: 型エラー、依存関係、JSON保存の問題解決
- **レガシーコード削除**: 古いファイル整理、構成最適化
- **ドキュメント更新**: README全面改訂、機能説明追加

## IJCNLP2025拡張機能詳細

### 1. 暗黙的vs明示的編集 (`IJCNLPExtensions.generate_implicit_explicit_pairs`)
```python
# 暗黙的編集例
"石垣龍馬 is skilled in Python."

# 明示的編集例  
"石垣龍馬はJava、Excelだけでなく、Pythonもスキルとして習得している"
```

### 2. エンティティ類似度分析 (`calculate_entity_similarity`)
- コサイン類似度計算（sklearn使用、フォールバック実装あり）
- 類似・非類似エンティティでの編集シーケンス生成
- 干渉パターンの定量評価

### 3. 編集順序効果 (`generate_order_variations`)
- 同一編集セットの順序バリエーション生成
- Kendall's tau距離による順序変化定量化
- 順序感度の統計的評価

### 4. 確率分布変化分析 (`analyze_probability_distribution_changes`)
- 編集ステップごとの確率ランキング追跡
- エントロピー計算、安定性スコア算出
- ランキング変化の統計分析

### 5. Hidden States分析 (`analyze_hidden_states`)
- 層別の活性化ノルム変化追跡
- 局所的vs全体的効果の定量化
- 層間相関分析

### 6. セットカバレッジ評価 (`analyze_set_coverage`)
- 共有関係：蓄積的知識の網羅性評価
- 排他関係：最新知識の正確性評価
- 精度・再現率・F1スコア計算

## 今後の研究拡張

- **実機評価**: GPU環境での大規模実験
- **追加手法**: AlphaEdit、WISE等の新手法統合
- **多言語対応**: 日本語以外の言語での評価
- **長期記憶**: より多数の編集での長期効果分析
- **マルチモーダル**: 画像-テキスト知識編集への拡張

## ライセンス

MIT License

## 貢献

研究目的のリポジトリとして、実験の厳密性を重視しています。EasyEditのベースアーキテクチャとの互換性を維持しながら機能を拡張し、再現可能な実験と包括的なログに重点を置いています。
