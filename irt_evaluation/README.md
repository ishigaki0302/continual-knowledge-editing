# 継続的知識編集 評価フレームワーク

## 概要
継続的知識編集の性能を統計的に評価するためのフレームワークです。即時応答と累積応答の両方を分析し、編集手法の特性を明らかにします。

## ファイル構成
```
irt_evaluation/
├── README.md                     # このファイル
├── 評価フレームワーク説明.md        # 詳細な評価説明（日本語）
├── simple_analysis.py           # メイン評価スクリプト
├── run_analysis.py              # 高度な分析オプション付きスクリプト
├── data_processor.py            # データ処理モジュール
├── config.yaml                  # 設定ファイル
├── requirements_extended.txt    # 必要なライブラリ
└── output/                      # 分析結果出力先
    ├── analysis_data.csv        # 分析用データ
    ├── summary_statistics.json  # 統計サマリー
    ├── reports/
    │   └── simple_analysis_report.html  # 分析レポート
    └── figures/                 # 可視化結果
        ├── success_rates_by_condition.png    # 条件別成功率
        ├── success_rates_by_method.png       # 手法別成功率
        ├── probability_distributions.png     # 確率分布
        ├── probability_changes.png           # 確率変化
        └── performance_by_edit_order.png     # 編集順序効果
```

## 使用方法

### 基本実行
```bash
cd /app/EasyEdit/irt_evaluation
python simple_analysis.py
```

### 結果確認
- **レポート**: `output/reports/simple_analysis_report.html`
- **可視化**: `output/figures/` 内の PNG ファイル
- **データ**: `output/analysis_data.csv` と `output/summary_statistics.json`

### カスタム実行
```bash
# カスタム設定での実行
python run_analysis.py --results-dir /custom/path --output-dir /custom/output

# ログレベル変更
python run_analysis.py --log-level DEBUG
```

## 生成される評価結果

### 1. 成功率分析
- **条件別成功率**: 編集条件A, B, Cでの即時・累積成功率比較
- **手法別成功率**: ROME, MEMITなど各手法の性能比較

### 2. 確率分析
- **確率分布**: 即時・累積応答の確率分布
- **確率変化**: 編集による確率変化パターン

### 3. 編集順序効果
- **順序別性能**: 編集回数による成功率の変化

### 4. 統計サマリー
- 全体統計、手法比較、条件比較の数値データ

## 評価指標
- **即時成功率**: 編集直後の正解率
- **累積成功率**: 全編集終了後の正解率  
- **性能劣化度**: 即時から累積への成功率低下
- **確率変化**: 編集による確率分布の変動

詳細な評価手法については `評価フレームワーク説明.md` を参照してください。