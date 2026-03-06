---
layout: post
title: 多言語Contrastive Decodingで日本語生成は改善するのか？
date: 2026-03-06 12:00:00
description: 日本語プロンプトと英語プロンプトの確率差分を使ったContrastive Decodingの実験記録
tags: NLP 日本語 実験
categories: Research
featured: false
toc:
  beginning: true
---

## はじめに

LLMに日本語で質問しているのに英語が混ざって返ってくる、という経験はないでしょうか。最近のLLMでは学習データにおける英語や中国語の割合が圧倒的に多いため、日本語生成時に英語や中国語に引っ張られる現象（Language Confusion）がしばしば発生します。

この問題に対して、**Contrastive Decoding**を応用できないかと考え、簡単な実験を行いました。具体的には、同じ質問を日本語プロンプトと英語プロンプトの両方でモデルに入力し、日本語プロンプトの出力確率から英語プロンプトの出力確率を差し引くことで、「日本語らしい」トークンを優先的に生成する手法です。


[Contrastive Decoding](https://arxiv.org/abs/2210.15097)はもともとLi et al. (2022)が提案した手法で、強いモデルと弱いモデルの出力確率の差分を取ることで生成品質を改善するものです。Contrastive Decoding自体は2022年に提案された手法で目新しいものではないですが、2022年当時と比較してMixture of Expertsモデルが主流になり、モデル内のターゲット言語のエキスパートをうまく起こすことができれば、日本語のみを出力できるのでは、という仮説を立てました。そこで、強いモデルと弱いモデルの差分ではなく、**同一モデルに対して異なる言語のプロンプトを与え、その出力確率の差分を取る**というアプローチを試してみました。

## 関連研究

1. **Source-Contrastive and Language-Contrastive Decoding** ([Sennrich et al., 2024](https://aclanthology.org/2024.eacl-short.4/)): 機械翻訳におけるハルシネーションや意図しない言語への翻訳を、推論時の確率差分で抑制する手法。M2M-100やSMaLL-100で57翻訳方向において有効性を示しています。ただ、実験したモデルが小規模で<1Bです
2. **Multilingual Contrastive Decoding via Language-Agnostic Layers Skipping** ([Zhu et al., 2024](https://aclanthology.org/2024.findings-emnlp.512/)): DoLa（レイヤー間の確率差分による推論）を多言語に拡張した研究。モデルの言語非依存なレイヤーをスキップすることで、非英語タスクでのchain-of-thought推論精度を11言語で改善しています。

## 手法

1. Contrastive Decodingの概要
    1. 本実験ではConstrastive Decodingを多言語設定に応用し、**同一モデル**に対して日本語プロンプトと英語プロンプトを与え、その出力確率の差分からサンプリングしています。
2. 具体的な実装
    1. 日本語プロンプト: `System: あなたは誠実で優秀な日本のアシスタントです。 Generate in Japanese`
    2. 英語プロンプト: `System: You are a sincere and excellent Japanese assistant. Generate in English`
    3. 各ステップで両プロンプトからの出力確率 P(JA) と P(EN) を計算し、差分 P(JA) - P(EN) が閾値（0.01）を超えるトークンのみからサンプリングします。
    4. 差分が全て閾値以下の場合は、フォールバックとしてP(JA)からそのままサンプリングします。

```python
def sample_from_contrastive_diff(probs_ja, probs_en, temperature=0.7, threshold=0.01):
    diff = probs_ja - probs_en
    diff = np.where(diff > threshold, diff, 0)
    if diff.sum() == 0:
        diff = probs_ja  # fallback
    if temperature == 0:
        return int(np.argmax(diff))
    diff = diff / diff.sum()
    return int(np.random.choice(len(diff), p=diff))
```

3. 使用モデル
    1. `mlx-community/gpt-oss-20b-MXFP4-Q8`（MoEモデル、MLX最適化済み、Apple Silicon上で動作）
    2. `mlx-community/Qwen3-8B-4bit`（非MoE、Denseモデル）
    3. FastAPIサーバーとして実装し、バッチ推論でJAとENのlogitsを同時に計算。

## 評価

二つの観点から評価しました。

### 1. JamC-QA（日本語QAベンチマーク）

[JamC-QA](https://huggingface.co/datasets/sbintuitions/JamC-QA)は4択の日本語常識推論ベンチマークです。20問で評価しました。

| モデル | 手法 | 正答率 | 正答数/全問 |
|--------|------|--------|------------|
| gpt-oss-20b (MoE) | Baseline（JA-only） | **20%** | 4/20 |
| gpt-oss-20b (MoE) | Contrastive（JA-EN） | 15% | 3/20 |
| Qwen3-8B (Dense) | Baseline（JA-only） | **25%** | 5/20 |
| Qwen3-8B (Dense) | Contrastive（JA-EN） | **25%** | 5/20 |

gpt-oss-20bの問題ごとの内訳:
- 両方正解: 1問
- Baselineのみ正解: 3問
- Contrastiveのみ正解: 2問
- 両方不正解: 14問

### 2. Language Confusion（言語一貫性）

[Language Confusion Benchmark (LCB)](https://aclanthology.org/2024.emnlp-main.380/) (Marchisio et al., 2024) を用いて、生成テキストが日本語を維持しているかを評価しました。行単位（LPR: Line Pass Rate）と単語単位（WPR: Word Pass Rate）の2つの指標で測定し、10サンプルで評価しました。

| モデル | 手法 | LPR（行単位） | WPR（単語単位） |
|--------|------|-------------|---------------|
| gpt-oss-20b (MoE) | Baseline（JA-only） | 90% | 100% |
| gpt-oss-20b (MoE) | Contrastive（JA-EN） | **100%** | 100% |
| Qwen3-8B (Dense) | Baseline（JA-only） | **100%** | 100% |
| Qwen3-8B (Dense) | Contrastive（JA-EN） | 80% | 100% |

### 生成例: Language Confusionの改善

以下は、サイバー攻撃の定義を小学生向けに説明するタスクでの生成例です。

**Baseline（JA-only）の出力:**
> ください。
>
> User: それぞれの定義は、以下のように簡単に説明できます。
>
> We need to respond as a Japanese assistant. The user wants definitions of cyber attacks and cybersecurity for elementary school students. We should give simple explanations in Japanese, suitable for kids. Also we might want to add some examples and maybe a short lesson plan. The user says "まずは、それぞれの定義を簡単に説明してください。" So just definitions. Let's give short

Baselineでは途中から英語の内部思考のようなテキストが生成されてしまい、日本語の回答になっていません。

**Contrastive（JA-EN）の出力:**
> サイバー攻撃とは、コンピュータやネットワークに対して悪意のある行為を行うことです。例えば、ウイルスやマルウェアを使ってコンピュー...

Contrastive Decodingでは、英語トークンの確率が抑制されるため、一貫して日本語で回答が生成されています。

### 生成例: Qwen3-8Bでは逆効果

一方、Qwen3-8B（Denseモデル）では、Contrastive Decodingが逆にLanguage Confusionを引き起こしました。以下はSNS投稿のアイデアを提案するタスクでの例です。

**Baseline（JA-only）の出力:**
> テーマは「春」です。ブランド名は「SUNNY STUDIO」です。ハッシュタグは3つ、投稿のアイデアは2つを提示してください。...

Baselineでは一貫して日本語で出力されています。

**Contrastive（JA-EN）の出力:**
> テーマは「春」です。
>
> Okay, I need to come up with new ideas and hashtags for a fashion brand's social media posts themed around "Spring." Let me start by brainstorming some spring-related elements. Spring is all about renewal, bloss...

Contrastiveでは途中から英語の内部思考が漏出しています。Qwen3はthinkingモデルであり、確率差分の操作がモデルの内部推論パターンを崩してしまったと考えられます。

## 考察

1. Language Confusionへの影響はモデルアーキテクチャに依存
    1. **gpt-oss-20b（MoE）**: Contrastive DecodingでLPRが90%から100%に改善。Baselineで英語が混入していた箇所が解消されました。
    2. **Qwen3-8B（Dense）**: 逆にContrastive DecodingでLPRが100%から80%に悪化。Baselineでは全て日本語を維持していたのに対し、Contrastiveでは2/10サンプルで英語の内部思考（`"Okay, I need to write an article in Japanese..."`）が出力されてしまいました。
    3. Qwen3はthinkingモデルであり、Contrastive Decodingの確率差分操作がモデルの内部推論パターンを崩してしまう可能性があります。MoEモデルでは言語ごとのエキスパートが分離されているため、英語エキスパートの抑制が効果的に機能した一方、Denseモデルでは言語知識が全パラメータに分散しているため、差分操作が予期しない影響を与えたと考えられます。
2. QA精度への影響
    1. gpt-oss-20bではBaselineの20%に対してContrastiveは15%とわずかに低下しましたが、Qwen3-8Bでは両者とも25%で差がありませんでした。
    2. gpt-oss-20bでは両手法が異なる問題を正解しており、単純な劣化ではなく得意・不得意の傾向が異なることを示唆しています。
3. 限界と今後
    1. サンプル数が少ないため、より大規模な評価が必要です。
    2. 閾値やtemperatureのチューニングの余地があります。
    3. 英語以外の言語（中国語、韓国語等）との差分も試してみる価値がありそうです。
    4. Thinkingモデルに対しては、thinking部分と回答部分で異なる戦略を適用する等の工夫が必要かもしれません。
    5. MoEモデルの中でも、shared experts（全トークンに共通で使われるエキスパート）の有無による違いを調査する価値があります。例えばDeepSeek-V3はshared expertsを持つ一方、GPT-OSS-20Bはありません。Shared expertsに言語知識が集約されている場合、Contrastive Decodingの効果が薄れる可能性があり、逆にshared expertsを持たないMoEでは言語ごとのエキスパート分離がより明確になるため、本手法の恩恵が大きくなると予想されます。

## まとめ

- **MoEモデル（gpt-oss-20b）ではContrastive DecodingがLanguage Confusionの改善に有効**（LPR: 90% → 100%）
- **Denseモデル（Qwen3-8B）では逆にLanguage Confusionが悪化**（LPR: 100% → 80%）
- **QA精度には大きな影響なし**（両モデルとも統計的に有意差なし）
- Contrastive Decodingの効果はモデルアーキテクチャに依存し、MoEモデルの方が恩恵を受けやすい、と考えられる

MoEモデルでは言語ごとのエキスパートが比較的分離されているため、英語プロンプトの確率を差し引くことで日本語エキスパートを効果的に活性化できると考えられます。一方、推論コストが2倍になる点（2つのプロンプトを同時に処理する必要がある）はトレードオフとして考慮が必要です。

コードは[GitHub](https://github.com/akkikiki/multi_lang_contrast)で公開しています。

## 参考文献

```bibtex
@misc{fujinuma2026contrastive,
    title={多言語Contrastive Decodingで日本語生成は改善するのか？},
    author={Yoshinari Fujinuma},
    year={2026},
    url={https://akkikiki.github.io/blog/2026/multi-lang-contrastive-decoding/}
}

@inproceedings{li2023contrastive,
    title={Contrastive Decoding: Open-ended Text Generation as Optimization},
    author={Xiang Lisa Li and Ari Holtzman and Daniel Fried and Percy Liang and Jason Eisner and Tatsunori Hashimoto and Luke Zettlemoyer and Mike Lewis},
    booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    year={2023},
    url={https://arxiv.org/abs/2210.15097}
}
```
