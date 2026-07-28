# 指标口径冻结（2026-07-26）

实验阶段结束。此后只重算统计量 / 填表 / 写稿，**不再加模型或生成器**。

## 定位（更新）
受控轻量比较研究；**MobileMamba-lite = 推荐部署 operating point**。  
LiteFreqNet v2 = 频域机制/消融角色，**不是** JPEG 主方法。

## 指标定义（正文必须区分）

| 名称 | 定义 | 用途 |
|------|------|------|
| **ID AUC** | `test.jsonl` 合并 AUC | 主表 |
| **OOD Pooled AUC** | `test_ood.jsonl` 全部样本合并 AUC | 补充（≠ UFD macro） |
| **UFD Macro AUC** | UFD 各子集（real+fake 同 source）AUC **等权平均** | **主泛化指标** |
| **Worst-Generator AUC** | UFD 子集 AUC 的最小值 | 风险/失败边界 |
| **Domain AUC** | ID 内 CelebA-HQ / Bedroom 成对 AUC | 分域表 |
| **JPEG ΔAUC@Q70** | Q70 recompress 后 ID AUC − Clean | 鲁棒表 |
| **FLUX AUC** | 100 real + 100 FLUX.1-schnell | 探索性 §4.7 |

> 旧 bake-off `ood≈0.722`（MambaPSA）= **OOD Pooled**（`test_ood`），不是 UFD Macro（0.700）。二者不可混写。

## 主叙事（允许）
1. MobileMamba-lite：ID / UFD macro / Bedroom / FLUX / 效率上的综合推荐点  
2. SSM 改善多个分布上的轻量表现；**生成器特定失效仍在**（DALL·E≈chance）  
3. ID 增益主要来自困难 Bedroom，非近饱和的 CelebA-HQ  
4. Q70 下全体轻量模型 `|ΔAUC|≤0.003`；频域无独占 JPEG 优势  
5. 频域增强不是普适提升  

## 禁止
SOTA / 通杀 / SSM 通用泛化 / LiteFreq JPEG 主贡献 / 正文提 SD3.5 失败 / Deepfake 视频

## 标题
Compact AI-Generated Image Detection under Compression and Cross-Generator Shift: A Reproducible Study of CNN, Frequency-Aware, and State-Space Models
