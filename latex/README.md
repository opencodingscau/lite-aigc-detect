# MDPI Applied Sciences LaTeX 工程说明

## 目录
```
latex/
  main.tex                 # 入口（journal=applsci）
  Definitions/             # 官方 MDPI 类文件（勿改路径）
  figures/
    fig_protocol_overview.png
    fig_ssm_architecture.png
    fig2_pareto_ufd_macro.png
    fig_domain_gap.png
    fig3_generator_heatmap.png
    fig_jpeg_quality_sweep.png
    fig_seed_sensitivity.png
    fig_graphical_abstract.png   # MDPI graphical abstract upload (not inlined)
  sections/
    01_introduction.tex … 06_conclusions.tex
    A_flux_appendix.tex
    B_jpeg_appendix.tex
    C_seed_appendix.tex
  mdpi_official/           # 原始 template 备份
  README.md                # 本文件
```

## 重新生成图
```bash
python scripts/render_paper_figures.py
```
（从仓库根目录运行；数字来自 `freeze/` / `jpeg_results/` / `docs/seed_sweep_summary.json`。）

## 编译
在 `latex/` 目录下：
```bash
pdflatex main
pdflatex main
```
（参考文献为 `thebibliography`，一般无需 bibtex。）

Overleaf：上传整个 `latex/`（含 `Definitions/` 与 `figures/`），主文件设为 `main.tex`。
投稿时可另传 `figures/fig_graphical_abstract.png` 作为 graphical abstract。

## 命名与数字
- 正文模型名：**LiteSSM-A** / **LiteSSM-B**（勿写 MobileMamba / MambaPSA / PSA）
- 唯一数字源：`../freeze/frozen_numbers.json`（LiteSSM-A：B1 p50 **144.4 ms**，B32 **226** img/s）
- 旧 367 img/s 已废止，不得进入 PDF
- FLUX 仅 Appendix；0.722 仅作 OOD Pooled（LiteSSM-B），不得写成 UFD Macro

## 投稿前必改
1. `\Author` / `\address` / `\corres` / ORCID  
2. `\funding`、贡献声明  
3. References 中带 `TODO` 的条目（页码、正式出版信息）  
4. 确认 Abstract 字数（约 200 words）  

## 图表交叉引用
| 标签 | 内容 |
|------|------|
| `tab:ssm-arch` | LiteSSM 结构表 |
| `fig:ssm-arch` | LiteSSM 结构图 |
| `tab:overall` | Table 2 总体/效率 |
| `tab:external` | Panel B 外部参考 |
| `tab:domain` | Table 3 分域 |
| `tab:ufd` | Table 4 UFD 分源 |
| `tab:jpeg` | Table 5 JPEG Q70 |
| `tab:flux-app` | Appendix FLUX |
| `fig:pareto` | Figure 2 Pareto（B1 latency） |
| `fig:heatmap` | Figure 3 Heatmap |

## 与草稿关系
旧 Markdown 草稿已标 SUPERSEDED。权威正文：本目录 `sections/`；数字：`../freeze/frozen_numbers.json`。
