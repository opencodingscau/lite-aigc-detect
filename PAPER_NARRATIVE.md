# Paper narrative (locked)

**受控轻量比较研究**；**LiteSSM-A** 为推荐部署 operating point（非最低绝对延迟）。

正式命名：

| 冻结实验内部名称 | 论文正式名称 |
|------------------|--------------|
| `mobilemamba_lite` | LiteSSM-A |
| `mambapsa_cls` | LiteSSM-B |

数字唯一源：`freeze/frozen_numbers.json`

- LiteSSM-A：ID 0.946；UFD Macro **0.718**；B1 p50 **144.4 ms**；B32 **226** img/s  
- LiteSSM-B：UFD Macro 0.700；OOD Pooled **0.722**（不得写成 Macro）  
- 旧 **367** FPS：已废止，不进正文  

叙事要点：

1. LiteSSM-A：ID、UFD Macro、Bedroom 综合推荐点（226 thr@32）  
2. LiteSSM-B 竞争；与 LiteSSM-A 部分差异未必显著；OOD Pooled 更高但 Macro 更低  
3. FLUX：仅附录；不作架构结论  
4. 不写 “faster than MobileNet”
