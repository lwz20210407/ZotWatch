# 首批追踪映射的依据

2026-10-01 首批仅登记当前研究链已有的学者 Dirk Mohr、期刊 IJIE，及这位学者的 ETH Zurich
机构关联。机构扩展仍受力学主题门、原评分和去重限制，不把该机构全部跨学科产出推送进周报。

核对种子为现有 `authors.yaml` 中的 DOI
[10.1016/j.ijimpeng.2019.05.008](https://doi.org/10.1016/j.ijimpeng.2019.05.008)，题名为
“Dynamic perforation of ultra-hard high-strength armor steel: Impact experiments and modeling”。

| 对象 | AMiner ID | 已核实对应 | 依据 |
|---|---|---|---|
| Dirk Mohr | `611a3ad89e795e8d33c67994` | OpenAlex `A5110517133` | 已有追踪身份；AMiner 候选为 ETH 力学学者，兴趣包含塑性/延性断裂/SHPB；种子论文的 OpenAlex 作者与机构记录一致 |
| IJIE | `5eade71cedb6e7d53c077e0f` | OpenAlex `S9070569`、ISSN `0734-743X` | 精确 DOI 对应 AMiner 论文 `5d0b000f8607575390fa7206` 的免费信息返回该 venue_id；OpenAlex 返回相同刊名与 ISSN |
| ETH Zurich | `62331e350a6eb147dca8a805` | OpenAlex `I35440088`、ROR `05a28rw58` | AMiner 机构规范名查询与上述种子论文的 ETH 作者单位交叉核对 |

[OpenAlex 种子元数据](https://api.openalex.org/works/https://doi.org/10.1016/j.ijimpeng.2019.05.008)
提供作者 ID、ORCID `0000-0003-2810-1893`、单位 ID/ROR、期刊 ID/ISSN。
[AMiner 种子页](https://www.aminer.cn/pub/5d0b000f8607575390fa7206) 对应论文搜索和信息补全实测。

没有把 Michael Brünig 的近名查询首项登记进去：实测返回无线传感网等其他方向的近名者，
机构过滤后仍未命中。AMiner 学者结果中的 ETH 机构 ID 与机构搜索的规范 ID 不同，也没有
据名称把两种 ID 自动合并为别名。

隔离试验验证了这三种身份在种子论文上的匹配、两轮有限 OpenAlex 抓取（29 篇去重候选，3 篇
通过主题筛选）及停用/重新启用。学者 rec5 返回的 3 篇推荐缺少作者 ID，未标记为该学者署名。
这验证追踪流程，不代表自动推荐的相关性已经人工确认。

请求仍使用既有预算：每轮最多两个实体、一页 20 条；AMiner 学者推荐仍处于 shadow 模式，
共享原请求上限。可以分别用 `entity_tracking disable` 停用每条映射。
