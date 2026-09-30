# 学者、机构、期刊：查询到追踪的闭环

此功能把 AMiner 实体查询产生的待核查候选，连接到显式身份确认、检索、论文归属标记、报告统计
和启停操作。专利不在本项目接入范围内。不会按同名字符串自动合并身份，也不会调用收费的
学者论文、机构产出或期刊论文清单接口。

## 查询并确认

先运行 `aminer_entities`，得到含多个候选的 `entity-review.json`。核对姓名、机构、研究方向和
代表作后，明确选择 ID：

```text
python -m src.entity_tracking confirm --review <entity-review.json> --id <返回的AMiner-ID> --reason "核对依据与代表作 DOI"
```

确认写入 `config/entity-tracking.json`，记录来源审核文件哈希与时间。已有条目不能被同名结果
覆盖；更换映射须明确审阅配置。已存在同一 ID 或互相冲突的跨库映射会被拒绝。

可以同时提供已核实的 `--openalex-id A...`（学者）、`I...`（机构）或 `S...`（期刊）；
期刊还可用重复的 `--issn 0734-743X` 记录经过核验的 ISSN。类型与 ISSN 校验位必须正确。
这些参数是显式确认，不是程序通过名字推导出的事实；应使用代表作、机构、ORCID/ROR/ISSN
等证据核对。将配置正常提交后，GitHub Actions 才能读取。

## 如何检索与标记

| 确认信息 | 使用方式 |
|---|---|
| 学者 AMiner ID | 在原 AMiner 请求/推荐上限内，加入带 `aminer_author_id` 的 rec5 查询并参与轮换。仅表示该学者触发的推荐，不代表论文由其署名。 |
| 学者/机构/期刊的 AMiner ID | 后续论文元数据提供相应作者、机构或期刊 ID 时才标记匹配；只返回名字时保持未确认。 |
| 经核实的 OpenAlex ID | 使用既有 OpenAlex 预算定向读取文献；每轮最多轮换两个实体、每个一页 20 条，再核验返回的实体 ID。 |
| 期刊 ISSN | 可识别 Crossref/OpenAlex 候选中同一刊物；刊名近似本身不构成匹配。 |

缺少跨库映射的机构和期刊只能识别本轮已有候选中的身份匹配，不能声称完整覆盖其全部产出。
所有新增候选仍走原有专业主题筛选、去重、排序、日期和历史规则；实体匹配本身不加分，也不会
让跨学科论文绕过力学主题门。AMiner 处于 shadow 模式时，学者 rec5 候选仍只进入影子对照。

## 反馈与停用

报告卡片显示匹配的实体，运行诊断列出各实体的候选命中数、报告入选数和已有论文反馈。
统计按论文身份去重；`selected_for_report` 不代表发送成功，零命中也不代表没有新作。
反馈沿用既有论文按钮、所有者校验和已读过滤；系统不会因一次差评擅自删除实体。

```text
python -m src.entity_tracking list
python -m src.entity_tracking disable --key person:<AMiner-ID>
python -m src.entity_tracking enable --key person:<AMiner-ID>
```

停用会停止其新增定向查询，并在重新处理候选时清除该实体标记，不删除论文本身或历史。
旧作者配置不被此命令改写：若同一学者还在 `authors.yaml` 独立启用，那条已有追踪仍继续。
配置损坏或单个实体接口失败会报告诊断，原周报继续运行。

## 已知接口边界

2026-10-01 实测，Michael Brünig 的免费姓名查询返回了无线传感网等方向的近名候选，加机构
条件后未命中，不能据此绑定当前力学学者。Dirk Mohr 的查询同时包含力学与生物学同名者，
同样必须结合机构、研究方向和代表作筛选。官方免费信息返回的作者记录可能只有姓名，没有
作者 ID；这种结果只能用于阅读，不能自动确认为作者归属。

依据：[官方免费接口目录](https://github.com/AMinerOrg/aminer-open-skill/blob/main/skills/aminer-free-academic/references/api-catalog.md)、
[官方 rec5 参数实现](https://github.com/AMinerOrg/aminer-open-skill/blob/main/skills/aminer-daily-paper/scripts/rec5_api.py)。
