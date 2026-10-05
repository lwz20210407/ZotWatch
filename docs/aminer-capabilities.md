# AMiner 能力接入边界

按用户 2026-10-01 的明确要求，**不接入专利**。API、MCP、Skills 是不同层：本机注册工具或
安装技能，不表示 ZotWatch 后台已经内置同一研究工作流。

| 能力 | ZotWatch 实现 | 验证与边界 |
|---|---|---|
| 限免论文搜索、推荐、批量信息 | 来源调度、元数据规范化、缓存、预算、影子对照、可选方法补漏 | 已做真实接口测试；完整摘要与日期不保证存在；生成概述不能充当摘要 |
| 限免学者搜索 | 多候选复核、显式确认、稳定身份登记、rec5 学者线索、可选 OpenAlex 映射追踪、反馈诊断 | 已查询真实学者并识别近名误配；推荐不证明作者署名，缺作者 ID 时不猜测 |
| 限免机构搜索 | 名称候选、确认登记、元数据 ID 匹配、可选已核实 OpenAlex I-ID 定向抓取 | 没有映射时只识别已有候选；不冒充免费获取 AMiner 全部机构产出 |
| 限免期刊搜索 | 确认登记、AMiner venue ID / OpenAlex S-ID / ISSN 匹配、有限抓取与反馈诊断 | 已用真实 DOI 核对 IJIE 的 venue ID、S-ID 与 ISSN |
| 本地专题与引文线索 | 统一专题流程、原文位置、关键词比较、数字引用核验、私有归档与显式公开目录 | 已测试；不是完整实验提取或自动完成的贡献继承/支持/反驳分析 |
| 高级检索、有限多跳引文扩展、学者/机构/期刊资料 | 已有按需命令与页面，显式收费开关、预算及缓存重放 | 19 个非专利高级 API 入口；15 个做过真实请求，未在每周任务自动启用，图谱边不证明引用意图 |
| 平台结构化实验记录检索 | 已接入，保留多方法及数据集原始结构 | 只检索平台已有记录，不等同于任意全文的实验提取；原文真实性仍需核验 |
| OCR、实验数值自动提取、PDF 引用真实性/忠实性核查 | 尚未接入项目 | 部分工作流依赖额外服务、模型或全文证据，需分别设计和验证 |
| 专利 | 明确排除 | 不新增专利白名单、任务或推送 |

工程测试通过验证的是代码行为；真实检索计数、同名消歧样例和人工相关性评价应分开报告。
实体追踪的实际启用条目由 `config/entity-tracking.json` 决定，项目不会自动把查询首项登记进去。
API 免费/限免状态以官方当时说明为准，客户端保持明确白名单，不自动升级收费端点。

接口依据：[官方免费接口目录](https://github.com/AMinerOrg/aminer-open-skill/blob/main/skills/aminer-free-academic/references/api-catalog.md)，
推荐参数依据：[官方 rec5 实现](https://github.com/AMinerOrg/aminer-open-skill/blob/main/skills/aminer-daily-paper/scripts/rec5_api.py)。

## 与官方文档的差异（2026-10-05 对照 aminer.cn/open/docs 核实）

- **rec5 路径**：官方文档写 `POST /api/paper/rec5`，本仓库和官方 aminer-daily-paper 技能都用 `POST /api/v3/paper/rec5`。目前可用；若某周 rec5 全部失败，先查是否旧路径下线。
- **实验记录检索**：`/api/v3/paper/search/experiment_data/SearchPro` 不在官方 32 项接口列表里，单价 0.10 元取自技能目录。只在按需命令里用，不进周报。
- **参数类型**（已修正 `src/aminer_advanced.py`）：`paper_qa_search` 的 `year` 是数组（`[]number`），`author_id`、`org_id` 是字符串数组；`paper_keywords` 的 `size` 上限 10。此前按文档写法传参会被本地校验拒绝。
- **MCP 传输**：官方推荐 Streamable HTTP（`https://mcp.aminer.cn/mcp`），SSE（`/sse`）为兼容保留。MCP 只用于本机 Claude/Codex 会话，GitHub Actions 不经过 MCP。
