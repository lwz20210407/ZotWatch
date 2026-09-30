# 按需高级检索、实体资料与实验记录

新增的高级入口与每周限免来源分开。**默认不允许联网收费调用**：只有显式 `--allow-paid`
才可请求高级 API；`--cache-only` 连免费接口也不请求，只读取已有响应。每周任务不调用这些入口。
专利仍然完全排除。

## 费用与恢复

单个 API 运行目录包含持久 `spending.json`，预算范围为 ¥0.01–¥5.00。每次网络尝试前按
2026-10-01 官方目录价格预留费用，失败/超时也不退款估算，不自动重试；成功缓存复用不重复
记费。预算耗尽后停下，恢复不会重置预算，不能在同一目录中改大额度。`actual_billed_yuan`
始终为 null：这是保守估算，不是读取了账户账单。

`--api-run-dir` 可以让多项研究操作共享同一缓存与预算；不指定时使用输出目录中的 `api/`。
不要为了绕过限额把一次任务拆成多个预算目录。报告分别显示共享预算累计值和本专题新增估算。
缺凭据、鉴权失败、限流和业务错误不会伪装成正常空结果；错误正文和 Token 不进入日志。

## 高级专题流程

下面会真实调用收费接口，须明确选择后执行：

```text
python -m src.aminer_research --query "How are metal ductile fracture models calibrated under high strain rates?" --mode semantic --pages 1 --depth 1 --seeds-per-depth 2 --details 5 --budget-yuan 1 --allow-paid --output-dir <new-directory>
```

- `semantic` 保留自然语言问题，使用 QA Search Pro；固定每页 10 条，用 cursor-only 请求翻页。
- `pro` 使用题名短语查询，适合批量候选，例如 `--query "ductile fracture" --mode pro --page-size 100`。
  不能把自然语言长句当作普通 Pro 的字面检索短语。
- `--year-from/--year-to` 对候选施加年份约束，未知年份不冒充满足约束；显式 `--seed-id` 可作为
  历史锚点保留。`--depth` 最大 2，`--seeds-per-depth` 最大 5，不无界滚雪球。
- `--details` 有限补原始摘要和详细元数据；`--experiments` 明确指定多少篇查询平台实验记录。
- 请求预算、候选上限、页数和截止时间共同约束流程；报错时保留 partial 状态，可用相同输入
  加 `--resume` 继续。成功响应会复用；候选池仍需相关性筛选，不声称完成系统综述。

输出 `index.html`、候选 CSV、`citation-graph.json`、`experiments.json` 和摘要证据工作台。
引文图只表示 AMiner 数据库的引用边，不写入“已核验本地引用”账本，也不推断支持/反驳/继承。
实验记录保留 methods/datasets 等原始数组，不把多个方法压成一个字符串；没有返回时保持空缺。
这一接口检索已有结构化实验记录，不等于从你上传的任意全文自动提取实验。

## 学者、机构、期刊资料

```text
python -m src.aminer_profiles --kind person --id <confirmed-aminer-id> --budget-yuan 3 --allow-paid --output-dir <new-directory>
```

学者默认读取详情、研究画像、论文；机构默认详情、学者和论文；期刊默认详情和论文。
`--section` 可重复，选取确实需要的部分；学者项目只有显式选择 `--section projects` 才请求。
机构用 offset 翻页，期刊另支持 year/limit。生成可读页面、原始资料和 `papers-candidates.json`，
后者可交给既有专题/画像筛选流程。来源关系不自动改写追踪身份配置。

## 只用缓存开发或复核

```text
python -m src.aminer_research --query "ductile fracture" --mode pro --page-size 10 --depth 1 --details 0 --cache-only --api-run-dir <existing-api-directory> --budget-yuan <该目录原预算> --output-dir <new-directory>
python -m src.aminer_profiles --kind person --id <id> --cache-only --api-run-dir <existing-api-directory> --budget-yuan <该目录原预算> --output-dir <new-directory>
```

缓存未命中只记录 `cache_miss`，绝不会转成真实请求。此模式不需要 Token，适合开发和界面验收。
本地单元测试用 `python -B tools/run_offline_tests.py`，统一禁止实际 requests HTTP 调用；CI 与
每周任务的单元测试步骤也使用这一入口。它不会阻止正式 watch 阶段正常检索。

## 原子接口入口

`python -m src.aminer_advanced catalog` 列出接口、字段和估算单价。支持 19 个非专利高级接口：
Pro/两种 QA 检索、论文详情/引用/多关键词/按年期刊检索、学者详情/画像/论文/项目、机构详情/
学者/论文/两种消歧、期刊详情/论文、实验检索，另提供免费批量论文信息辅助入口。

需要单独调用时，用 `query --endpoint <name> --params-file <json> --output-dir <api-directory>`，
并明确选择 `--allow-paid` 或 `--cache-only`。未知端点、未知字段和明显错误的分页参数会在
联网前拒绝。接口存在不代表账户权限、领域覆盖或每次调用结果都完整。

## 本轮验证范围

15 个高级接口做过真实请求验证，按目录估算累计 ¥4.49；其后按用户费用疑问暂停了真实付费
请求，改用禁止网络的响应重放。缓存重放生成 34 篇候选、23 条数据库引用边及三类实体资料页，
新增估算费用为 0。种子论文的实验检索返回 0 条，未制造实验结果。
其余接口依照合同接入并以模拟响应测试，未声称全部在线实测。

GLM 原始全文实验提取仍需单独的 `BIGMODEL_API_KEY`；本机未配置，因此尚未在线执行该模型
链路。AMiner Token 不能替代它；已有模型/embedding 凭据也不会被偷偷挪用。

接口与价格依据：[AMiner 官方接口目录](https://github.com/AMinerOrg/aminer-open-skill/blob/main/skills/aminer-academic-search/references/api-catalog.md)。
