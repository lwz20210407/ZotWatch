# AMiner 限免发现链路

第一期增加论文推荐、标题短语搜索、批量论文信息三个 AMiner 入口。它们按 2026-09-30 官方页面及账户实测为限免接口；配置 `free_only: true` 使用代码白名单，不会自动切换收费 Pro、详情或引用接口。限免价格未来仍以官方控制台为准。

## 启用与关闭

当前部署配置进入首轮影子观察，详见 [发布与效果验收](aminer-rollout.md)。下列开关说明仍适用。

在运行进程环境或 GitHub Actions repository secret 设置 `AMINER_API_KEY`，值为有效 Auth Token（不是签名用的原始 API Key）。不要写入 YAML、提交到 Git 或放进命令行参数。Windows 本机已配置用户环境变量时，可以先运行：

```powershell
$env:AMINER_API_KEY = [Environment]::GetEnvironmentVariable('AMINER_API_KEY', 'User')
```

`config/sources.yaml` 的 `aminer` 支持：

- `enabled: false`：不调用 AMiner。其缓存不会混进其他来源的聚合缓存。
- `enabled: true, mode: shadow`：抓取并保存候选对照，不改变推荐主链。
- `enabled: true, mode: live`：合并候选进入现有主题筛选、去重与评分。默认 `delivery: backfill`，AMiner 独有发现仅进入方法补漏，不占近期主列表名额。独立来源已找到、由 AMiner 补元数据的记录保留原有路径。

只有显式设置 `delivery: recent_and_backfill`，才允许适用性通过且具有可核实日精度日期的 AMiner 独有候选竞争近期列表。此开关不绕过原有主题、评分、已读或日期筛选。

未配置 Token 或来源故障会记录不完整状态，继续其他来源。没有改评分权重，也没有启用合作群体分析。正式工作流读取 `${{ secrets.AMINER_API_KEY }}`，本机 MCP 配置不会自动同步给 Actions。

## 查询与预算

现有 `research.facets` 同时提供推荐主题和短语，不建立第二套研究画像。推荐使用 `semantic_query`；搜索优先取同一方向下的 `aminer_phrases`，未配置时才回退到 `terms` 中的多词短语。不会拼接多个短语冒充布尔条件。修改 `aminer_phrases` 不改变已有文库向量的画像指纹。

先覆盖每个方向的第一条短语，然后执行有上限的推荐，再继续第二条短语/分页。默认每轮最多 4 个推荐查询；连续两组新增 AMiner ID 比例低于 25% 时，停止本轮剩余推荐，继续短语搜索和元数据补全。推荐起点独立持久轮换，避免每周只查同几个方向。记录的观察值是本轮不同推荐结果间的增量，不代表全网召回率；停止可能漏掉后续查询的个别新记录，因此保留轮换而不永久封禁任何研究方向。

每次实际尝试（包括重试）都计入 `max_requests`。推荐较慢，`max_run_seconds` 限制整轮处理，发现阶段为信息补全预留时间和请求。单次网络超时为 `timeout_seconds`；服务连续传输的异常不等同严格墙钟超时。429/40306 记录持久冷却，不让周报工作进程长时间等待。

缓存位于 `data/cache/aminer`，缓存键包含接口、请求参数/JSON 请求体和查询指纹，不包含凭据。成功空列表可以缓存，失败不能缓存。原聚合缓存有来源/筛选配置指纹，首次部署会刷新旧缓存。

## 数据合同

- 实测 rec5 返回 `data[].papers`；普通搜索和批量信息返回 `data[]`。客户端同时验证 HTTP 状态、业务状态和数据形状。
- 推荐的 `summary` 独立放在 `extra.generated_summary`，不会伪装成论文摘要。
- 免费 `abstract_slice` 标注为片段；优先保留已有完整原始摘要。
- AMiner ID 为 `aminer:<id>`；同 ID 可合并推荐中缺失而搜索中提供的 DOI。跨来源仅在题名、年份和完整第一作者一致且 DOI 无歧义时补齐身份；不会合并不同 DOI。
- 在严格主题筛选前批量补全缺摘要候选。少量免费完整题名查询仅在返回 AMiner ID 一致时补 DOI。
- live 模式下，少量已有 DOI 的候选使用现有 OpenAlex 预算读取精确 DOI 元数据；此项不是 AMiner 收费接口，可用 `max_doi_resolutions: 0` 关闭。
- 只有年份时 `published` 为 null，年份单独保留。主列表要求日精度；默认所有合格 AMiner 独有发现最多进入 `backfill_items` 个方法补漏位置（默认 3），不称“本周新发表”。未来日期不进入补漏。
- `provenance`、`external_ids`、`source_metrics` 保留来源；引用分桶不当精确数字，不相加不同来源引用量。
- 无 DOI 的文献仍可发现和阅读，第一版沿用原 DOI 反馈合同，不制造无效反馈链接。

## 方法适用性

适用性只依据原始题名和摘要/摘要片段，不依据生成式概述，不更改评分公式。其他金属只要提供现有研究链的证据仍可通过，不因不是 TC4 而降级。

- 氢环境、切削加工、胞状/夹层结构及软件资源：标记 `conditional`，写明需要核对的条件。
- 缺摘要或缺研究方向证据：标记 `insufficient_evidence`，保留待补证。
- 前言/编辑内容以及题名明确指向本轮已识别的砂土、岩石、混凝土、弹性体对象：标记 `exclude`。
- 其余有研究方向线索的条目标记 `suitable`，不等同于已经全文验证。

这些规则只控制 AMiner 独有发现的自动交付，不删除文献或已有反馈，也不改变其他来源的既有推荐路径。条件参考和待补证条目保存在诊断，并在网页的折叠运行诊断中展示达到评分阈值的条目，不进入邮件推荐、RSS 或已推送历史。

## 独立对照（不发送或发布）

从项目目录运行，输出必须选定为本任务专用目录：

```powershell
.\.venv\Scripts\python.exe -B -m src.aminer_compare --output-dir 'F:\PythonWoking\temp\YYYYMMDD-aminer-compare'
```

默认读取 `data/cache/candidate_cache.json` 作为冻结基准，也可通过 `--baseline` 指定其他相同结构快照。输出包括 `comparison.json`、三组候选及差异记录 `candidates.json`、人工标签表 `review.csv`、浏览用 `review.html`。基准抓取时间保存在报告中；旧快照不能冒充同一时刻的在线检索。

`--facets strain_rate ductile_fracture` 可限研究方向，`--max-requests 12` 可限请求数。`--resolve-dates` 使用隔离的 OpenAlex 预算缓存核对少量精确 DOI。`--rank` 使用原画像与现有 embedding 服务做额外排名对照，会产生该 embedding 服务的请求；不重建画像，不修改文库或推送历史。其 top20 是原始评分对照，不包含完整邮件流程的已读历史、反馈和多样性筛选。排名工作进程默认 180 秒总上限，可用 `--rank-timeout` 调整；超时保留候选对照并写明 `ranking-status.json`，不宣称排名验收成功。未人工标注时，不报告准确率或“召回提高百分比”。

正式 watch 生成报告时会额外保存 `reports/aminer-candidates.json`，诊断中的 `aminer` 列出实际调用、缓存命中和不完整状态。摘要缺失率、日期精度、相关性及独有文献应一起复核，不能以候选数量证明改善。

## 验证

```text
python -B -m unittest discover -s tests -p test_aminer.py -v
python -B -m unittest discover -s tests -v
python -B -m src.check_config
```

测试使用 mock，不应消耗 AMiner 额度。源层测试覆盖接口白名单、鉴权业务错误、限流、POST 缓存、请求预算、rec5 实际包裹结构、日期精度、身份与来源合并、严格专业筛选和开关隔离。
