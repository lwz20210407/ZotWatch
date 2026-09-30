# 一次运行完成专题候选与证据整理

`research_workflow` 串联已有 AMiner 发现、主题筛选、画像库/已推送去重、可选评分、证据工作台
和私有归档。默认读取快照，不联网检索，也不调用 embedding。输出属于本地研究任务，不发送
邮件、不更新推送历史，不写生产配置或自动发布 Pages。

## 离线快照

从项目目录运行，输出使用新的任务目录：

```powershell
python -m src.research_workflow --topic "延性断裂的标定与验证" --snapshot "<候选快照.json>" --max-papers 10 --archive --output-dir "<任务目录>"
```

快照格式与 `research_dossier` 一致，`--cohort` 默认 `combined`。读取当前项目的
`data/profile.sqlite` 和已推送历史去重；画像库须已建立。不启用评分时按输入顺序选取，页面明确
提示不代表推荐排名。已读反馈和 AMiner 适用性仍会用于筛选；暂缓/待补证内容保留在筛选记录中。
`--facet` 可重复，用于限制已有研究方向；题名/摘要必须包含该方向的证据。

此入口面向尚未入库/推送的候选发现。如果要分析已经在库的种子论文，请直接使用
`research_dossier --work-id ...`；不要为绕过去重而修改画像或历史。

## 显式调用 AMiner 限免发现

```powershell
python -m src.research_workflow --topic "塑性参数反演与非线性 VFM" --discover --facet calibration --max-requests 12 --max-papers 8 --archive --output-dir "<任务目录>"
```

`--discover` 必须配合已配置的 `--facet`。本次临时启用 AMiner，沿用现有端点白名单和鉴权环境，
生产开关保持不变。默认 12 次请求，允许 1–24 次，最多两个推荐查询、120 秒来源预算；实际
调用仍受请求超时及已有客户端预算实现约束。不会升级到收费 Pro/详情/引文接口。可同时传入
快照并保守合并身份。反馈可调整推荐查询优先级，短语来自选定方向配置。

Token、权限、服务错误会保留在运行状态中；故障导致没有返回结果时为 `partial`，不能解释为
领域没有论文。正常的请求/候选上限也保留说明，不声称检索完整。若已完成发现阶段，恢复时
复用冻结候选；想重新搜索需另建任务目录。

## 画像评分与恢复

添加 `--rank` 调用现有独立验收工作进程，使用当前画像及相同 embedding 模型。默认最多尝试
20 个未缓存文本，评分进程期限 180 秒；用 `--max-new-texts` 和 `--deadline` 调整。评分需要
原 embedding 服务凭据，可能产生该服务调用。`--rank --cache-only` 仅用已存在的本次任务向量
缓存，不发 embedding 请求。

评分未完成时只保存部分候选，不生成冒充完整结果的专题。保持题目、输入文件与配置不变，
添加 `--resume` 再运行，可增加本次评分预算；成功向量位于 `ranking/vector-cache/`，不会重算。
切换模型、修改输入文档、反馈或画像后必须新建任务目录，避免混合不同依据。

完成后重复 `--resume` 不产生新请求或重写完整结果；已完成产物被外部编辑/移除时拒绝覆盖。
归档失败可复用已经完成的证据包重新归档。写入中途异常留下的未确认目录保留供恢复，工具
不会自行删除；若状态不允许安全恢复，请换一个新目录。不要通过删除别人留下的锁来强行运行。

## 本地全文与引文

可添加 `--evidence-map <map.json> --evidence-root <directory> --auto-citations`，格式与
[证据工作台](research-dossiers.md) 相同。只使用最终入选论文绑定的文档，未入选文档数量记入
说明；文档身份不靠相似题名猜测。文档内容哈希参与恢复检查。没有全文时只提供题名/摘要线索，
引文脉络明确留空；不能凭空补出实验参数或论文关系。

## 输出

- `index.html`：运行概览、候选数量、限制说明及证据入口。
- `run.json`：阶段状态、输入/配置指纹、产物哈希及请求统计；不保存 Token。
- `candidates.json`：本次冻结候选。
- `selection.json`：入选文献或评分未完成时的部分结果、适用性待核查记录。
- `comparison.csv`：八类方法线索的比较长表，包括未知项和证据 ID；不是已核实的实验结论。
- `dossier/`：证据工作台、账本、引文脉络及后续精读交接文件。
- `private-archive/`：仅显式添加 `--archive` 才写入的任务内私有归档。

公开分享仍需另行调用 `research_archive publish-summary` 并正常审阅、提交、部署。本命令没有
自动发布选项。全文不会发送到 AMiner 或 embedding；embedding 继续仅使用题名和摘要。
