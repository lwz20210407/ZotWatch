# ZotWatch LPBF/TC4 精准深化设计

## 目标

在不扩大到陶瓷球等旁支主题的前提下，深化 LPBF/TC4 钛合金博士主线的文献监测配置，并通过一次有实际研究价值的仓库更新避免 GitHub Actions 因 60 天无仓库活动而进入 `disabled_inactivity`。

## 修改范围

仅修改 `config/sources.yaml`：

- 补充 2–3 种当前缺失且与 LPBF/TC4 力学响应高度相关的期刊。
- 补充少量缺失词形与研究概念，包括 L-PBF/SLM、thermo-viscoplastic、thermal softening、adiabatic heating、anisotropic ductile fracture、load-path-dependent damage、shear localization 和 LS-OPT inverse parameter identification。
- 新增约 5 条包含 Ti-6Al-4V/TC4 材料限定词的复合检索式。
- 保留现有 `required_any_group_sets`、排除词、评分逻辑和 workflow 配置。

不修改邮件 Secrets、Zotero 凭据、运行频率、Pages 部署方式或源代码。

## 精度策略

新增概念不以宽泛单词单独检索，而与 `Ti-6Al-4V`、`TC4` 或 `LPBF` 及 constitutive/fracture/damage 等力学问题词组合。现有的“材料词 + 力学问题词”过滤门槛继续生效，以提高召回率但限制无关推送。

## GitHub Actions 处理

配置提交推送到 `main` 后，现有 `push` 触发器将启动一次 `Weekly Watch & RSS`。该真实提交同时构成新的仓库活动，避免当前 60 天 inactivity 停用。不会加入定时空提交或其他人为 keepalive 机制；若后续再次连续 60 天无仓库活动，仍需进行有意义的维护或在 GitHub Actions 页面重新启用 workflow。

## 验证

- 解析 `config/sources.yaml`，确认 YAML 语法和配置模型加载正常。
- 检查新增期刊、查询和关键词无重复且均受 TC4 主题门槛约束。
- 审查最终 diff，不包含 Secrets 或无关文件。
- 推送后确认 workflow 状态仍为 `active`，并核对本次 `push` 触发的运行结果。
