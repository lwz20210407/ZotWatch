# ZotWatch TC4 中心与跨金属方法监测设计

## 目标

在不扩大到陶瓷球等旁支主题的前提下，以 LPBF/TC4 钛合金博士主线为核心，同时保留铝合金、钢等金属中可迁移的本构、断裂、损伤与参数反演方法，并通过一次有实际研究价值的仓库更新避免 GitHub Actions 因 60 天无仓库活动而进入 `disabled_inactivity`。

## 修改范围

监测配置仅修改 `config/sources.yaml`：

- 保留现有期刊列表，不新增宽口径期刊，避免现有非 TC4 匹配路径放大噪声。
- 新增 5 条包含 Ti-6Al-4V/TC4 材料限定词的直接检索式，以及 2 条跨金属方法检索式。
- 补充 thermo-viscoplastic、thermal softening、adiabatic heating、shear localization、inverse identification 和 LS-OPT 等方法/机理词。
- 将热软化等概念加入多金属力学路径，并为 LS-OPT/参数反演建立“金属材料 + 反演方法 + 力学对象”三重门槛。
- 将 ML 路径改为“ML 方法 + 力学对象 + 金属材料”三重门槛，避免仅凭宽泛的 `metal` 或 `alloy` 放行。
- 保留现有排除词、评分逻辑和 workflow 配置。

不修改邮件 Secrets、Zotero 凭据、运行频率、Pages 部署方式或源代码。

## 精度策略

TC4 直接证据继续要求“TC4 材料词 + 力学问题词”。铝合金、钢等可迁移证据要求“金属材料词 + 热软化/损伤等力学问题词”。LS-OPT 与参数反演证据必须同时命中金属材料、反演方法和本构/断裂/损伤对象；不相关的软件优化、金融、生物医学等内容继续被排除。

## GitHub Actions 处理

配置提交推送到 `main` 后，现有 `push` 触发器将启动一次 `Weekly Watch & RSS`。该真实提交同时构成新的仓库活动，避免当前 60 天 inactivity 停用。不会加入定时空提交或其他人为 keepalive 机制；若后续再次连续 60 天无仓库活动，仍需进行有意义的维护或在 GitHub Actions 页面重新启用 workflow。

## 验证

- 解析 `config/sources.yaml`，确认 YAML 语法和配置模型加载正常。
- 检查 TC4 直接正例与铝合金/钢的可迁移方法正例能够通过，同时验证无力学对象的泛化 LS-OPT 内容被拒绝。
- 审查最终 diff，不包含 Secrets 或无关文件。
- 推送后确认 workflow 状态仍为 `active`，并核对本次 `push` 触发的运行结果。
