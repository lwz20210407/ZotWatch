# AMiner 首轮生产接入：影子模式

首次启用采用 `enabled: true, mode: shadow`，每轮最多 24 次请求、2 个推荐查询，来源时间预算
120 秒，单次超时 15 秒。现有研究方向、旧来源预算和评分权重保持不变。AMiner 候选只进入
对照与诊断，不进入正式选文、RSS 或推送历史。

## 发布验收

1. Actions 配置 `AMINER_API_KEY` repository secret，复用有效 Auth Token；不得提交凭据文件。
2. 对待发布分支执行 Weekly Watch 的 `workflow_dispatch dry_run=true`。它使用真实画像、来源、
   排序和报告流程，跳过 Pages、邮件和历史提交。测试邮件预览使用无效示例地址，避免收件人
   地址进入公共 artifact。单元测试步骤不读取 AMiner 生产 Token。
3. 检查运行结果及 artifact：AMiner 实际调用与警告、原周报是否完成、历史未变、未调用发送步骤。
4. 按依赖顺序合并功能 PR，最后合并影子开关；验证 main CI。正常周四任务继续按原计划交付。

## 观察材料

同一轮 watch 导出 `reports/aminer-shadow.json`，保存 baseline/aminer/combined 三组主题筛选后
候选，供后续独立去重和画像验收复用；保留生成时间、基准缓存时间及 API 警告。不同来源缓存与
索引时点仍可能不同，因此“同一轮”不是“完全同时采集”，也不代表最终排名对照。

`reports/aminer-review.csv` 列出相对旧候选集合独有的 AMiner 身份，并留空人工相关性及原因。
它尚未排除已在个人文库/推送历史中的文献，不应直接当作新增有效论文数。无 DOI 文献保留
稳定 AMiner ID。模型评分、规则标签不会回填为人工判断。

已有 `ranking-baseline.json` 表示同一候选集的旧权重排序，不是 AMiner 关闭状态的对照排名。
正式声称推荐质量改善，需要用冻结三组候选和同一画像/历史进行对照，再完成独立人工标注；
未完成时只报告调用、候选、去重与运行可靠性，不报告提升百分比。

## 回退

- 关闭 AMiner：把 `config/sources.yaml` 中该来源的 `enabled` 改回 `false`，正常提交部署。
- 保留影子观察：维持 `mode: shadow`；不要在质量证据不足时改成 `live`。
- 影子源的初始化、网络、规范化或缓存写入失败会记录告警并保留原来源候选。
- 回退不删除凭据、缓存、已交付历史或其他来源数据。
