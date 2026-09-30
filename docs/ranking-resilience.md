# 候选排名恢复与独立验收

候选向量持久保存在 `data/cache/candidate-vectors`，按实际编码器签名、服务地址、维度、分隔符和文本内容哈希隔离。文件只含哈希键和向量，不保存文本或 API Key。成功批次立即保存；下次仅补缺失项。错误形状、非有限值和零向量不会进入缓存。

候选默认每批 8 条，超时可有限拆批（最多 2 次拆分），候选请求由这一层控制重试。库内画像构建保留原有行为；不因失败更换编码器。排名前核对实际编码器与既有画像签名，拒绝混用不同向量空间。

周报的初排、最终排序和多样性选择复用同一缓存。AMiner 独有候选的向量失败会被记为本轮延后，既有来源继续排名；既有来源自身失败仍明确报错。GitHub Actions 单独恢复/保存 AMiner 检索状态和候选向量缓存，包括失败运行的进度；不把私有全文档案加入缓存或发布。

## 对 AMiner 候选做可恢复验收

```text
python -m src.aminer_acceptance --snapshot <candidates.json> --cohort aminer --output-dir <task-directory> --max-new-texts 60 --deadline 180
python -m src.aminer_acceptance --snapshot <candidates.json> --cohort aminer --output-dir <same-directory> --cache-only
```

读取当前画像和交付历史，先进行主题、文库及历史去重，再补候选向量、应用反馈与适用性策略。输出 `progress.json`、`status.json`、`acceptance.json` 与持久向量缓存。

- 每次最多处理指定数量的新文本，有独立工作进程总时限。
- 部分失败或超时不伪造分数，也不冒充完整验收；退出码为 2。以本次 `status.json` 为准，超时后旧 acceptance 文件不代表本轮结果。
- 退出码 0 只代表所选快照的技术验收完成，不代表人工相关性评估通过、全网召回更高或生产可自动上线。
- 不重建画像、不发送邮件、不修改已推送状态。记录输入哈希以检查验收期间数据是否变化。
- 不自动导入来源不明的旧向量文件；模型、维度、文本必须一致。

这是 AMiner 候选集验收，不是完整旧源/新源周报 A/B 实验。原 `aminer_compare --rank` 也已使用持久缓存，跨次运行可恢复。
