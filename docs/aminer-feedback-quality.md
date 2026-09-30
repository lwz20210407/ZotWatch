# 第二阶段：反馈闭环与实体核查

此阶段建立在 AMiner 发现/方法补漏之上，不启用正式周报来源，不自动上传全文或修改作者追踪配置。

## 无 DOI 文献反馈

旧 `doi` 反馈和 `zotwatch-feedback-v1` 链接继续有效。无 DOI 的 AMiner 文献使用稳定的 `aminer:<id>` 与 v2 标记，不用题名猜身份，不伪造 DOI。

```text
python -m src.cli feedback --work-id aminer:PUBLIC_PAPER_ID --rating read
python -m src.cli feedback --work-id aminer:PUBLIC_PAPER_ID --rating irrelevant --reason wrong_material --scope lpbf_tc4 --facets lpbf_tc4
```

新链接仍然只是预填 GitHub Issue，需用户确认提交；只解析仓库所有者的结构化反馈，不执行 Issue 中的其他文字。公开反馈不要填写私人笔记。

DOI 补齐后，候选仍携带 AMiner ID，旧 ID 反馈可以继续匹配。包含 DOI+AMiner ID 的明确反馈可建立别名关联，避免重复计算同一条反馈；出现互相冲突的 DOI 映射时不自动合并。没有明确关联时，当前候选的 DOI 反馈优先于 ID 反馈，同一身份的研究方向作用域优先于全局。`reset` 撤销对应身份/作用域的明确反馈。

原因码支持 `off_topic / wrong_material / wrong_conditions / already_known / metadata_error / useful_method`，诊断按研究方向统计有效反馈记录。它不是人工准确率或独立论文数量。原先有界的方向偏好更新继续工作，不因新增来源扩大加分上限。

## 确认条件参考

网页待核查条目增加“确认方法可迁移”“暂缓自动推荐”等预填链接，也可本地执行：

```text
python -m src.cli feedback --work-id aminer:PUBLIC_PAPER_ID --rating transferable --reason useful_method --applicability approve
python -m src.cli feedback --work-id aminer:PUBLIC_PAPER_ID --rating later --applicability hold
```

确认只能解除有原始摘要证据的工况条件限制，不能绕过撤稿、前言/编辑内容、缺原始证据、未来日期、评分阈值或已读状态。研究方向限定的确认只在该方向确有文本证据时生效。其他来源的既有交付路径不受 AMiner 专有策略影响。

## 限免实体查询

手动查询学者、机构或期刊，用于名称规范化与候选 ID 复核，不会自动把同名作者合并为同一个人：

```text
python -m src.aminer_entities --venue "International Journal of Impact Engineering" --output-dir <task-output>
python -m src.aminer_entities --person "Researcher Name" --org "Institution" --output-dir <task-output>
python -m src.aminer_entities --organization "Institution" --output-dir <task-output>
```

输出 `entity-review.json`，`selected_id` 始终为空，保留多候选与歧义。人工需结合机构、论文、ORCID 或 ISSN 再决定映射。本命令只调用对应限免搜索接口；详情、学者论文清单和机构产出不在此阶段自动调用。
