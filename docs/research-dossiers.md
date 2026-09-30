# 第三阶段：按需专题证据工作台

此入口把选定论文、原始摘要和用户明确提供的本地全文整理成可审阅材料。默认离线，不运行向量服务，不自动下载/上传 PDF，不进入每周邮件或更新已推送历史。它生成证据比较与待核查问题，不冒充自动完成的全文综述。

## 从候选快照生成专题

支持独立对照的 `candidates.json`、普通候选缓存，以及每周的 `aminer-candidates.json`（后者字段较少，只有题名依据时标为题名线索，其余字段保持未知）：

```text
python -m src.research_dossier --topic "延性断裂与正则化" --snapshot <snapshot.json> --cohort combined --work-id doi:10.xxxx/example --output-dir <new-task-directory>
```

`--work-id` 可重复，接受 DOI、`doi:<doi>` 或 `aminer:<id>`。不指定时按快照顺序取最多 10 篇，不代表重新完成推荐排序；`--max-papers` 最大 20。输出目录必须是新目录或空目录，已有文件不会被覆盖。

输出：

- `dossier.html`：材料、制备状态、载荷条件、模型、标定、实现、验证、局限八类线索及原文位置。
- `evidence-ledger.json`：版本化证据账本，保存论文身份、证据级别、摘录/句子哈希、未知项、已核验与未核验的引用。
- `agent-handoff.json`：交给 Codex/Claude 继续精读的任务范围及证据约束。引用必须指向账本证据和原文位置，缺证据时补文献，不根据题名或图谱边猜结论。
- `citation-traces.html/json`：可按题名、方法或原文筛选的引文脉络，展示来源论文、目标论文、引用语境和参考文献原文，并可返回工作台对应论文。

“提及 GTN”不等于“采用 GTN”；否定句仍作为原文语境保留。摘要片段不冒充完整摘要，生成式概述不参与证据提取。相同证据摘录在页面仅显示一次。

## 加入本地全文

TXT/Markdown 直接读取，PDF 文本抽取需单独安装 `requirements-research.txt`，每周环境无需此依赖。支持 UTF-8 文本、最多 30 MiB 文件、最多 200 页非加密 PDF；扫描 PDF 没有可抽取文本时明确标为缺失，不自动调用 OCR。

通过证据清单把文件绑定到已选论文：

```json
{
  "documents": [
    {"work_id": "doi:10.xxxx/example", "path": "paper.md"}
  ],
  "citations": []
}
```

添加 `--evidence-map <map.json> --evidence-root <allowed-directory>`。默认根目录是清单所在目录。只能读取该目录内的 TXT/Markdown/PDF，解析后的路径（包含链接）不得越界；不把机器绝对路径写入导出的证据账本。

文本使用 `L1` 等行号，PDF 使用 `p.1` 等页号。文档 SHA-256 帮助核对版本。文档绑定也支持已知 AMiner ID；有多个冲突身份时拒绝自动关联。

## 可核验引用关系

支持数字引用标记。清单中每条引用必须同时给出引用语境和参考文献条目，二者都要在所绑定原文位置找到，并且引用号、目标 DOI/题名一致：

```json
{
  "from": "doi:10.xxxx/source",
  "to": "doi:10.xxxx/target",
  "marker": "[1]",
  "locator": "L12",
  "quote": "Original sentence containing [1].",
  "reference_locator": "L80",
  "reference_quote": "[1] Target paper title. DOI:10.xxxx/target",
  "relation": "cites"
}
```

只有上下文、参考文献和目标身份都能核验时，才生成 `grounded_citation`。伪造摘录、缺失目标、无法支持的映射以及 `supports/extends/refutes` 等语义断言进入未核验记录。单凭引用边不能证明采用、改进或反驳。复杂引用格式和跨页断行仍需人工整理，不做全自动承诺。

### 自动定位本地数字引用

添加 `--auto-citations`，即可扫描清单绑定的本地全文，减少手填引用清单：

```text
python -m src.research_dossier --topic "损伤模型的来源" --snapshot <snapshot.json> --evidence-map <map.json> --evidence-root <directory> --auto-citations --output-dir <new-task-directory>
```

自动定位要求明确的 `References`、`Bibliography` 或“参考文献”标题，条目须以 `[1]` 等编号开头。
正文支持 `[1]`、`[1, 3]`、`[2–5]`，单个标记最多展开 20 个编号。只匹配已选论文；未知目标
进入待核查，不自动扩展收费引文接口。最多检查 100 个正文引用目标，达到上限时标记未扫描完整。

同一 PDF 页内的换行可合并；跨证据位置的参考文献需要人工清理定位。作者年份制、上标引用、
扫描 PDF OCR 和不规范书目暂不自动处理。相同题名有多个候选、编号重复或 DOI 冲突时不猜测。
DOI 必须完整相等；已出现冲突 DOI 时不会靠相似题名覆盖它。组合引用手动标注时，`marker`
填写原文完整标记，另用 `reference_marker` 指定目标条目的单一编号。

脉络只表示“此处引用了这篇论文”。保留否定语境，不自动标成“采用”“改进”“支持”等研究结论。
没有本地证据时页面明确显示未发现可靠关系；归档重新渲染时标明未重新核验原文。这些输出属于
完整私有证据包，`publish-summary` 不会将引文摘录复制到公开目录。

## 可选限免发现

只有显式提供 `--discover --facet <existing-facet-id>` 才联网：

```text
python -m src.research_dossier --topic "损伤正则化与网格客观性" --discover --facet damage_evolution --facet umat_implementation --output-dir <new-task-directory>
```

专题文字参与推荐主题；短语检索仍来自指定方向的短语配置。每次最多 12 个 AMiner 请求、2 个推荐查询，只走限免白名单，不调用付费引文或深研接口。结果仍是检索候选，不代表完成原周报评分、文库去重或完整性证明。

建议流程：先从周报选择有价值的论文，再补全文生成本地证据包；在 Codex/Claude 中结合 `paper-source-trace` 的证据原则继续精读，将可核验引用上下文回填清单后重新生成到新的任务目录。不要为全部周报候选自动运行全文分析。
