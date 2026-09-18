# ZotWatch

[English Version](README.en.md)

从 Zotero 文库反推研究兴趣画像，每周自动检索新发表文献，按相关度排序后推送到邮箱、
RSS 和 GitHub Pages。

每周四北京时间 07:00 由 GitHub Actions 自动运行，也可手动触发或本地执行。

> Fork 自 [Yorks0n/ZotWatch](https://github.com/Yorks0n/ZotWatch)，保留原作者的整体思路，
> 在检索、打分、画像来源和投递链路上做了较多改动，详见 [与上游的差异](#与上游的差异)。

---

## 它怎么判断"跟我相关"

不是靠一张手写的关键词表，而是靠你的文库：

```
1. 画像   读本地 zotero.sqlite 的全部 4399 篇 → 标题+摘要编码成向量 → FAISS 索引
2. 检索   OpenAlex / Crossref 拉近 30 天新论文（主题、期刊、重点作者、引文关系四条线）
3. 打分   每篇新论文编码后，与画像中最近 5 篇求余弦均值 → 语义分（权重 0.68）
4. 投递   去重、多样性选择 Top 20 → 邮件正文 + RSS + Pages
```

关键词配置（`config/sources.yaml`）只负责**粗筛**决定去抓什么，排序由向量相似度决定。

### 打分是有界的

7 个分量各自压到 [0, 1]，权重和为 1，所以总分也在 [0, 1]，`must_read` / `consider`
阈值跨轮次可比。完整公式、饱和参数和设计理由见 **[docs/scoring-model.md](docs/scoring-model.md)**。

### 画像来自本地库，不是云端

Zotero Web API 只返回**已同步到 zotero.org** 的条目。本库 4399 篇里有 3881 篇
`version = 0`，从未上云——只用 API 的话画像只覆盖 12%。所以画像改为本地生成：

```bash
python -m src.cli profile --local --bundle
```

直接读 `zotero.sqlite`（先复制再只读打开，绝不写入你的库），产出 `data/profile-bundle.tar.gz`，
上传为 **draft release** 供 CI 取用。用 draft 是因为 bundle 含全库标题摘要，而本仓库是公开的。

---

## 日常维护

| 什么时候 | 做什么 |
|---|---|
| 每周四 07:00 | 自动运行，无需干预 |
| 文库新增较多后（约每月） | 本地刷新画像并重新上传 bundle，见下 |
| 想改推荐口径 | 调 `config/scoring.yaml` 的权重/阈值，或 `config/sources.yaml` 的期刊与关键词 |
| 收到不相关的推荐 | 点报告里的反馈链接开 Issue，下轮生效 |

### 刷新画像

```bash
python -m src.cli profile --local --bundle
gh release upload profile-latest data/profile-bundle.tar.gz --clobber
```

不刷新也能跑，只是新加入文库的论文暂时不影响排序。

### 手动触发

Actions → **Weekly Watch & RSS** → Run workflow。勾选 `dry_run` 可演练：
照常检索打分出报告，但**不发邮件、不部署 Pages、不记录推送历史**。

---

## 首次部署

1. **Fork 本仓库**，在 Settings → Pages 把 Source 设为 **GitHub Actions**。

2. **配置 Secrets**（Settings → Secrets and variables → Actions）：

   | 名称 | 用途 |
   |---|---|
   | `ZOTERO_API_KEY` | Zotero [个人设置](https://www.zotero.org/settings/security) → Create new private key，Personal Library 给读权限 |
   | `ZOTERO_USER_ID` | 同页面 `User ID: ...` |
   | `EMBEDDING_API_KEY` | 嵌入服务的 key，见 `config/embedding.yaml` |
   | `EMAIL_TO` | 收件人，多个用逗号分隔 |
   | `SMTP_HOST` / `SMTP_PORT` / `SMTP_USERNAME` / `SMTP_PASSWORD` / `SMTP_FROM` | 发信账户 |
   | `CROSSREF_MAILTO` / `OPENALEX_MAILTO` | 礼貌标注邮箱（可选但建议） |

   **`EMAIL_TO` 要存成 Secret，不要存成 Variable**：公开仓库的 Actions 日志也是公开的，
   只有 Secret 会被打码，Variable 会把邮箱明文写进日志。

   缺 `EMBEDDING_API_KEY` 或 `EMAIL_TO` 时，工作流在**开头**就报错退出，
   而不是跑完 40 分钟、Pages 都部署完了才在最后一步失败。

3. **改配置**：`config/sources.yaml` 的关键词和期刊、`config/research.yaml` 的研究方向，
   都是按 TC4 钛合金本构与断裂写的，换方向必须改。

4. **生成画像**：本地 `python -m src.cli profile --local --bundle` 后上传 release，
   或者直接跑一次工作流让它走 Web API 路径（只覆盖已同步的部分）。

5. Actions → 启用 workflow → Run workflow。

订阅地址：`https://<用户名>.github.io/ZotWatch/feed.xml`，
历史推送在 `https://<用户名>.github.io/ZotWatch/archive.html`。

---

## 本地运行

```bash
git clone https://github.com/lwz20210407/ZotWatch.git && cd ZotWatch
pip install -r requirements.txt

python -m src.check_config                      # 配置自检，不联网
python -m src.cli profile --local --bundle      # 从本地 zotero.sqlite 建画像
python -m src.cli watch --rss --report --top 20 # 检索打分出报告
python -m src.cli notify --dry-run              # 只生成邮件到 reports/email-preview.eml
```

密钥放仓库根目录的 `.env`（已 gitignore）：

```
ZOTERO_API_KEY=...
ZOTERO_USER_ID=...
EMBEDDING_API_KEY=...
```

嵌入走远程 API，因此**不需要装 `sentence-transformers` 和 torch**。
只有把 `config/embedding.yaml` 的 `provider` 改成 `local` 时才需要。

---

## 配置文件

| 文件 | 内容 |
|---|---|
| `config/zotero.yaml` | Web API 参数；`local.data_dir` 指向本地 Zotero 数据目录 |
| `config/embedding.yaml` | 编码器。默认 OpenAI 兼容远程 API，1024 维；含离线回退设置 |
| `config/scoring.yaml` | 权重、阈值、饱和参数、研究优先级规则、期刊白名单 |
| `config/sources.yaml` | 检索关键词、追踪期刊、时间窗口（30 天）、分页与限速 |
| `config/research.yaml` | 7 个研究方向画像、语义查询、多样性与反馈参数 |
| `config/authors.yaml` | 重点关注作者（OpenAlex ID） |
| `config/citations.yaml` | 引文追踪种子论文 |
| `config/network.yaml` | API 预算上限与缓存时长 |

`research_priorities` 是**首个命中生效**，所以系数必须自上而下递减；
`python -m src.check_config` 会检查这一点和权重求和，CI 每次提交都跑。

---

## 阈值标定

换编码器后相似度分布会变，需要按实测重标。每轮运行日志会打印：

```text
Score distribution over 1029 ranked works: p50=0.533 p75=0.623 p90=0.681 p99=0.767 max=0.798
Label counts: {'must_read': 68, 'consider': 716, 'ignore': 245}
```

`must_read` 长期为 0 或过多时，参照 p90/p99 调 `config/scoring.yaml` 的 `thresholds`。

---

## 与上游的差异

| 方面 | 上游 | 本仓库 |
|---|---|---|
| 触发 | `push` + 定时 | 仅定时 + 手动（push 触发会导致每次提交都发一封"周报"并污染推送历史） |
| 打分 | 各分量量纲不一，引用项无上界 | 全部压到 [0, 1]，引用在 50 次饱和；时效改连续衰减 |
| 画像来源 | Zotero Web API | 本地 `zotero.sqlite` 全库（API 只能看到 12%） |
| 编码器 | 本地 `all-MiniLM-L6-v2` | 远程 API，CI 不下载模型，依赖去掉 torch |
| 向量缓存 | 每轮全库重算 | 按 `(模型签名, item version)` 缓存，只重算变动项 |
| 去重 | `token_set_ratio`，子集标题被误判为重复 | `token_sort_ratio` + 长度守卫，抑制记录提到 INFO |
| 邮件 | 正文是链接，报告在附件 | 报告即正文（`multipart/alternative`），带线程化和重试 |
| 推送历史 | 90 天过期的 CI artifact | 提交进仓库 |
| 历史报告 | 每次部署被覆盖 | 入库并提供 `archive.html` |
| 收件人 | 硬编码在 workflow | Secret |
| 测试 | 19 个 | 111 个，含缺陷回归用例 |

---

## 常见问题

- **推荐为空或太少**：先看报告顶部的漏斗（抓取 → 主题通过 → 去重后 → 推送）定位卡在哪步，
  再检查时间窗口、预印本比例上限，或调 `thresholds`。
- **推荐跑偏**：多半是画像太旧或覆盖不全，刷新一次 bundle。
- **缓存过旧**：删 `data/cache/candidate_cache.json` 强制刷新。
- **本地报错找不到时区**：Windows 需要 `tzdata`（已在 `requirements.txt`）；
  缺失时会退回固定 +08:00，不影响出图。
- **换了模型后 CI 报签名不匹配**：说明 release 里的画像还是旧模型建的，
  本地重跑 `profile --local --bundle` 再上传。

---

## 相关文档

- [打分模型](docs/scoring-model.md)
- [引文发现](docs/citation-discovery.md)
- [研究助手功能](docs/research-assistant.md)
- [可靠检索与分问题推荐](docs/reliable-discovery.md)
- [作者追踪](docs/author-tracking.md)
- [主题监测](docs/topic-monitoring.md)

全流程只使用标题、摘要和引文元数据，**不获取正文**。

## 许可证

[MIT License](LICENSE)，与上游一致。
