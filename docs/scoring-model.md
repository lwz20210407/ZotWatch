# 打分模型

## 结论

总分是 7 个分量的加权和。**每个分量都被压到 [0, 1]**，`config/scoring.yaml` 的
`weights` 求和为 1，因此总分也落在 [0, 1]，`thresholds` 是跨轮次可比的绝对阈值。

```
score = 0.68·semantic + 0.12·recency + 0.08·citations + 0.02·altmetric
      + 0.06·journal_quality + 0.00·author_bonus + 0.04·venue_bonus
score = score × priority_multiplier + watched_author_bonus + citation_bonus
```

`python -m src.check_config` 会在 CI 里校验权重求和与规则顺序。

## 分量定义

| 分量 | 计算 | 饱和/参数 |
|---|---|---|
| `semantic` | `clamp01(0.4·sim + 0.6·affinity)` | `sim` = 与最近 `embedding.neighbors` 篇库内文献的余弦均值 |
| `recency` | `exp(-ln2 · days / half_life)` | `recency_half_life_days: 21` |
| `citations` | `clamp01(log1p(c) / log1p(S))` | `citation_saturation: 50` |
| `altmetric` | `clamp01(log1p(a) / log1p(S))` | `altmetric_saturation: 100` |
| `journal_quality` | SJR 在 log 空间从 floor 映射到 ceiling | `sjr_floor: 0.3`、`sjr_ceiling: 4.0`、未知期刊 `0.25` |
| `author_bonus` / `venue_bonus` | 白名单命中即 1，否则 0 | — |

## 为什么要有界

改动前 `citations` 是 `log1p(c)`，没有上界：

| 论文 | semantic | recency | citations | journal | 旧总分 |
|---|---|---|---|---|---|
| 完全对口的本周新作（0 引用） | 0.61 | 0.12 | 0.00 | 0.06 | **0.79** |
| 半相关的 3 年前论文（2000 引用） | 0.37 | 0.01 | **0.61** | 0.06 | **1.05** |

引用项单独一项就能贡献 0.61，超过语义项的理论上限 0.68 的大半，于是"被引多"压过
"和我相关"。而周更推送里的新论文按定义引用数为零，这一项等于在系统性地惩罚订阅对象。
现在引用项在 50 次引用处饱和，只能当同分时的排序依据。

`journal_quality` 原本是 `max(log1p(SJR), 1.0)`，且未知期刊同样返回 1.0。常见期刊
SJR 落在 1.0–3.0，`log1p` 后 0.69–1.39，被 floor 抹平后几乎全部等于 1.0——占着权重却
不区分任何东西。现在改成 floor/ceiling 之间的归一化，未知期刊取 0.25。

`recency` 原本是 4 级台阶，第 30 天 0.4、第 31 天 0.1，一天差 4 倍。现在是连续衰减。

回归用例见 `tests/test_scoring_bounds.py`。

## 相似度取近邻均值

`WorkRanker.rank` 检索最近 `embedding.neighbors`（默认 5）篇库内文献并取余弦均值。
原先只取最近 1 篇，库里任意一篇碰巧撞上就会把候选推高。单篇最近值仍保留在
`extra.nearest_similarity` 里，供证据图使用。

## 优先级规则顺序

`research_priority()` 返回**第一条**命中的规则，因此 `research_priorities` 的系数必须
自上而下递减。改动前 `外围方法参考`（×0.70）排在首位，标题里出现 `coating`、`ceramic`、
`polymer` 等词的核心论文会先被它命中并降权 30%，永远走不到 `TC4核心研究`（×1.0）。

## 换模型后要重新标定

`config/embedding.yaml` 默认走远程 API（SiliconFlow `Qwen/Qwen3-Embedding-8B`，1024 维），
与本机 ZotPilot 同一个编码器。不同模型的余弦分布不同，所以阈值需要按实测分布调整。
每轮运行日志会打印：

```text
Score distribution over N ranked works: p50=... p75=... p90=... p99=... max=...
Label counts: {'must_read': .., 'consider': .., 'ignore': ..}
```

`must_read` 长期为 0 或过多时，按 p90/p99 调 `thresholds`。

### 当前标定（2026-09-18，4399 篇画像实测）

```text
p50=0.522  p75=0.602  p90=0.655  p99=0.713  max=0.774
must_read=0.70 → 21 篇 (3.0%)     consider=0.52 → 约中位数以上
```

对照：用只覆盖 518 篇已同步条目的旧画像时，同一天的分布是
`p50=0.533 p90=0.681 p99=0.767`，must_read 68 篇。全库画像的分数整体偏低，
因为 7 个方向的质心从几十篇扩到上千篇后更"居中"，与单篇的余弦自然下降。
这不是变差，是质心第一次真正代表了阅读范围——所以阈值必须在换画像之后重标。
