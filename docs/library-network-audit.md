# 全库研究网络审计与 ZotWatch 扩展

日期：2026-09-17。目标：以 LPBF/SLM TC4 为中心，兼顾其他金属上可迁移的
应力状态依赖塑性、断裂、标定、数值实现和冲击验证；发现经典工作之后的持续发展者。

## 覆盖口径

| 项目 | 数量 | 说明 |
| --- | ---: | --- |
| Zotero 顶层条目 | 4407 | 本地只读 API 快照，library version 48661 |
| 顶层附件 | 8 | 不作为独立论文统计 |
| 按 DOI/规范题名合并的重复条目 | 11 | 仅审计数据合并，未删除 Zotero 条目；不等于完整人工查重 |
| 审计书目条目 | 4388 | 包括期刊、学位论文、会议、书籍等，不全是期刊论文 |
| 有效 DOI | 3682 | 全部已向 OpenAlex 查询 |
| 匹配到 OpenAlex work | 3129 | 553 个 DOI 未匹配；无 DOI 的记录不在此计数中 |
| 带参考文献列表的匹配 work | 3003 | 最后100条用较精简的响应恢复，未取得其完整作者字段 |
| 库内有向引用连接 | 37778 | OpenAlex referenced_works 与库内已解析 work ID 相交所得 |
| ZotPilot 文献视图 | 4338 | 与顶层 API 分类口径不同 |
| ZotPilot 已索引/未索引 | 4263 / 75 | 语义索引覆盖不等于全库题录覆盖；未索引中包括 JC 1983 |

全库元数据已处理，但没有宣称逐篇精读4388篇全文。语义检索用于发现主题及关键
段落，再用原始题录、代表作和引用关系交叉核对。部分语义高分命中来自参考文献
列表，不能将综述作者误认为模型原创者。索引中也观察到少量题录/文本不一致风险。

## 方法与限制

1. 从 Zotero 本地只读 API 分页读取全部顶层条目和集合。Zotero 数据未被修改。
2. 以题名、摘要、DOI、作者和集合构建元数据清单。主题启发式筛出2676项候选，
   用于分布和作者发现，不是对所有文献相关性的最终人工判定。
3. 作者共著边与论文引用边分别建立。共著关系不等于引用关系，也不等于师生关系。
4. 主题初筛涉及TC4、增材、应力状态、断裂损伤、速率温度、标定、数值实现、冲击、
   组织机制。作者发现权重对多人论文折减，超过30作者的roadmap不参与共著中心性排名。
5. 引用记录查询覆盖全部3682个有效DOI；网络失败批次已用小批次补跑。
   引文网络仍缺少无DOI、未收录和缺参考文献记录，不宣称完整穷尽。
6. 重点作者ID至少由两篇已匹配库内相关代表作交叉支持；常见名字还需逐篇机构匹配。
   6位候选身份信息不足，暂不自动启用，不代表其研究价值低。

作者发文量、年代、库内收藏偏好和引用数量都影响网络结构。引用多不是论文质量
或研究价值的充分条件。本轮同时考虑近期发展方向与TC4方法可迁移性，不按辈分排序。

## 经库内引文确认的若干发展连接

下表只表达“新论文引用了这些先前工作”，不推断导师关系，也不声称某一模型仅有单一来源。

| 后续工作 | 库内可核对引用 | 与博士课题的关系 |
| --- | --- | --- |
| 李文超等：[Lode-dependent plasticity，2023](https://doi.org/10.1016/j.jcsr.2023.108202) | Bai–Wierzbicki 2008；Hosford屈服；Gao等2011 | 应力状态依赖塑性，不能仅归为钢结构断裂应用 |
| 李文超等：[应力状态/硬化/孔洞形状断裂模型，2022](https://doi.org/10.1016/j.tws.2022.109280) | Gurson、Rice–Tracey、Bao–Wierzbicki、Bai–Wierzbicki | 具体后续模型及标定适用性 |
| 吴帅涛—肖新科合作线：[TC4塑性、断裂及弹道冲击，2023](https://doi.org/10.1016/j.ijimpeng.2023.104493) | JC、MMC、HC、Bai–Wierzbicki | 从试样材料模型走向TC4靶板验证 |
| 邓云飞等：[Lode效应与不同厚度靶板，2023](https://doi.org/10.1016/j.engfracmech.2023.109634) | JC、MMC、七模型标定比较、Lou–Yoon–Huh 2014 | 不同应力状态/破坏模式的跨金属验证 |
| Wei—Koirala—Gerke—Brünig：[各向异性塑性损伤数值实现，2026](https://doi.org/10.1016/j.ijsolstr.2025.113770) | Tvergaard–Needleman、Nahshon–Hutchinson、HC、Bai–Wierzbicki | 继承并发展本构积分和各向异性损伤 |
| Zhiyang Xie等：[颈缩后硬化的应力状态依赖，2023](https://doi.org/10.1016/j.jcsr.2023.107797) | Bai–Wierzbicki、Gurson、Rice–Tracey等 | 硬化外推选择如何影响断裂预测 |
| Zhang等：[差异硬化与各向异性扩展，2025](https://doi.org/10.1007/s12289-025-01877-9) | Barlat 2003/2005、Cazacu–Plunkett–Barlat 2006 | 屈服面演化、线性变换与非关联流动 |
| Huachao Yang—Wen Zhang—Xincun Zhuang等：[非比例塑性流动，2025](https://doi.org/10.1016/j.ijsolstr.2024.113174) | Stoughton非关联流动、路径变化硬化比较等 | 路径依赖研究线，不能只用单调比例加载模型覆盖 |

## 全库分析补出的作者群

- **Brünig—Gerke—Zistl—Wei—Koirala**：各向异性损伤、双轴/非比例试验、数值积分。
- **Šebek—Kubík—Petruška**：非二次屈服函数、负三轴度标定及断裂预测。
- **Lou—Yoon—Hu—Zhang，以及 Hou、Xie**：差异硬化、屈服面演化、颈缩后与路径效应。
- **Lian—Münstermann—Liu—Shen**：应力状态相关塑性损伤、材料模型与多尺度验证。
- **Xiao—Deng—Li/Jing及相关合作作者**：Lode塑性/断裂改进、Taylor/靶板与材料验证。
- **Beese—Wilson-Heid—Qin，Pellegrino—Gour，Cortese—Nalli—Concli**：AM多轴与动态研究。
- **Mohr—Roth—Marcadet、Korkolis等**：标定、FEMU及颈缩后响应。
- **Børvik—Hopperstad—Morin—Dæhli、Erice**：冲击、断裂和结构尺度验证。

完整58人、65 ID、机构条件及代表作见 [tracked-authors.md](tracked-authors.md)。

## 本轮配置变更

| 配置 | 更新前 | 当前 |
| --- | ---: | ---: |
| 期刊 | 57 | 70 |
| 主题查询 | 165 | 179 |
| 纳入词 | 383 | 406 |
| 组合规则 | 19 | 21 |
| 硬排除词组 | 18 | 18（另有论文撤稿状态过滤） |
| 作者主动追踪 | 0 | 58人/65 ID |

新增13本期刊来自库内相关论文和已解析的期刊身份/ISSN：
Mechanics of Advanced Materials and Structures；Journal of Materials Science & Technology；
International Journal of Applied Mechanics；The International Journal of Advanced Manufacturing Technology；
International Journal of Pressure Vessels and Piping；Finite Elements in Analysis and Design；
Modelling and Simulation in Materials Science and Engineering；Strain；Optics & Laser Technology；
International Journal of Machine Tools and Manufacture；PAMM；Acta Mechanica Sinica；Acta Mechanica Solida Sinica。

新增主题集中在颈缩后应力状态依赖、Lode塑性、孔洞形状、预测—校正本构积分、
非比例路径、负三轴度、主应力方向和屈服面演化。中文短语增加独立匹配路径。
中文期刊/学位论文在OpenAlex和Crossref的覆盖仍不完整；未宣称已经接入CNKI全库。

## 撤稿与元数据质量

库题名与OpenAlex共同标识3篇撤稿记录，均不用于正面推荐：

- `10.1016/j.jallcom.2024.177005`：[出版社撤稿说明](https://www.sciencedirect.com/science/article/abs/pii/S0925838824035928)。
- `10.1016/j.engfracmech.2022.108273`：[出版社撤稿说明](https://www.sciencedirect.com/science/article/abs/pii/S0013794422000406)。该篇作者不是邓云飞，不可因检索返回混合结果而误归属。
- `10.1007/s40436-024-00487-z`：当前库题名及OpenAlex标为撤稿；本轮未成功取得出版社页面。

撤稿是论文层面的状态，不自动否定作者的其他作品。另发现3项题名对照疑点，
其中两项可能只是书名缩写；一项Material Point Method记录的DOI需后续人工修复。
没有自动修改或清理Zotero。

## 审计材料

完整库快照、候选筛选、OpenAlex原始响应、37778条引用边及可复现分析脚本只保留
在本地本任务的日期目录。未将私人全库快照上传GitHub。仓库仅保存筛选后的公开
作者/期刊/DOI配置与汇总说明。

本报告说明本次快照中的可得证据，不是“所有研究者与论文均已穷尽”的断言。
