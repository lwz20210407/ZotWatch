# TC4 博士主线：文献监测覆盖说明

更新日期：2026-09-17。研究定位：LPBF/SLM TC4（Ti-6Al-4V）在温度、
应变率和复杂应力状态下的塑性、延性断裂与损伤演化；本构/失效模型建立、
参数标定、LS-DYNA/Abaqus 实现和冲击侵彻验证；兼顾其他金属的可迁移方法。

## 研究环节与检索入口

| 环节 | 代表性词组 | 纳入原则 |
| --- | --- | --- |
| 材料与制造 | Ti-6Al-4V、Ti6Al4V、Ti64、TC4、LPBF、SLM、PBF-LB/M | 材料/工艺别名统一；与力学主题组合 |
| 温度和应变率 | high/elevated/low temperature、cryogenic、strain rate sensitivity、thermomechanical coupling、thermal softening | 兼顾准静态、中间及高应变率，温度与速率词不要求同时出现 |
| 应力状态和路径 | triaxiality、Lode、tension shear、compression shear、biaxial、plane strain、nonproportional loading | 拉、压、剪及非比例加载均有入口 |
| 本构与硬化 | JC、ZA、PTW、KHL、RK、MTS、Swift、Voce、Hockett-Sherby、Ludwik、Ludwigson、post-necking | 不要求一篇论文同时出现多个互为替代的模型 |
| 屈服与各向异性 | CPB06、Barlat、Hill、Hershey-Hosford、pressure dependence、strength differential、flow potential | 保留金属通用理论和屈服/断裂耦合方法 |
| 断裂与损伤 | MMC、HC、DF 系列、GTN、Lemaitre、GISSMO、DIEM、void nucleation/growth/coalescence、shear banding | 核心对象与模型、机理同时覆盖 |
| 试验测量 | SHPB/SHTB、dynamic tension/torsion、pulse shaping、stress equilibrium、DIC、infrared thermography、VFM | 测量方法可不注明具体合金，但须涉及力学试验 |
| 参数标定 | LS-OPT、inverse modeling、FEMU、Bayesian calibration、identifiability、sensitivity、UQ、multi-objective calibration | 保留 Al/steel 等材料及通用力学反演方法 |
| 数值实现 | UMAT/VUMAT、UMAT41、MAT_224、return mapping、stress integration、consistent tangent | 明确关联本构、塑性或损伤，避免泛软件内容 |
| 网格与演化 | regularization/regularisation、mesh objectivity、characteristic length、fracture energy、nonlocal damage、element deletion | 不仅搜起裂准则，也搜演化与数值客观性 |
| 组织机制 | porosity、lack of fusion、build orientation、texture、prior beta grains、martensite、EBSD、fractography、HIP、heat treatment | 与金属力学响应关联后纳入，减少硬排除 |
| 冲击验证 | ballistic limit、V50、residual velocity、perforation、shear plugging、petalling、spall、Taylor impact | 独立验证入口，不要求论文一定出现具体模型名称 |

## 排序与排除

保留 Zotero 画像相似度、时间和原有文献指标评分，再施加经验系数：
TC4 核心 1.00、跨金属方法 0.95、机制参考 0.85、其他相关研究 0.90、外围参考 0.70。
系数只是当前可调整设置，不是经过标注数据验证的准确率或相关概率，也不保证严格分组顺序。
外围主题仅在标题中命中时降权；摘要里作为对比方法提及不会触发该降权。
降权后的论文仍可能超过推荐阈值，或因评分降低而不进入最终前 20 篇。

硬排除精简为 18 个明确离题词组。腐蚀、涂层、焊接、疲劳、晶体塑性、脆性转变、
分子动力学等不再仅因词汇出现就全部删除。扩大候选范围后仍需结合实际推送调整。

## 新增期刊（12 本）

下表给出监测用途，非投稿推荐。名称/ISSN 经 Crossref `/journals/{issn}` 核对；
范围依据出版社或期刊主办方页面。ScienceDirect 部分页面返回 403 时，使用出版社
Elsevier Shop 或 Acta Materialia 主办方页面。没有声称覆盖所有相关期刊。

| 期刊 | ISSN | 监测用途与官方来源 |
| --- | --- | --- |
| Journal of Dynamic Behavior of Materials | 2199-7446 | 动态响应、极端温度/压力、Hopkinson 与模型验证；[范围](https://link.springer.com/journal/40870/aims-and-scope) |
| Metallurgical and Materials Transactions A | 1073-5623 | 金属组织、加工与力学性能；[范围](https://link.springer.com/journal/11661/aims-and-scope) |
| Computational Mechanics | 0178-7675 | 计算力学、材料模型与数值方法；[范围](https://link.springer.com/journal/466/aims-and-scope) |
| Journal of Materials Engineering and Performance | 1059-9495 | 工程材料的加工—性能关联；[范围](https://link.springer.com/journal/11665/aims-and-scope) |
| Experimental Techniques | 0732-8818 | 实验力学与测量方法；[范围](https://link.springer.com/journal/40799/aims-and-scope) |
| Journal of Manufacturing Processes | 1526-6125 | 增材与制造过程相关力学研究；[出版社](https://shop.elsevier.com/journals/journal-of-manufacturing-processes/1526-6125) |
| Scripta Materialia | 1359-6462 | 组织—性能机制短文；[主办方](https://actamaterialia.org/journals/scripta-materialia) |
| Materialia | 2589-1529 | 加工—组织—性能研究；[主办方](https://actamaterialia.org/journals/materialia) |
| Measurement | 0263-2241 | 测量、标定和不确定度方法；不是侵彻方向投稿推荐；[出版社](https://shop.elsevier.com/journals/measurement/0263-2241) |
| Mechanics Research Communications | 0093-6413 | 力学理论与方法短文；[出版社](https://shop.elsevier.com/journals/mechanics-research-communications/0093-6413) |
| Optics and Lasers in Engineering | 0143-8166 | 光学测量、DIC/全场实验方法；[出版社](https://shop.elsevier.com/journals/optics-and-lasers-in-engineering/0143-8166) |
| International Journal for Numerical Methods in Engineering | 0029-5981 | 有限元、算法及数值实现；[出版社](https://onlinelibrary.wiley.com/page/journal/10970207/homepage/productinformation.html) |

原 45 本保留，总数 57 本。主题检索不限定在这些期刊内，列表以外的论文也可以进入。
Materials Science and Engineering: A 的 Crossref 刊名为 Materials Science and Engineering A；
已用 ISSN 0921-5093 定位，并规范刊名标点用于排序加分匹配。

## 检索行为核验与限制

- [OpenAlex 官方检索说明](https://help.openalex.org/api/searching/)说明，无显式 Boolean
  运算符的词按 AND 组合。因此旧的长串替代模型查询改为 165 条分主题短查询。
- [Crossref filters](https://www.crossref.org/documentation/retrieve-metadata/rest-api/rest-api-filters/)
  说明 `container-title` 是精确匹配，`issn` 可用于稳定定位。
- [Crossref API 文档](https://github.com/CrossRef/rest-api-doc)提供 cursor 分页、相关性排序
  和发表日期字段；抓取遵守 Retry-After。主题每次最多 2×100 条，期刊最多 5×100 条。
- 本机有限在线抽样：OpenAlex 的 Ti-6Al-4V ductile fracture 查询返回 HTTP 200；
  Crossref 相同主题的 relevance/cursor 查询返回 HTTP 200。期刊身份查询全部核对完成。
  本机期刊 works 抽样遭遇 429，因此不把身份核验宣称为所有期刊全文抓取成功。
- 索引延迟、缺摘要、拼写、语言、网络限流、分页上限和最终 top-20 都可能造成遗漏。
  日志中的 incomplete coverage / coverage cap 是实际覆盖限制，需要据推送结果继续调优。
