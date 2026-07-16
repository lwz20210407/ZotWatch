# ZotWatch TC4 中心与跨金属方法监测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 深化 LPBF/TC4 主线并保留跨金属可迁移的本构、断裂、损伤和参数反演方法，通过有效提交维持 GitHub Actions 定时监测。

**Architecture:** 保持现有 Zotero 人物画像、候选抓取和 GitHub Actions 不变，仅扩展 `config/sources.yaml`。TC4 直接证据采用材料与力学双门槛；跨金属 LS-OPT/反演方法采用材料、方法和力学对象三重门槛。

**Tech Stack:** YAML、Python 3.11、PyYAML、Pydantic、GitHub Actions

---

### Task 1: 扩展 TC4 精准监测配置

**Files:**
- Modify: `config/sources.yaml`

- [ ] **Step 1: 保持期刊列表不变**

不向 `tracked_venues` 新增宽口径期刊。现有过滤器仍包含非 TC4 材料路径，新增宽口径期刊会放大非 TC4 噪声。

- [ ] **Step 2: 添加五条 TC4 直接查询和两条跨金属方法查询**

在 `queries` 中加入：

```yaml
  - "LPBF Ti-6Al-4V thermoviscoplastic constitutive model thermal softening"
  - "LPBF Ti-6Al-4V anisotropic ductile fracture stress state"
  - "LPBF Ti-6Al-4V load path dependent damage evolution non-proportional loading"
  - "Ti-6Al-4V adiabatic heating shear localization high strain rate"
  - "Ti-6Al-4V inverse parameter identification LS-OPT constitutive fracture model"
  - "LS-OPT inverse parameter identification constitutive fracture damage metal alloy"
  - "thermal softening thermoviscoplastic constitutive model aluminum steel titanium alloy"
```

- [ ] **Step 3: 补充跨金属可迁移方法词**

在 `include_keywords` 中加入热软化、局部化和参数反演相关词；通过后续三重门槛控制 LS-OPT 噪声，而不是把可迁移方法限制为 TC4 独占。

- [ ] **Step 4: 建立 TC4 直接、多金属力学和 LS-OPT 三重门槛**

向 TC4 力学组与多金属力学组补充热软化等概念；新增 LS-OPT 三组规则：

```yaml
金属材料组 AND LS-OPT/反演方法组 AND 本构/断裂/损伤对象组
```

同时将 ML 路径拆分为 ML 方法、力学对象、金属材料三组。

### Task 2: 验证配置质量

**Files:**
- Verify: `config/sources.yaml`
- Verify: `src/settings.py`

- [ ] **Step 1: 验证 YAML 和 Pydantic 配置加载**

Run:

```powershell
C:\Users\yuguo\.conda\envs\py3.11env\python.exe -c "from pathlib import Path; from src.settings import load_settings; s=load_settings(Path('.')); print(len(s.sources.tracked_venues), len(s.sources.queries), len(s.sources.include_keywords))"
```

Expected: 进程退出码为 0，并输出三个正整数。

- [ ] **Step 2: 检查重复项**

Run:

```powershell
C:\Users\yuguo\.conda\envs\py3.11env\python.exe -c "from pathlib import Path; from src.settings import load_settings; s=load_settings(Path('.')).sources; groups={'tracked_venues':s.tracked_venues,'queries':s.queries,'include_keywords':s.include_keywords}; duplicates={k:sorted({x for x in v if v.count(x)>1}) for k,v in groups.items()}; print(duplicates); assert not any(duplicates.values())"
```

Expected: 输出的三个列表均为空，退出码为 0；`tracked_venues` 数量保持为 45。

- [ ] **Step 3: 审查 diff 与敏感信息**

Run:

```powershell
git diff --check
git diff -- config/sources.yaml
git status --short
```

Expected: 无 whitespace error；除计划文件外，仅 `config/sources.yaml` 被修改；不出现 API key、token 或密码。

### Task 3: 提交、推送和 GitHub Actions 验证

**Files:**
- Commit: `config/sources.yaml`
- Commit: `plan/2026-07-16-zotwatch-tc4-monitoring-design.md`
- Commit: `plan/2026-07-16-zotwatch-tc4-monitoring-implementation.md`

- [ ] **Step 1: 提交实现**

Run:

```powershell
git add config/sources.yaml plan/2026-07-16-zotwatch-tc4-monitoring-design.md plan/2026-07-16-zotwatch-tc4-monitoring-implementation.md
git commit -m "feat(config): deepen LPBF TC4 monitoring"
```

Expected: Conventional Commit 成功，worktree 工作树干净，功能分支包含配置与说明修订。

- [ ] **Step 2: 合并到本地 main 并推送**

Run:

```powershell
git -C F:\PythonWoking\zotwatch merge --no-ff codex/zotwatch-tc4-monitoring -m "chore: merge LPBF TC4 monitoring update"
git push origin main
```

Expected: 本地 `main` 以 merge commit 整合功能分支，远端 `main` 更新成功，并因 `push` 触发 `Weekly Watch & RSS`。

- [ ] **Step 3: 核验 workflow**

查询 GitHub workflow API，确认 `Weekly Watch & RSS` 的状态为 `active`，并检查最新一次运行由新提交触发且完成或正在运行。若运行失败，仅诊断并报告，不改动 Secrets。
