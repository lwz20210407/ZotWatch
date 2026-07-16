# ZotWatch LPBF/TC4 精准深化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 深化 LPBF/TC4 钛合金博士主线的期刊、复合查询和主题词配置，并通过有效提交维持 GitHub Actions 定时监测。

**Architecture:** 保持现有 Zotero 人物画像、候选抓取、双重主题过滤和 GitHub Actions 不变，仅扩展 `config/sources.yaml` 的声明式监测先验。新增词同时写入 TC4 主题门槛，确保新检索结果仍需满足材料相关性和力学问题相关性。

**Tech Stack:** YAML、Python 3.11、PyYAML、Pydantic、GitHub Actions

---

### Task 1: 扩展 TC4 精准监测配置

**Files:**
- Modify: `config/sources.yaml`

- [ ] **Step 1: 添加三种高相关期刊**

在 `tracked_venues` 中加入：

```yaml
  - Journal of Dynamic Behavior of Materials
  - Metallurgical and Materials Transactions A
  - Journal of Manufacturing Processes
```

- [ ] **Step 2: 添加五条材料限定复合查询**

在 `queries` 中加入：

```yaml
  - "LPBF Ti-6Al-4V thermoviscoplastic constitutive model thermal softening"
  - "LPBF Ti-6Al-4V anisotropic ductile fracture stress state"
  - "LPBF Ti-6Al-4V load path dependent damage evolution non-proportional loading"
  - "Ti-6Al-4V adiabatic heating shear localization high strain rate"
  - "Ti-6Al-4V inverse parameter identification LS-OPT constitutive fracture model"
```

- [ ] **Step 3: 补充缺失词形和精准研究概念**

在 `include_keywords` 中加入：

```yaml
  - "L-PBF"
  - "SLM"
  - "selective laser melting"
  - "additively manufactured"
  - "thermo-viscoplastic"
  - "thermoviscoplastic"
  - "thermal softening"
  - "adiabatic heating"
  - "shear localization"
  - "anisotropic ductile fracture"
  - "load-path-dependent damage"
  - "load path dependent damage"
  - "inverse parameter identification"
  - "LS-OPT"
```

- [ ] **Step 4: 让新增概念进入 TC4 力学问题门槛**

向 `required_keyword_groups` 的第二组，以及 `required_any_group_sets` 第一套 TC4 规则的第二组，加入：

```yaml
"thermo-viscoplastic", "thermoviscoplastic", "thermal softening", "adiabatic heating", "shear localization", "anisotropic ductile fracture", "load-path-dependent damage", "load path dependent damage", "inverse parameter identification", "LS-OPT"
```

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

Expected: 输出的三个列表均为空，退出码为 0。

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
- Commit: `plan/2026-07-16-zotwatch-tc4-monitoring-implementation.md`

- [ ] **Step 1: 提交实现**

Run:

```powershell
git add config/sources.yaml plan/2026-07-16-zotwatch-tc4-monitoring-implementation.md
git commit -m "feat(config): deepen LPBF TC4 monitoring"
```

Expected: Conventional Commit 成功，工作树干净，本地 `main` 领先远端两个提交。

- [ ] **Step 2: 推送到远端 main**

Run:

```powershell
git push origin main
```

Expected: 远端 `main` 更新成功，并因 `push` 触发 `Weekly Watch & RSS`。

- [ ] **Step 3: 核验 workflow**

查询 GitHub workflow API，确认 `Weekly Watch & RSS` 的状态为 `active`，并检查最新一次运行由新提交触发且完成或正在运行。若运行失败，仅诊断并报告，不改动 Secrets。
