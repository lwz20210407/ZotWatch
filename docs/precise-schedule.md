# 准点推送：用外部定时器触发周报

## 为什么需要

GitHub 的 `schedule` 不保证准点。官方文档写明高负载时"can be delayed"，严重时直接丢弃；
2026 年 6 月 GitHub Actions 产品经理在社区讨论 #196910 里说，这是全局限流，换哪个时间都差不多。
本仓库实测：2026-09-24 晚 5 小时 09 分，2026-10-01 晚 7 小时 08 分。

`workflow_dispatch`（通过 API 手动触发）几秒内就会启动。所以主触发交给外部定时器，
GitHub 自己的三条 cron 只作兜底。

## 为什么不会重复发

workflow 里的 guard 按北京时间 ISO 周去重：

- 外部定时器触发时带 `trigger=scheduled`，和 GitHub cron 一样受门控。
- 本周第一次成功投递后，写入 `data/watch-state/scheduled-deliveries.json`。
- 之后再来的触发（兜底 cron、第二个外部作业）读到这条记录，几秒内退出。
- 手动运行（`trigger=manual`）和 dry run 不受门控，也不写记录，不会吞掉周四的推送。

## 配置步骤（约 10 分钟，需要你本人操作）

### 1. 建一个只能触发 workflow 的 token

GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token：

- Resource owner：`lwz20210407`
- Repository access：Only select repositories → `ZotWatch`
- Repository permissions：只开 **Actions: Read and write**（Metadata: Read 会自动带上）
- Expiration：按需。注意 token 一年不用会被 GitHub 自动吊销；每周调用一次就不会。

不要用 classic token，也不要给 Contents 权限。这个 token 泄露后，最多能触发、取消或删除 Actions 运行，
推不了代码、读不了 secrets。

### 2. 在 cron-job.org 建两个作业

注册 https://console.cron-job.org （邮箱注册，免费）。每个作业填：

| 字段 | 值 |
|---|---|
| URL | `https://api.github.com/repos/lwz20210407/ZotWatch/actions/workflows/daily_watch.yml/dispatches` |
| 时间 | 自定义：每周四，**15:00**；时区 **Asia/Shanghai**（整个流程约 30 分钟，邮件约 15:30 到） |
| Request method | POST |
| Headers | `Authorization: Bearer github_pat_…`（Bearer、空格、token 本身，不加尖括号）<br>`Accept: application/vnd.github+json`<br>`X-GitHub-Api-Version: 2022-11-28`<br>`Content-Type: application/json` |
| Body | `{"ref":"main","inputs":{"dry_run":"false","trigger":"scheduled"}}` |
| 通知 | 勾选：执行失败时、失败后恢复时、作业被自动停用时 |

第二个作业完全相同，只把时间改成 **周四 17:00**，作为重试。15:00 那次已经投递的话，它几秒内就退出。

**当前状态（2026-10-05 已配置并测试）**：两个作业 15:00 / 17:00 已建好；token 只有 ZotWatch 仓库的
Actions 读写权限。测试运行返回 204，GitHub 在请求的同一秒建好运行（run 37280475729），guard 判定正确。

注意：

- input 的值一律写成字符串（`"false"`），写成布尔值会返回 422。
- 成功时 GitHub 返回 **204**。
- **不要在控制台对这两个作业点 "Test run"**：本周还没投递时，会真的发出邮件。
  想测试，就把 Body 里的 `dry_run` 改成 `"true"` 再点。

### 3. 先测一次（不发邮件）

在本机执行，`$TOKEN` 换成你的 token：

```bash
curl -i -X POST https://api.github.com/repos/lwz20210407/ZotWatch/actions/workflows/daily_watch.yml/dispatches -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" -d '{"ref":"main","inputs":{"dry_run":"true","trigger":"scheduled"}}'
```

返回 `HTTP/2 204`，且 Actions 页面几秒内出现一次新运行，就是通了。

### 4.（可选）到点没收到就告警

在 https://healthchecks.io 建一个检查：Cron `30 15 * * 4`，时区 Asia/Shanghai，Grace 2 小时。
把它给的 ping 地址存成仓库 secret `HC_PING_URL`。workflow 在每周投递成功后会 ping 它；
到点没收到 ping，不管原因是外部定时器、GitHub、任务失败还是邮件发送失败，它都会给你发邮件。

## 残余风险

- cron-job.org 自己也不承诺准点，失败不重试，所以配两个作业，再加 GitHub cron 兜底。
- GitHub Actions 整体故障时，外部触发也救不了；第 4 步的心跳至少能让你及时知道。
- 周四 15:00 之前如果手动跑过一次**正式**运行，并且选了 `trigger=scheduled`，当周自动推送会被跳过。
  手动运行默认是 `manual`，不受影响。
