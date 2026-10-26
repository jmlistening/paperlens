# PaperLens 前沿 arXiv 推送

可选功能：每天抓取符合兴趣的前沿 arXiv 论文，排序、写中文速览，生成 links-first 摘要邮件，并可在首次人工批准后通过 163/QQ 邮箱发送。

默认关闭。代码与测试都不会自动发信；真正的首次发送必须由本人在场批准。

---

## 1. 安全边界

- 授权码只放在项目根目录的 `.env`，不要写进 `push_config.json`、脚本、日志或 Git。
- `.env` 已被 `.gitignore` 排除；仓库里只有 `.env.example` 占位符。
- 状态、日志、Streamlit 页面只显示掩码地址（如 `u***@163.com`），不显示授权码。
- 预览（preview）不连接 SMTP，也不写投递账本。
- `SMTP accepted` 只说明服务器接收了邮件，不代表已进入收件箱；不要把它写成“已送达”。
- 不要删除 `data/delivery.sqlite3`：它是防止重复发信的唯一权威账本。

---

## 1.1 简单订阅页（默认入口）

启动 `app.py` 后默认就是「每日推送」页；日常订阅不需要编辑 JSON。

- **主题**：在「我想关注什么」写一段话或每行一个主题。预览/发送时生成并保存检索组；生成失败不会保存一半设置。
- **邮箱**：填写后由预览/发送保存，保存后只显示掩码；换邮箱会清除验证并自动关闭自动推送。
- **篇数**：1–10（含 1/3/5/10 快捷键），写入 `max_per_digest` 与 `top_k`；预览、手动、自动共用同一个值。
- **时间**：本地 `HH:MM`，写入 `time_local`；自动推送开启时同步重注册计划任务，失败会明确提示。
- **预览**：「先预览这周会推什么」会抓取、排序并调用模型，但**永不连接 SMTP**；自动推送关闭时也可用。
- **首次开启**：填写邮箱 → 点「先预览这周会推什么」保存邮箱与主题（页面没有单独的保存按钮，保存后才会出现测试邮件区；「现在推送今天的」在确认后也会保存，但首次开启需先完成邮箱验证）→「发送测试邮件」→ 在收件箱确认收到 → 勾选「我收到了，开启推送」。成功后才开启自动推送；任一步失败都保持关闭。
  只想手动推送时，可在同一测试邮件区改点「仅验证邮箱（不开启自动推送）」：它只写邮箱验证和首封批准，**不安装计划任务、不调用自动推送开关、不改 `enabled`/`paused`**。已验证但尚未批准首封时，按钮显示为「完成首次发送批准（不开启自动推送）」，无需再发一次测试邮件。
  已有 `.env` `MAIL_TO` 的旧安装（还没有 `data/push/ui_state.json`）会把该地址当作已验证收件人，所以测试邮件区可能在通过简单页保存之前就出现；迁移到简单页后建议重发一次测试邮件，完成一次重新验证。
- **手动推送**：「现在推送今天的」是显式手动发送，**自动推送关闭时仍可用**；需勾选确认，且要求邮箱已验证、首封已批准。手动发送不会改变 `enabled`/`paused`，**同一天可以多次手动推送**。改了主题后再预览/手动推送会使用最近 `catchup_days`（默认 7）天的窗口，并重新打开该窗口内 `evaluated`/`skipped`/`selected` 的论文供新主题重新排序；`delivered`/`uncertain` 永不重开，投递账本仍会阻止重复发送。自动推送不使用这个宽窗口，继续按 watermark 增量抓取。
- **推送历史**：「高级 / 研究模式」→「推送历史」只读展示真实 `sent` 记录；不列预览，不含邮箱/密钥。

需要逐组编辑检索词、修改服务商、暂停/恢复、测试登录或查看批准状态时，进入「高级 / 研究模式」→「推送设置」或「精确兴趣组」。

---

## 2. 配置

1. 从 `.env.example` 复制一份 `.env`（不要提交）：

   ```bat
   copy .env.example .env
   ```

2. 填写 DeepSeek 与邮箱字段。`push_config.json` 控制行为；简单页会写 `time_local` 与 `max_per_digest`，其余字段通常不用手工改。默认值如下：

   ```json
   {
     "enabled": false,
     "paused": false,
     "quiet_mode": false,
     "frequency": "daily",
     "time_local": "08:30",
     "provider": "163",
     "attach_pdfs": false,
     "window_hours": 72,
     "overlap_hours": 24,
     "catchup_days": 7,
     "fetch_pool_limit": 40,
     "rank_max_candidates": 20,
     "max_per_digest": 3,
     "max_per_primary_category": 2,
     "min_new_papers": 1,
     "auto_pause_after_failures": 3
   }
   ```

3. 检查配置（不联网、不发信）：

   ```bat
   python push_ctl.py --config push_config.json doctor
   ```

---

## 2.1 个性化定义“前沿”

候选池由**你自己的检索条件**决定，不是由模型猜一个“前沿方向”：

- active profile（`data/push/profile.json`，已被 gitignore）的 `search.groups`：每组英文检索词（ANY / REQUIRE 组合）。
- `push_config.json` 的 `search_categories`：可选 arXiv 分类（如 `cs.CL`、`cs.IR`）。
- 只有命中这些查询组/分类的论文才会进入排名；模型只负责在候选池内排序、写理由和摘要。
- `topics`、`preferred_methods`、`less_interested` 只影响排序/解释，不会扩大或缩小候选池。
- 收藏、已读、忽略会作为排序反馈；它们不会把未命中查询组的论文加进来。
- 想更“严格”地只按自己的词推送：把 `min_score` 调低到 0、`top_k` 设为你愿意读的篇数；模型不会自行添加方向。

编辑方式：

- 日常在默认「每日推送」页写主题即可；预览/发送时会生成并保存 `search.groups`。
- 需要逐组核对时用「高级 / 研究模式」→「精确兴趣组」（`advanced/interests.py`）：页面会显示本轮实际执行的查询串，保存前可核对。
- 公开示例是 `interests.example.json`（只读）；active profile 写入 `data/push/profile.json`。`interests.json` 只是本地旧文件的兼容回退，已被 ignore，不要提交或手工维护真实兴趣。

---

## 3. 163 邮箱授权码

163 不接受邮箱登录密码作为 SMTP 密码，必须使用客户端授权码。

1. 登录 163 邮箱网页版。
2. 打开“设置 → POP3/SMTP/IMAP”，开启 SMTP 服务。
3. 按提示生成客户端授权码；授权码只显示一次。
4. 写入 `.env`（不要加引号、不要提交）：

   ```dotenv
   SMTP_HOST=smtp.163.com
   SMTP_PORT=465
   SMTP_SECURITY=ssl
   SMTP_USER=your_account@163.com
   SMTP_PASSWORD=你的163客户端授权码
   MAIL_TO=your_account@163.com
   ```

`provider` 保持 `163`。`SMTP_USER` 必须是完整邮箱地址，`MAIL_TO` 是收件地址（可与发件地址相同）。

## 4. QQ 邮箱授权码

QQ 使用 16 位授权码：

1. 登录 QQ 邮箱网页版。
2. 打开“设置 → 账户 → POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV 服务”。
3. 开启 SMTP 服务并生成授权码。
4. `.env` 使用：

   ```dotenv
   SMTP_HOST=smtp.qq.com
   SMTP_PORT=465
   SMTP_SECURITY=ssl
   SMTP_USER=your_account@qq.com
   SMTP_PASSWORD=16位QQ授权码
   MAIL_TO=your_account@qq.com
   ```

5. 把 `push_config.json` 的 `provider` 改为 `qq`（或使用页面“服务商”下拉框并点击“保存服务商”）。

也支持 `provider=custom`，此时需在 `.env` 明确写 `SMTP_HOST`、`SMTP_PORT`、`SMTP_SECURITY`（`ssl` 或 `starttls`）。`none` 只允许 localhost。

---

## 5. 首次真实发送（必须本人在场）

严格按顺序执行；不要把这些步骤写进自动任务或 CI。

**推荐：简单页流程**

1. 填授权码：本人把授权码写入 `.env`，配置好 SMTP 字段。
2. 打开「每日推送」，填写收件邮箱。
3. 点「先预览这周会推什么」保存邮箱与主题；页面没有单独的保存按钮，保存后只显示掩码（「现在推送今天的」在确认后也会保存，但首次开启需先完成邮箱验证）。预览会抓取 arXiv、调用模型，但不会发信，自动推送关闭时也能用。
4. 保存邮箱后，页面出现「发送测试邮件」；点它。这封邮件没有推荐内容，但会真实发送。
5. 去收件箱（包括垃圾箱）确认收到。只有本人确认收到后才继续。
6. 勾选「我收到了，开启推送」。成功会同时写入验证、完成首封批准、安装每日计划任务并开启自动推送；任一步失败都会保持关闭。
7. 想立即发一封：勾选确认后点「现在推送今天的」。自动推送关闭时也能手动发送，手动发送不会打开自动开关。

**等价 CLI 流程（本人在场）**

1. **只登录**（连接、认证、NOOP，不发信）：

   ```bat
   python push_ctl.py test-email
   ```

2. **发测试邮件**（会真实发送一封无推荐内容的邮件，必须显式确认）：

   ```bat
   python push_ctl.py test-email --send --confirm
   ```

3. **确认收到**后**批准首次发送**：

   ```bat
   python push_ctl.py approve-first-send --confirm
   ```

4. **生成真实预览**（不联网发信，但会抓取 arXiv 并调用模型）：

   ```bat
   python push_run.py --preview
   ```

   预览文件在 `data/push/runs/<run_id>/digest_preview.html`。检查标题、链接、收件人。
5. **临时开启自动推送总开关**（CLI `--send` 不会绕过 `enabled`/`paused` 门禁，所以必须先开启）：

   ```bat
   python push_ctl.py enable
   ```

6. **首次真实发送**（仅本人执行）：

   ```bat
   python push_run.py --send --confirm
   ```

   发送结束后可用 `push_ctl.py disable` 关闭自动推送；`--send` 本身不会改动 `enabled`/`paused`。

计划任务调用的是 `push_run.py --scheduled`，它无法绕过首次批准；未批准时状态为 `send_not_approved`，不会发信。CLI `--send` **不是**简单页的「现在推送今天的」：它不传 `manual=True`，仍要求 `enabled=true` 且 `paused=false`（否则返回 `disabled`/`paused`），也不会修改这两个开关；在自动推送关闭时仍能真实发送的只有简单页的手动推送。

定时槽位按「本地日期 + `time_local`」计算：同一时间每天只成功发一次；`sent`/`already_sent`/`uncertain` 保留该时间槽位，`no_digest`、失败、仅预览等没有真正发信的尝试会释放该时间槽位。**同一天改时间并确认会创建新槽位，可以再次测试**；投递账本只允许未发送过的论文进入邮件，因此不会重复同一篇。简单页会显示「上次定时尝试」的时间与结果。

---

## 6. 预览、启停与暂停

```bat
rem 启用 / 关闭总开关（启用前会校验 SMTP 环境变量）
python push_ctl.py enable
python push_ctl.py disable

rem 暂停 / 恢复（恢复同时清除自动暂停标记）
python push_ctl.py pause
python push_ctl.py resume

rem 查看状态与历史（邮箱已掩码，不含授权码）
python push_ctl.py status --json
python push_ctl.py history

rem 手动预览（永不发信）
python push_run.py --preview
```

- `enabled=false`：自动 `push_run --scheduled` 在任何网络、模型、SMTP 调用前退出，状态 `disabled`；CLI `push_run.py --send` 同样返回 `disabled`，只有简单页的显式手动推送不受此开关影响。
- `paused=true`：自动推送直接跳过，状态 `paused`；简单页手动推送仍可显式执行，CLI `--send` 仍返回 `paused`。
- `quiet_mode=true`：只生成预览，状态 `quiet_preview`，不发信。
- 默认「每日推送」页提供主题、邮箱、篇数（1–10）、时间、预览、手动推送和「每天自动推送」开关。
- 「高级 / 研究模式」→「推送设置」（`advanced/push_settings.py`）提供总开关、暂停/恢复、服务商、预览、测试登录、首次批准和最近运行。
- 「高级 / 研究模式」→「精确兴趣组」（`advanced/interests.py`）逐组编辑检索词并核对实际查询串。
- 「高级 / 研究模式」→「推送历史」（`advanced/history.py`）只读展示真实 `sent` 记录；预览不会冒充已推送。
- 简单页修改时间会写 `time_local`；自动推送开启时同步重注册计划任务，失败会提示而不是假装生效。
- 简单页修改篇数会写 `max_per_digest` 与 profile `top_k`；实际可推论文不足时按实际数量发送。

Streamlit 页面只应在受信任的本机使用；启动时绑定 `127.0.0.1` 并关闭 usage stats。

---

## 7. Windows 任务计划程序安装与移除

脚本：

- `scripts/run_push.bat`：使用 `PAPERLENS_PYTHON`（未设置时回退到 `python`；计划任务里建议显式设置成项目解释器），切换到**该批处理所在的 checkout**（`%~dp0..`）并执行 `push_run.py --scheduled`，把退出码传回任务计划程序。因此从 worktree 注册的任务会使用该 worktree 的配置，不会落到主仓库。任务窗口会显示中文进度（抓取、排序、摘要、发送、结果），结束后停留 6 秒再关闭。
- `scripts/register_push_task.ps1`：注册任务 `PaperLens Frontier Push`。
- `scripts/unregister_push_task.ps1`：安全移除该任务。

首次开启自动推送时，简单页会安装该任务；自动推送开启期间修改时间会按新 `time_local` 重注册。安装失败时不会开启自动推送，并在页面上显示简短错误。

先做 `-WhatIf` 预览（不会真正注册或删除）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_push_task.ps1 -WhatIf
powershell -ExecutionPolicy Bypass -File scripts\unregister_push_task.ps1 -WhatIf
```

确认后安装 / 移除：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\register_push_task.ps1
powershell -ExecutionPolicy Bypass -File scripts\unregister_push_task.ps1
```

任务设置：

| 设置 | 值 |
|---|---|
| 任务名 | `PaperLens Frontier Push` |
| 触发 | 每天 `time_local`（默认 08:30） |
| 登录类型 | 当前用户，仅交互登录（Interactive） |
| StartWhenAvailable | 是（错过后尽快补跑） |
| MultipleInstances | `IgnoreNew`（不并行） |
| RestartCount | 2 |
| RestartInterval | 10 分钟 |
| ExecutionTimeLimit | 2 小时 |

脚本不保存任何口令：任务只以当前登录用户身份运行，Python 在运行时读取 `.env`。因此**必须保持该用户已登录**；注销或关机期间任务不会运行。

卸载不会删除 `data/push/`、`push_config.json` 或 `.env`。

---

## 8. on-time 与 catch-up

- **on-time**：到点时用户已登录、机器可用，任务按计划时间启动。
- **catch-up**：到点时机休眠/关机/任务计划程序未运行，`StartWhenAvailable` 会在条件恢复后补跑一次。
- 如果错过超过抓取窗口，`push_fetch` 使用 `start = max(watermark - overlap_hours, now - catchup_days)`；首次运行至少回看 `initial_lookback_days=7`（默认）与 `window_hours` 中的较大者。
- 简单页的预览和手动推送是“下一封手动发送”的预测，固定使用 `[now - catchup_days, now]` 宽窗口，并在抓取前把窗口内未送达的 `evaluated`/`skipped`/`selected` 论文改回 `queued`，以便新主题当天重新排序；计划任务/自动推送不传这个参数，仍用上面的增量窗口。
- 429 限流时不得静默跳天：本次记录 `rate_limited`，watermark 不前进，冷却结束后由下一次任务重试。
- 报告必须区分 on-time 与 catch-up；查看 `data/push/runs/<run_id>/run.json` 的 `started_at` 与实际计划时间，不能把补跑说成准时。

任务计划的每日 slot 用本地日期做幂等键：同一天重复触发不会重复发送。

---

## 9. 失败恢复

每次运行写 `data/push/runs/<run_id>/`：

- `run.json`：`status`、`counts`、`errors`、`paper_keys`
- `digest_preview.html` / `digest_preview.txt`：邮件预览
- `pipeline.log`：简要日志

常见状态与处理：

| 状态 | 含义 | 处理 |
|---|---|---|
| `disabled` / `paused` | 门禁直接退出，无网络调用 | 需要时 `enable` / `resume` |
| `busy` | 已有运行持有锁 | 等待；不要手动并发启动 |
| `no_digest` | 没有新的合格论文 | 正常，无需处理 |
| `config_error` | `.env` 或配置缺项 | `doctor` 查看问题，补全 `.env` |
| `rate_limited` | arXiv 429 | 等待本地 15 分钟冷却；watermark 未前进 |
| `cooldown` | 冷却期内启动 | 等冷却结束 |
| `failed` | 抓取/排序/摘要失败 | 看 `run.json.errors` 与日志；修复后重跑 |
| `uncertain` | SMTP 提交后结果不明 | 先查邮箱与 `send_digest.py --history`，人工核实 |
| `send_not_approved` | 未批准首次发送 | 按第 5 节人工批准 |
| `auto_paused` | 已自动暂停 | 排查后 `push_ctl.py resume` |

连续真实发送失败会触发自动暂停（`auto_pause_after_failures`）。`uncertain` 绝不能直接重发：

```bat
python send_digest.py --history
python send_digest.py --resolve <attempt_id> --result accepted --note "已在邮箱确认" --confirm-stopped
```

`pending` / `sending` / `uncertain` 都会阻止同一论文再次自动发送；`failed` 才允许重试。

---

## 10. 排障速查

```bat
rem 配置检查（不联网）
python push_ctl.py doctor

rem SMTP 只登录（不发信）
python push_ctl.py test-email

rem 最近投递账本
python send_digest.py --history

rem 查看任务计划程序中的任务
schtasks /Query /TN "PaperLens Frontier Push" /V /FO LIST
```

- **登录失败**：确认开的是 SMTP 服务、用的是授权码而不是登录密码，账号与 `SMTP_USER` 一致。
- **能登录但收不到**：先查垃圾箱；163/QQ 对新发件人可能风控。必要时把发件地址加入白名单，再发一次 `test-email --send --confirm`。
- **任务没跑**：确认用户已登录、任务处于启用状态、`run_push.bat` 路径存在。任务动作里的 `run_push.bat` 会使用它自己所在 checkout 的 `push_config.json`；如果任务指向 worktree 但批处理跑的是主仓库配置，会出现“界面已开启、任务时间到了却不发信”。
- **补跑重复**：同一本地日期 slot 幂等；若确需重跑，请先检查 `data/delivery.sqlite3`，不要删除账本。
- **状态里有完整邮箱**：属于缺陷；请检查是否绕过 `push_status.py` 直接打印了原始环境变量或 run 记录。

所有真实发送、批准和任务注册都由用户本人执行。

---

## 11. 隐私与公开仓库

- 收件人只保存在 `data/push/ui_state.json`（`data/` 已被 gitignore）；`push_config.json`、日志、推送历史和状态 JSON 都不含完整邮箱。
- SMTP 授权码与 API key 只在 `.env`；仓库只提交 `.env.example` 占位符。
- active 兴趣 profile 在 `data/push/profile.json`；公开示例是 `interests.example.json`。`interests.json`、`docs/handoff/`、旧中文说明/验收/接续文档均已 ignore 且不再 tracked。
- 测试 fixture 只使用 `example.com` 等占位地址和明显的假 key/授权码。
- 提交或发布前运行：

  ```bat
  python scripts/check_public_repo.py
  ```

  默认只扫描 tracked 文件；发现真实邮箱、`sk-` key 或非占位 `SMTP_PASSWORD=` 时退出非零，且只输出路径/掩码，不打印完整值。

- 可选只读历史审计：

  ```bat
  python scripts/check_public_repo.py --history
  ```

  它报告历史中仍含旧个人 profile/handoff 文档的路径与 commit；**不会重写历史、创建分支或推送**。是否公开旧历史由仓库所有者单独决定。
