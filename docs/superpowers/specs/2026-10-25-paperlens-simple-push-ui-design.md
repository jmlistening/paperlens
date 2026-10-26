# PaperLens 简单推送界面设计

状态：待确认。确认后才进入实施计划与代码修改。

## 意图

当前 Streamlit 入口是工程/研究控制台：问答、证据、执行记录、评测表、推送设置、兴趣查询组都在一层。个人使用时真正高频的动作只有三件：

1. 输入“我想关注什么”；
2. 输入“推送到哪个邮箱”；
3. 先预览这周会推什么，再决定是否每天推送。

因此把默认入口改成订阅向导，把问答/证据/评测/精确兴趣组收进一个不显眼的“高级 / 研究模式”入口。底层能力保留，不删除。

## 目标

- 大屏只有一个简单页面：主题输入、邮箱输入、预览按钮、推送开关、推送时间。
- 推送开关只负责“推 / 不推”；不负责预览、不修改主题或邮箱。
- 主屏可选推送时间（默认 08:30），并按该时间更新 Windows 每日任务。
- 主屏可选每次推送篇数（1–10）；今天想 1 篇、明天想 10 篇都可以改，预览/手动立即使用，自动推送使用最近保存的值。
- 自动推送关闭时，仍可手动“现在推送今天的”一次；手动发送不改变自动开关。
- 第一次开启推送必须强制发送一封测试邮件并确认收到；之后一键开关。
- 预览永远不发邮件，推送关闭时也能预览（dry-run）。
- 主题输入能够改变候选池，而不是只改变排序。
- 高级入口收纳现有页面；默认导航看不到研究/评测内容。
- 不把邮箱、SMTP 授权码写进 git、日志或状态 JSON。

## 非目标

- 不做多用户账号、登录或权限系统。
- 不重写现有 push 排序/摘要算法；只为简单输入生成检索组。
- 不删除问答、证据、评测、论文库、精确兴趣组等模块。
- 不新增邮件服务商；继续使用现有 163/QQ/custom preset。
- 不把 citation-level 指标包装成 answer accuracy/faithfulness。

## 反目标

- 预览时发送邮件。
- 在界面显示或记录 SMTP 密码/授权码。
- 把收件人邮箱提交到 git 或原始状态 JSON。
- 未确认测试邮件就允许开启推送。
- 手动发送绕过测试邮件/首封批准，或悄悄打开自动推送。
- 改了主题但候选池没有变化。
- 默认大屏出现评测表、token、trace、evidence 等研究内容。

## 用户流程

1. 打开 app，直接是简单订阅页。
2. 输入“我想关注什么”（中文一段话或几行主题）、收件邮箱，选择这次推送篇数（1–10）。
3. 点“先预览这周会推什么”：
   - 保存主题/邮箱到本地状态；
   - 把主题转成 1–3 组英文 arXiv 检索条件；
   - 抓取 arXiv、排序、生成摘要；
   - 展示当前篇数（默认 3，最多 10）的论文卡片；
   - 不发送任何邮件。
4. 在预览下方操作“每天推送”开关和时间：
   - 关闭 → 开启：如果当前邮箱尚未验证，先进入一次测试邮件流程；
   - 测试邮件发送成功且用户确认收到后，记录该邮箱已验证，安装/确认每日任务（如未安装），开启推送；
   - 之后同一邮箱的开/关是一键操作；
   - 修改邮箱会清除验证状态，必须重新测试；
   - 时间选择（HH:mm）写入配置并更新每日任务；自动开关关闭时也保存时间，下次开启生效。
5. 点“现在推送今天的”可手动发送一次：
   - 自动推送关闭时也可用；不会改变自动开关；
   - 真实发送前仍要求当前邮箱已验证、首封已批准，并弹出一次确认；
   - 如果刚生成的预览与当前主题/邮箱/窗口一致，发送这份预览；否则重新生成当天 digest 后发送；
   - 发送后推进生产推送状态，避免后续自动推送重复发送。
6. 点底部很轻的“高级 / 研究模式”进入旧页面。

## 主屏布局

```text
PaperLens
每天挑几篇你可能关心的 arXiv 论文

我想关注什么
┌──────────────────────────────────────────────┐
│ 例如：RAG、Agent、检索证据与回答可靠性       │
└──────────────────────────────────────────────┘

推送到哪个邮箱
┌──────────────────────────────┐
│ you@example.com              │
└──────────────────────────────┘
当前收件人：a***@163.com（如已配置）

推送篇数  [−] 3 [+] 篇/次     ← 屏幕可见，随手改；1/3/5/10 快捷

[ 先预览这周会推什么 ]   [ 现在推送今天的 ]
   ↑ 不发信                 ↑ 真实发送，先确认

── 预览：当前篇数的论文卡片 ──
标题 / 一句推荐理由 / arXiv 链接

每天推送   [ 开 / 关 ]   时间 [ 08:30 ]   ← 开关只控制自动发送
下次发送：明天 08:30 · 状态：已开启/已关闭

高级 / 研究模式               ← 轻量入口
```

主屏不显示：schema、quote、tokens、latency、evidence、trace、rerank、评测表、查询组编辑器。

## 推荐按钮/控件文案

| 位置 | 文案 | 作用 |
|---|---|---|
| 主题 | 我想关注什么 | 中文/英文主题，一段话或几行 |
| 邮箱 | 推送到哪个邮箱 | 收件人；保存后显示掩码 |
| 篇数 | 每次推送几篇 `[− 3 +]` | 主屏可见的 stepper/滑块，1–10；带 1/3/5/10 快捷 |
| 主按钮 | 先预览这周会推什么 | 只生成预览，不发信 |
| 次按钮 | 现在推送今天的 | 真实发送，弹确认；自动关闭也可用 |
| 自动开关 | 每天自动推送 `[开/关]` | 只控制自动发送 |
| 时间 | 每天几点 `[08:30]` | 本地时间；自动开启时更新任务 |
| 首次验证 | 发送测试邮件 / 我收到了，开启推送 | 第一次强制；之后一键 |
| 高级入口 | 高级 / 研究模式 | 收纳旧页面 |

## 推送开关语义

- 开关只写 `enabled`/`paused`（以及首次验证/计划任务状态），不触发预览、不修改主题/邮箱。
- 关闭：`enabled=false`；定时任务直接 no-op；不发送。不会清除 `auto_paused`/`quiet_mode` 等安全状态。
- 开启（同一已验证邮箱）：写 `enabled=true, paused=false`，立即生效；若 `auto_paused=true`，**不**自动清除，显示“发送失败已自动暂停，请先去高级检查”；若 `quiet_mode=true`，显示“静默模式只预览”，不开启。
- 开启（未验证邮箱）：不直接开启；显示测试邮件步骤。
- 修改邮箱：清除验证、自动关闭自动推送；必须重新测试后才能再次开启。
- 主屏邮箱输入框里有未保存的新地址时，自动开关禁用，提示“先保存/验证邮箱”；开关只作用于已保存的收件人。
- SMTP 凭据缺失：不开启；显示一句“需要先在高级设置里配置邮箱授权码”的链接。
- 测试邮件失败：不开启；显示错误类型，不显示 SMTP 密码/响应体。
- 计划任务未安装：首次开启时尝试用现有 `scripts/register_push_task.ps1` 安装；失败则不开启，显示重试/高级入口。

## 时间选择

- 主屏提供一个时间选择（HH:mm），默认读取 `push_config.json` 的 `time_local`（当前 08:30）。
- 保存时写回 `time_local`；按本地时间解释。
- 自动推送开启且计划任务已安装时，用 `scripts/register_push_task.ps1 -Time HH:mm` 重新注册任务。
- 自动推送关闭时只保存时间，不安装/改任务；下次开启时按新时间安装。
- 任务更新失败时保留配置值，但主屏明确显示“计划任务时间未更新”，不假装新时间已生效；高级页可重试。
- 手动推送不受这个时间影响。

## 推送篇数

- 主屏明显位置提供 1–10 的 stepper/滑块（`− 3 +`），旁边 1/3/5/10 快捷；不是高级页里的隐藏设置。
- 默认取当前 `push_config.max_per_digest`（当前 3）。
- 写入 `push_config.max_per_digest` 和 `interests.json` 的 `top_k`，两者保持一致；实际入选数取两者较小值和可用论文数。
- 预览和手动推送立即使用当前值；自动推送使用最近保存的值。
- “今天 1 篇、明天 10 篇”通过每次发送前/后改这个数字实现；v1 不做按日期预排的日历。
- 如果某个主题当天只有 2 篇合格论文，就推 2 篇并显示“只找到 2 篇”，不凑数。
- 改篇数不发送、不触发预览；只影响下一次预览/发送。

## 手动推送

- 主屏有独立的“现在推送今天的”按钮；这是真实发送，不是自动开关。
- 自动推送关闭（`enabled=false` 或 `paused=true`）时仍然可用；手动发送不改变自动开关，也不要求安装计划任务。
- 发送前仍要求：当前邮箱已验证、首封已批准、用户在确认框中确认；不绕过测试邮件/首封门禁。
- 如果刚生成的预览与当前主题、邮箱、时间窗口一致，发送这份预览；否则重新跑当天 digest 再发送。
- 手动发送使用生产推送状态/watermark，发送后推进状态，避免后续自动推送重复发送同一批论文。
- 确认框显示掩码收件人和“会真实发送邮件”；失败只显示状态，不自动重试。
- 手动发送成功后，主屏显示“今天已手动推送 N 篇”，自动开关保持原状态。

## 收件人邮箱与隐私

- 只保留**一个当前收件人**和它的 `verified_at`，不做“已验证地址白名单”。换邮箱（包括换回旧地址）都清除验证，必须重测。
- 收件人保存在 `data/push/ui_state.json`（`data/` 已被 gitignore），不写入 `push_config.json`/`.env`/git。
- `MAIL_TO` 解析：如果 `ui_state.json` 存在，只用它的收件人并要求已验验证；**不**回退到 `.env`。只有从未使用简单页（state 不存在）时，才兼容 `.env` 的 `MAIL_TO`。
- 已保存邮箱在自动推送开启时被修改：先自动关闭自动推送（`enabled=false`），清除验证，保存新地址，提示“请测试后重新开启”。
- 输入框里有未保存的新地址时，自动开关禁用；开关只作用于已保存的收件人。
- 界面保存后显示掩码（如 `a***@163.com`）；输入框允许替换。
- SMTP 授权码继续只从 `.env` 读取；界面不读取、不显示、不记录。
- 测试邮件使用现有 `push_ctl.test_email(send=True, confirm=True)`，通过 env override 把 `MAIL_TO` 设为当前收件人；不写入 digest 投递账本。
- 测试成功且用户确认收到后写 `verified_at`；邮箱变化即失效。
- 现有 `first_send_approved_at` 在测试邮件确认后由 UI 调用 `approve_first_send` 写入；不绕过现有首封发送门禁。
- 日志/状态 JSON/测试 fixture 不出现原始邮箱；沿用 `push_status.mask_email` 与脱敏逻辑。

## 主题 → 检索条件

- 输入接受中文/英文，一段话或几行主题。
- 主题变化时调用一次小型 LLM（沿用 DeepSeek 配置）生成 1–3 组：
  - `name`、`any_terms`、`require_any_terms`；
  - 只做翻译/受控扩展，不新增无关研究方向；
  - 结果通过 `profile_config`/`push_config` 校验后写入 `interests.json`；
  - `research_goal` 保存原始输入，`topics` 保存分行主题；`preferred_methods`/`less_interested` 等高级字段保持不变。
- 采用事务式保存：先成功生成并校验查询组，才写 `research_goal`/`topics`/`search.groups`。
- LLM 失败或生成结果无效时：**不保存新主题**，保留旧 profile/groups；显示“主题未保存，检索词生成失败”，预览不运行，可重试或去高级手动编辑。不允许出现“新主题已保存、候选池还是旧的”。
- 邮箱/篇数/时间可以独立保存；主题与查询组作为一个原子单元。
- 折叠显示“这次会搜什么”；高级页仍可精确编辑查询组。
- 候选池仍然只来自用户主题生成的查询组 + arXiv 分类；模型只在候选池内排序/解释。

## 预览 dry-run

- 现有 `run_push` 在 `enabled=false`/`paused=true` 时直接返回 disabled/paused。新增 preview-only 的 bypass（例如 `force_preview=True` 或内存中 `enabled=True,paused=False` 的副本），只对 `mode="preview"` 生效，不写配置。
- 预览改为**下一封的预测**：只读生产 frontier 状态（watermark、已发送/已忽略/已投递），使用与下一次发送相同的窗口和排除条件，但不写生产状态。这样不会把已经发过的论文再展示成新推荐。
- 预览需要当前主题/邮箱/篇数/日期窗口；设置变化或跨天后预览标记为过期，手动发送前重新生成。
- 手动发送如果使用刚生成且未过期的预览，就发送这份预览的选中集合；否则重新跑当天 digest。
- 预览不创建 SMTP 连接；即使 `.env` 没有 SMTP 授权码也能生成预览。
- 预览会抓取 arXiv 并调用排序/摘要模型；界面明确一句“预览会联网并调用模型，不会发邮件”。
- 失败只显示简短原因和重试；不把 provider exception 原文/凭据写进日志。

## 导航与高级入口

- Streamlit 1.66 支持 `st.navigation`/`st.Page`/`st.page_link`。
- 顶层导航只显示：`每日推送`（简单页）和 `高级 / 研究模式`（一个入口）。
- 高级入口内再按组收纳：
  - 总览/评测表（现有 `pages/0_总览.py` 的研究内容；baseline、检索/答案/Agent A/B 结果保留，供简历/面试演示）；
  - **推送历史（新增：日期 + 已推送论文 + 中文标题/摘要 + arXiv 链接）**；
  - 证据问答（现有 `app.py` 的问答/证据/trace 逻辑）；
  - 答案/检索评测页；
  - 推送设置（SMTP/计划任务/历史/时间）；
  - 精确兴趣组；
  - 论文库/论文发现。
- 总览页的重复自动推送开关不再保留独立语义：删除或改为链接到简单主屏；如果保留，必须调用与主屏相同的共享校验函数，不能成为绕过测试邮件的第二条开启路径。
- 旧 `pages/*.py` 移到 `advanced/` 或包一层 page wrapper；不删除底层模块。
- 主屏不显示 sidebar 的多个研究页；`initial_sidebar_state="collapsed"` 或自定义导航。
- 高级页保持现有安全边界：预览不发信、真实发送需确认、状态掩码邮箱。
- 高级模式同时是“研究/简历模式”：baseline、检索/答案/Agent 评测、CLAIMS 和证据问答都在这里；简单主屏只是默认入口，不删除、不改写这些资产。

## 推送历史（高级）

- 只列**真实发送过**的 digest（`status=sent`，mode 为 send/scheduled；不把 preview 当已推送）。
- 按 `finished_at`/`started_at` 倒序；每行显示本地推送日期时间、主题、篇数、手动/自动标记。
- 展开后显示该次 digest 的论文卡片：
  - 英文标题；
  - 当时已生成的中文标题、推荐理由、中文摘要/研究问题/方法/结果（digest 里已有，不再额外调用模型翻译）；
  - 兴趣匹配分（可选）、arXiv 链接；
  - “看邮件版”打开当时的 `digest_preview.html`。
- 数据来源：`data/push/runs/<run_id>/run.json` + `digest_preview.html/.txt`；未来运行额外写结构化 `digest.json`（论文标题/中文标题/中文摘要/URL/日期），历史页优先读它，旧运行回退解析/展示 HTML。
- 不显示收件人、SMTP、token、评测指标；历史页只读，不重发。
- 中文翻译用当时 digest 的版本，保证“你看到的”和“当时收到的”一致；如果旧运行缺中文，显示原文并标注“缺中文摘要”，不偷偷重译。

## GitHub 隐私 / 公开仓库

用户已选择“都脱敏”：默认大屏简单，高级保留研究资产，但公开仓库不包含个人资料。

- **兴趣 profile**：active 版本放 `data/push/profile.json`（`data/` 已 gitignore）；公开 `interests.example.json`（通用示例）。`interests.json` 从 git index 移除并加入 gitignore，只留本地兼容/回退；不再把真实研究目标/主题提交到 GitHub。
- **旧 handoff 文档**：`docs/handoff/ACCEPTANCE_REPORT.md`、`CURRENT_STATUS.md`、`NEXT_STAGE_PLAN.md`、`PROJECT_ROADMAP.md` 从公开 index 移除（本地保留），`docs/handoff/` 整体忽略；不删除本地文件。
- **旧个人/交接产物**：根目录 `paperlens_answer*`、`review_notes/`、`PROJECT_NOTES.md`、`PROJECT_HANDOFF_*`、`COLLECTION_*`、`DATASET_*`、`EVAL_START_HERE.md`、`LARGE_DATASET_PLAN.md`、`PACKAGE_CHECKS.json`、`PREPARATION_REVIEW.json`、`PREPARE_DATASET_START.md`、`Q01_ACCEPTANCE.json`、`SCHEMA_FIX_START.md`、`SCREEN_DATASET_START.md`、`dataset_targets.json`、旧中文说明/验收文件、`docs/stage1a/`、`docs/stage1b/`、`docs/stage1c/`、`eval/results/`、`eval/reviews/` 全部 gitignore；不执行 `git add -A`。
- **公开但安全**：`push_config.json` 只含 enabled/time/provider 等非敏感设置，不含收件人；收件人在 `data/push/ui_state.json`；推送历史不显示邮箱；SMTP 授权码只在 `.env`。
- **提交前扫描**：publish 前检查 tracked 文件和 git 历史中是否有真实邮箱/API key/授权码；测试 fixture 只用假地址；最后用 `git grep`/脚本复扫。
- **研究资产保留**：baseline、检索/答案/Agent 评测、CLAIMS、证据问答、push 历史都在高级“研究模式”里；主屏简化不删除、不改写这些资产。
- **profile resolver**：`data/push/profile.json` > `interests.json`（本地旧文件）> `interests.example.json`；UI 只写 active 路径，公开示例只读。

## 计划任务

- 每日发送依赖现有 Windows Task Scheduler（任务名 `PaperLens Frontier Push`，默认 08:30）。
- 首次开启时在测试邮件确认后安装任务；可直接调用 `scripts/register_push_task.ps1`（per-user，无需保存密钥）。
- 安装失败则不开启，主屏显示简短错误和“高级/重试”入口。
- 关闭开关只改配置，不删除任务；任务恢复后按配置 no-op/发送。
- 计划任务在 Streamlit 之外运行，必须使用与 UI 相同的收件人解析：
  - `ui_state.json` 存在 → 只使用其中收件人；未验证/缺失/损坏时 fail closed，不发送，不回退 `.env`；
  - `ui_state.json` 不存在 → 兼容旧 `.env` `MAIL_TO`；
  - 错误状态只记录掩码原因，不记录原始邮箱。
- 时间/时区/任务状态在高级页显示与修改；主屏只显示“明天 08:30”这样的结果。

## 后端改动清单

- `simple_push_ui.py` 或 `ui_state.py`：读写 `data/push/ui_state.json`（原子写），收件人掩码/验证状态/preview 引用/最后手动发送、时间、本次篇数；不依赖 Streamlit，便于离线测试。
- `topic_queries.py`：主题 → 1–3 组查询；注入 `chat_fn` 便于测试；校验/回退。
- 篇数写入 `push_config.max_per_digest` 与 `profile.top_k`，两边一致；预览/手动/自动都受这个上限约束。
- `push_run.py`：增加 preview-only bypass，以及显式 manual-send bypass；两者都不改变 scheduled 的 enabled/paused 门禁，manual 仍走首封批准/SMTP。
- `push_ctl.py`：测试邮件/首封批准/任务安装/按 `-Time` 重注册的调用封装；不新增绕过门禁的命令。
- `profile_config.py`/UI：active profile resolver（`data/push/profile.json` > `interests.json` > `interests.example.json`）；UI 写 active；公开示例只读。
- `.gitignore` + git index：加入 active profile、旧 handoff/个人产物；`git rm --cached` 旧 handoff 4 文件和 `interests.json`；本地文件保留。
- `scripts/check_public_repo.py`：发布前扫描 tracked 文件/历史中的邮箱、API key、授权码模式；CI 或手动运行。
- `push_config.py`/`push_config.json`：`time_local` 继续作为唯一时间源；UI 只写这个字段。
- `push_history.py`：只读 `data/push/runs/*/run.json` + `digest.json`（新）/`digest_preview.html/.txt`（旧），按发送日期排序，只返回 `status=sent`；不读取收件人。
- `push_run.py`：每次发送/预览额外写结构化 `digest.json`（英文标题、中文标题、中文摘要、URL、日期、run 元数据），供历史页和无损展示；保留 html/txt。
- `scripts/register_push_task.ps1`：已有 `-Time` 参数，UI 复用它；补一个任务状态查询/重注册的轻封装。
- `app.py` / `pages/`：新简单页 + 高级导航；旧页面迁移。
- `README.md` / `docs/PUSH.md`：更新运行方式、隐私、首次测试、任务安装。

## 测试与验收

离线测试：
- UI state 原子写；邮箱变化清除验证；原始邮箱不出现在日志/状态 JSON。
- 首次开启：无验证 → 发测试邮件；测试失败/未确认 → 不开启；确认后 → 开启；再次开关 → 一键。
- 预览在 `enabled=false,paused=true` 时仍可运行（bypass），且不创建 SMTP 连接；生产 `scheduled/send` 门禁不变。
- 手动发送在 `enabled=false,paused=true` 时仍可运行（显式 manual bypass），但不改变 enabled/paused，且仍要求邮箱验证/首封批准/确认。
- 手动发送复用新鲜 preview；无 preview 时重新生成；发送后生产状态推进，下一次自动/手动不重复同一批论文。
- 时间选择写入 `time_local`（HH:mm）；自动开启时用 `-Time` 重注册任务；重注册失败可见且不假称已生效。
- 篇数写入 `max_per_digest` 和 `top_k`；预览/手动使用当前值；可用论文不足时按实际数量发送并提示。
- 主题生成：有效输出才写 groups；无效/失败不保存新主题、保留旧 groups；绝不出现“主题已变、候选池未变”。
- 高级页导入无网络/模型/SMTP 副作用；旧测试全部通过。
- 开关只改变 enabled/paused，不触发预览/主题/邮箱写入之外的副作用。
- 推送历史只列 `sent`，不列 preview；按日期倒序；中文标题/摘要来自当时 digest；无邮箱/密钥；旧运行缺 `digest.json` 时有 HTML/文本回退；历史页不调用模型。
- 简单主屏不出现推送历史、baseline、评测或证据；高级入口能打开它们。
- active profile 写入 gitignored 路径；`interests.example.json` 可公开；旧 handoff 文档不再 tracked；公共文件扫描无真实邮箱/密钥。
- 推送历史读取 `data/push` 本地数据，不把收件人/邮件内容提交到 git。

人工验收：
- 打开 app 只看到简单页；预览卡片正常；关闭推送时预览仍可用。
- 时间选择可改并生效；开启自动推送后任务按新时间触发（或明确报错）。
- 篇数可选 1–10；今天 1 篇、明天 10 篇都能通过改数字实现；预览与发送的篇数一致。
- 首次开启收到测试邮件；确认后开关生效；之后一键开关；换邮箱重新测试。
- 自动推送关闭时，点“现在推送今天的”能真实发送一次，且不会把自动开关打开。
- 手动发送前有确认；未验证邮箱/未批准首封时被拦住并引导测试。
- 高级入口可打开旧页面，问答/评测/baseline 仍可运行；简历用的数字和 CLAIMS 不被 UI 重构删除。
- 高级“推送历史”能看到真实发送过的论文，带推送日期、中文标题/摘要、arXiv 链接；preview 不冒充已推送。
- GitHub 公开仓库不含真实邮箱/API key/授权码/个人 profile/旧 handoff；`git grep` 复扫通过；高级研究/评测资产仍可公开演示。
- `git status` 不出现邮箱/`.env`；状态/日志中邮箱为掩码。

## 我的默认决定（可在确认时改）

- 收件人存 `data/push/ui_state.json`，不存 `.env`/`push_config.json`。
- 主题→检索组用一次小型 LLM 调用；失败不保存新主题，保留旧组，预览不运行；高级可编辑。
- 预览使用 dry-run bypass；推送关闭也能预览。
- 首次开启自动尝试安装 Windows 计划任务，失败则不开推送。
- 时间选择放主屏，默认 08:30，写 `time_local`；自动开启时重注册任务。
- “现在推送今天的”是独立手动发送，自动关闭也可用；不改自动开关；有新鲜预览就发预览，否则重新生成；发送前确认且仍要求测试邮件/首封批准。
- 篇数放主屏明显位置，1–10 的 stepper/滑块 + 1/3/5/10 快捷，默认 3；今天 1 篇/明天 10 篇用同一个数字随手改；v1 不做按日期预排日历。
- 主屏只保留主题/邮箱/篇数/预览/手动推送/自动开关/时间/高级链接；其他全部收纳。
- 推送历史放高级；只读真实 sent runs；中文内容用当时 digest 的版本，不额外翻译；新 run 写 `digest.json`，旧 run 回退 HTML/文本。
- baseline/评测/证据/CLAIMS 留在高级研究模式，作为简历/面试演示资产，不因主屏简化而删除或改写。
- 隐私按“都脱敏”：active profile 进 `data/`、公开 `interests.example.json`；旧 handoff 从公开 index 移除并忽略；旧个人产物全部 gitignore；发布前扫描。

## Builder check 后已定的实现口径

这些是设计层面的默认决定；如果用户不反对，按这里实现。

1. **验证门禁范围**：所有 UI 开启路径（主屏、总览、推送设置）共用 `set_auto_push`；开启前要求当前 UI 收件人已验证。高级页/CLI 只在完全没有 `ui_state.json` 时兼容 `.env` 旧收件人；一旦有 UI state，未验证就拒绝。计划任务/手动发送同样 fail closed。总览页的重复开关删除或改为链接。
2. **开关状态模型**：`on` 要求 `auto_paused=false` 且 `quiet_mode=false`，写 `enabled=true, paused=false`；不清除 `auto_paused`。`off` 写 `enabled=false`，不清除安全状态。状态显示要把 `paused`/`auto_paused`/`quiet` 解释成具体原因，不能显示“已开启”但实际不发。
3. **邮箱验证**：只保留当前收件人 + `verified_at`；换邮箱（含换回）清除验证并自动关闭自动推送；未保存的邮箱修改使自动开关禁用。不做已验证地址白名单。
4. **主题事务**：先成功生成/校验查询组，才写主题和 groups；失败不保存、不预览，保留旧设置。
5. **预览口径**：预测下一封（只读生产 frontier 状态，不重复已发送），不是独立样本；篇数用当前设置（1–10，不再写死 3–5）；预览过期/设置变化后手动发送前重新生成。
6. **计划任务收件人**：`ui_state.json` 存在时只用它并验证，缺失/损坏/未验证时 fail closed，不回退 `.env`；不存在时才用旧 `.env`。
7. **总览页**：收入高级；保留评测/推送状态，但删除或重定向重复的自动推送开关，避免第二条绕过测试邮件的开启路径。
8. **篇数**：v1 用单一“每次推送几篇”数字，今天 1 篇/明天 10 篇通过改数字实现；按日期预排日历不做。
