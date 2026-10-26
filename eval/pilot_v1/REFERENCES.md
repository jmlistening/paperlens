# PaperLens 试评测原文核对表

这是预先固定的评分依据，不是模型输出。页码均为 PDF 物理页。

## Q01 · 实验设置 · 2610.19819v1

《The Cost of Context》使用了什么 GPU、哪两个模型及什么数值精度？为了隔离 prefill 阶段，生成长度设置为多少 token，并做了什么预热处理？

- **Q01.1**：GPU 是 NVIDIA A100 80GB。

PDF 第 2 页：

> The benchmarks are conducted on NVIDIA A100 (80GB) instances across two model fam- ilies (Llama-3.1 8B, Qwen-2 7B) using bfloat16 precision to account for architectural variability in attention implementation.

- **Q01.2**：模型为 Llama-3.1 8B 和 Qwen-2 7B，精度为 bfloat16。

PDF 第 2 页：

> The benchmarks are conducted on NVIDIA A100 (80GB) instances across two model fam- ilies (Llama-3.1 8B, Qwen-2 7B) using bfloat16 precision to account for architectural variability in attention implementation.

- **Q01.3**：隔离 prefill 时生成长度设为恰好 1 token；各配置之前进行 warm-up，使 GPU 时钟速度及功率状态稳定。

PDF 第 3 页：

> In both cases, we isolate the prefill phase by setting the generation length to exactly 1 token. Each configuration is preceded by a warm-up pass to stabilise GPU clock speeds and power states.

## Q02 · 输入边界与训练机制 · 2610.19341v1

MMKG-R1 的构图为什么称为 question-blind？构造器能看到和不能看到哪些信息？训练时更新哪些组件，哪些组件保持固定？

- **Q02.1**：构造器对每个知识源单独读取可见内容 v(x)，不看问题 q、标准答案 Aq 或其他知识源；抽取结果可跨问题复用。

PDF 第 3 页：

> Construction is question- blind: πθ processes each x ∈Xq independently from v(x) alone, without access to q, Aq, or the other sources, and yields Gθ(Xq) via Eqs. (1)–(2), so its extractions can be reused across questions.

- **Q02.2**：来源元数据和原始内容引用等额外信息由环境维护，不显示给构造器。

PDF 第 3 页：

> Additional source information µx, such as prove- nance metadata and references to the original con- tent, is maintained by the environment and is never shown to πθ.

- **Q02.3**：仅训练/更新构造器；解析器、组装器固定，实体链接、检索与预算内证据打包等环境组件也固定。

PDF 第 3 页：

> Two fixed, non-trainable modules then process the extractions: the parser P parses each yx, and the assembler B uses the parsed results and µx to assign graph identifiers, attach prove- nance, link entities across sources, and assemble the graph as:

PDF 第 4 页：

> We use a shared, fixed environment to perform entity linking, graph assembly, retrieval, and budgeted evidence packing for all candidates. Throughout training, only the constructor is updated, while all other components remain fixed.

## Q03 · 检索流程机制 · 2610.22163v1

Multi-sourced AI Tutor 为什么在 RAG 之前单独做 Question Extraction（QE），普通向量或关键词检索有什么困难？QE 如何得到原题内容，后续相似度搜索使用哪些输入、返回多少条结果？

- **Q03.1**：学生可能只提供题号；短且数字化的引用不利于向量检索，题号缩写变化使关键词检索困难。

PDF 第 3 页：

> Embedding-based methods struggle due to the short and numeric nature of these references, while keyword-based systems encounter difficulties due to variability of question number abbreviations (e.g., "Project 2 Question 10" and "Fall 2022 Midterm 2 Question 10").

- **Q03.2**：Llama 3 8B 生成标准化题号，再由规则算法从题库获取题目内容。

PDF 第 3 页：

> To address these issues, we used the Llama 3 8B model to generate standardized question numbers from student posts.

PDF 第 3 页：

> The generated standardized question number enables us to accurately fetch the question content from a question bank using a rule-based algorithm.

- **Q03.3**：相似度检索输入学生帖子及已抽取的题目，返回最相关的 top 10。

PDF 第 3 页：

> In similarity search phase, we input both the student post and any extracted questions. We exclude past exams and Canvas quizzes from our search to avoid skewing results towards similar exam or quiz questions, which could prevent the retrieval of lecture materials. We retrieve the top 10 most relevant results.

## Q04 · 表格数值与适用范围 · 2610.19819v1

在《The Cost of Context》中，约 112k 输入 token 时，Llama-3.1 8B 的 HF Baseline 与 vLLM 分别报告了多少 prefill 延迟和能耗？论文报告的能效提升约为多少倍？请保留实验条件，不要推广成所有模型的固定提升。

- **Q04.1**：限定约 112k token、Llama-3.1 8B，测量对象为 prefill；不能写成 PaperLens 的实测表现。

PDF 第 3 页：

> As detailed in Table 1, at approximately 112,000 tokens, the vLLM engine processed the Llama-3.1 input in 3.54 seconds (1,005 Joules), while the baseline required 25.54 sec- onds (7,600 Joules). This represents a 7.6× improvement in energy efficiency for the optimised engine

- **Q04.2**：HF Baseline 延迟 25.54 s，能耗 7600.2 J（接受正文四舍五入 7600 J）；vLLM 延迟 3.54 s，能耗 1005.1 J（接受 1005 J）。

PDF 第 3 页：

> Llama-3.1 8B HF Baseline 25.54 7,600.2 0.0678 vLLM (Paged) 3.54 1,005.1 0.0090

- **Q04.3**：该模型该配置报告约 7.6 倍能效提升；不能把 Qwen-2 的最高 8.9 倍移给 Llama。

PDF 第 3 页：

> This represents a 7.6× improvement in energy efficiency for the optimised engine (reaching up to 8.9× for Qwen-2).

核对说明：表格读取困难可用正文等价数值；不能因引用命中而忽略模型名、单位或行列对应错误。

## Q05 · 奖励层级与职责 · 2610.19341v1

MMKG-R1 的 Deducibility、Source-Grounded Quality 和 Graph-Readiness 三种奖励各自检查什么？它们分别在图级还是知识源抽取级计算，为什么采用不同层级？

- **Q05.1**：Deducibility 检查检索证据是否足以支持正确答案，在候选图级计算。

PDF 第 3 页：

> (i) Deducibility Reward, which evaluates whether the retrieved evidence from a constructed MMKG is sufficient to support

PDF 第 4 页：

> a correct answer;

PDF 第 4 页：

> Evidence sufficiency depends on the question and on how extractions from different sources combine, so the Deducibility Reward is computed at the candidate-graph level.

- **Q05.2**：Source-Grounded Quality 检查抽取知识是否忠实保留原始知识源的核心内容，在知识源抽取级计算。

PDF 第 4 页：

> (ii) Source-Grounded Quality Reward, which assesses whether the knowledge extracted by the constructor faithfully preserves the central content of its original knowledge source;

PDF 第 4 页：

> Fidelity and usability are properties of each extraction re- gardless of the question, so the Source-Grounded Quality and Graph-Readiness Rewards are com- puted at the source-extraction level.

- **Q05.3**：Graph-Readiness 检查输出是否结构有效并可用于构图，在知识源抽取级计算。

PDF 第 4 页：

> and (iii) Graph-Readiness Reward, which encour- ages structurally valid and usable outputs for graph construction.

PDF 第 4 页：

> Fidelity and usability are properties of each extraction re- gardless of the question, so the Source-Grounded Quality and Graph-Readiness Rewards are com- puted at the source-extraction level.

- **Q05.4**：层级不同是因为证据充分性依赖问题及跨来源组合；忠实性和可用性是与问题无关的单次抽取属性。

PDF 第 4 页：

> Evidence sufficiency depends on the question and on how extractions from different sources combine, so the Deducibility Reward is computed at the candidate-graph level. Fidelity and usability are properties of each extraction re- gardless of the question, so the Source-Grounded Quality and Graph-Readiness Rewards are com- puted at the source-extraction level.

## Q06 · 不可回答边界 · 2610.19819v1

仅依据《The Cost of Context》，RTX 4060 运行 PaperLens 每月 1000 次论文问答的电费精确是多少人民币？请给出论文报告的金额；没有报告就明确说明，不要自行估算。

- **Q06.1**：明确原文没有报告这一 RTX 4060 / PaperLens / 每月 1000 次问答的人民币电费，不提供伪造金额。

- **Q06.2**：若补充实验范围，应正确指出研究测量 A100 上特定模型和引擎的能耗；不能把这些实验直接换算成本题账单。

PDF 第 2 页：

> The benchmarks are conducted on NVIDIA A100 (80GB) instances across two model fam- ilies (Llama-3.1 8B, Qwen-2 7B) using bfloat16 precision to account for architectural variability in attention implementation.

核对说明：已核对该 PDF 全部 5 页；缺失信息不能靠单个引文证明。第一项必需；第二项为条件性核对：只明确拒绝且没有补充范围也算通过。不强求特定 status 枚举，按内容判断。不把答案拒绝一切问题当作正确策略。
