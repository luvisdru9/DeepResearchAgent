# ResearchBench 评测体系

ResearchBench 可以覆盖事实问答、对比研究、趋势分析、争议问题和多跳推理。每个样本应包含问题、领域、标准答案要点、必要证据、禁止幻觉和评分 rubric。

规则指标可以包括 fact accuracy、citation coverage、citation precision、hallucination rate、answer completeness、latency 和 cost。LLM-as-Judge 可以从事实性、完整性、引用质量、逻辑结构和表达质量五个维度评分。

实验设计应包含 baseline linear、dag multi agent、dag memory、dag memory redteam 和完整系统，以便量化每个模块带来的收益与成本。
