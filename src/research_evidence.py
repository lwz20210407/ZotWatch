"""Title/abstract-only method cards. No full-text retrieval or citation-role invention."""
import re

from .research_features import feedback_links
from .topic_matching import matches_any


METHOD_NAMES = ["Johnson-Cook", "Hosford-Coulomb", "Mohr-Coulomb", "Gurson", "GISSMO",
                "LS-OPT", "VUMAT", "UMAT", "virtual fields method", "crystal plasticity",
                "Bai-Wierzbicki", "MMC", "GTN", "Yld2000", "Hill48", "FEMU", "VFM", "MAT_224"]


def abstract_signals(abstract):
    """A method mentioned in a sentence is NOT an attributed role for a cited paper."""
    signals = []
    patterns = [("局限/不适用线索", r"\b(fail\w*|inaccurate|limitation\w*|cannot|not applicable)\b|不适用|不足"),
                ("比较评估线索", r"\b(compar\w*|benchmark\w*|versus)\b|对比|比较"),
                ("改进扩展线索", r"\b(extend\w*|modif\w*|improv\w*)\b|改进|扩展"),
                ("采用方法线索", r"\b(adopt\w*|implement\w*|employ\w*|used?|using)\b|采用|使用")]
    for sentence in re.split(r"(?<=[.!?。！？])\s*", abstract or ""):
        names = [name for name in METHOD_NAMES if matches_any(sentence, [name])]
        if not names:
            continue
        if re.search(r"\b(not|never)\s+(?:\w+\s+){0,2}(adopt|use|employ|implement)|未采用|不使用", sentence, re.I):
            roles = ["未采用/否定语境线索"]
        else:
            roles = [label for label, pattern in patterns if re.search(pattern, sentence, re.I)]
        if roles:
            signals.append({"methods": names, "role": "；".join(roles), "sentence": sentence,
                            "caveat": "摘要句子层面的规则线索，不等同于对某篇被引论文的评价"})
    return signals[:2]


def attach_evidence(work, config):
    level = "摘要" if work.abstract else "仅题名"
    content = work.abstract or work.title
    cards = []
    budget = 24
    for facet in config.facets:
        sentences = re.split(r"(?<=[.!?。！？])\s*", content)
        sentence = next((s for s in sentences if matches_any(s, facet.terms)), None)
        if not sentence:
            continue
        snippet = " ".join(sentence.split()[:min(6, budget)])[:100]
        budget -= len(snippet.split())
        cards.append({"topic": facet.name, "use": facet.use + "（规则推断，非已验证迁移）",
                      "verify": facet.verify, "evidence": snippet + (" …" if snippet else ""),
                      "level": level, "source_url": work.url,
                      "status": "标题/摘要不能确认全部实验条件与方法适用性，未陈述内容保持未知"})
        if len(cards) == 2:
            break
    signals = abstract_signals(work.abstract)
    for signal in signals:
        sentence = signal.pop("sentence")
        snippet = " ".join(sentence.split()[:min(6, budget)])[:100]
        budget -= len(snippet.split())
        signal["evidence"] = snippet + (" …" if snippet else "")
    return work.model_copy(update={"extra": {**work.extra, "transfer_cards": cards,
        "abstract_method_signals": signals, "evidence_level": level,
        "feedback_links": feedback_links(work, config),
        "citation_context_status": "不读取正文；仅凭引文元数据不能判断对被引论文的采用、改进或反驳"}})
