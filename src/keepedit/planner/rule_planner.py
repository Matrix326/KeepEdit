from __future__ import annotations

import re

from keepedit.schemas import EditPlan, EditRequest


class RulePlanner:
    """A deterministic planner used as a robust baseline and fallback."""

    GLOBAL_KEYWORDS = {
        "style",
        "background",
        "scene",
        "weather",
        "lighting",
        "整体",
        "风格",
        "背景",
        "场景",
        "天气",
        "光照",
    }
    TEXT_KEYWORDS = {"text", "word", "letter", "sign", "logo", "文字", "文本", "标志", "招牌"}
    ADD_REMOVE_KEYWORDS = {"add", "insert", "remove", "delete", "erase", "添加", "增加", "删除", "移除"}
    COLOR_KEYWORDS = {"color", "colour", "red", "blue", "green", "yellow", "颜色", "红", "蓝", "绿", "黄"}
    REASONING_KEYWORDS = {"left", "right", "behind", "front", "larger", "smaller", "左", "右", "后", "前", "更大", "更小"}

    def plan(self, request: EditRequest) -> EditPlan:
        text = request.instruction.strip()
        lowered = text.lower()
        tokens = set(re.findall(r"[\w]+", lowered))

        edit_type = request.edit_type or self._classify(lowered, tokens)
        local_or_global = "global" if tokens & self.GLOBAL_KEYWORDS or any(k in lowered for k in self.GLOBAL_KEYWORDS) else "local"
        if edit_type in {"style", "background", "global"}:
            local_or_global = "global"

        target_phrases = self._extract_targets(text)
        preservation_level = "very_high" if local_or_global == "local" else "medium"
        return EditPlan(
            edit_type=edit_type,
            target_phrases=target_phrases,
            local_or_global=local_or_global,
            preservation_level=preservation_level,
            notes="rule_planner_v1",
        )

    def _classify(self, lowered: str, tokens: set[str]) -> str:
        if self._contains(lowered, tokens, self.TEXT_KEYWORDS):
            return "text"
        if self._contains(lowered, tokens, self.ADD_REMOVE_KEYWORDS):
            return "add_remove"
        if self._contains(lowered, tokens, self.COLOR_KEYWORDS):
            return "color"
        if self._contains(lowered, tokens, self.REASONING_KEYWORDS):
            return "reasoning"
        if self._contains(lowered, tokens, self.GLOBAL_KEYWORDS):
            return "style"
        return "local"

    @staticmethod
    def _contains(lowered: str, tokens: set[str], keywords: set[str]) -> bool:
        return bool(tokens & keywords) or any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _extract_targets(text: str) -> list[str]:
        patterns = [
            r"(?:change|turn|make|replace|remove|add)\s+(?:the\s+)?(.+?)(?:\s+to|\s+into|\s+with|$)",
            r"(?:把|将)(.+?)(?:改成|变成|换成|替换为|删除|移除)",
        ]
        targets: list[str] = []
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                phrase = match.group(1).strip(" ,.;，。")
                if phrase and len(phrase) <= 80:
                    targets.append(phrase)
        return targets
