import re
from typing import List, Tuple, Optional

RangeSide = Tuple[int, int, str]

def parse_side_mapping(raw: str) -> List[RangeSide]:
    """
    将类似：
      - "1-15:左,16-20:单,21-25:右"
      - "1:左, 2:单, 3-5:右"
    解析成 [(start, end, side), ...]，其中单点写法 "1:左" 会被视为 (1,1,"左").
    允许使用全角：'：'、'，'。
    """
    lst: List[RangeSide] = []
    if not raw:
        return lst

    # 统一全角标点
    s = raw.replace('：', ':').replace('，', ',').strip()
    if not s:
        return lst

    # 拆分片段，依次匹配“范围”或“单点”
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue

        # 先匹配范围 A-B:侧别
        m_range = re.match(r'^(\d+)\s*-\s*(\d+)\s*:\s*(左|右|单)$', part)
        if m_range:
            start, end, side = int(m_range.group(1)), int(m_range.group(2)), m_range.group(3)
            if start <= end:
                lst.append((start, end, side))
            continue

        # 再匹配单点 A:侧别  →  等价于 A-A:侧别
        m_single = re.match(r'^(\d+)\s*:\s*(左|右|单)$', part)
        if m_single:
            idx, side = int(m_single.group(1)), m_single.group(2)
            lst.append((idx, idx, side))
            continue

    return lst

def get_expected_side_for_tower(raw: Optional[str], tower_id: int) -> Optional[str]:
    """
    返回该塔号的期望侧别：
      - raw 为空 → None
      - raw 为整体侧别（仅“左/右”）→ 返回该整体侧别
      - raw 含片段（范围 A-B:侧别 或 单点 A:侧别）→ 命中则返回“左/右”，命中“单”或未命中 → None
    """
    if not raw:
        return None
    txt = raw.strip()
    if not txt:
        return None

    # 是否包含“片段映射”（范围或单点）：有冒号且左侧至少有数字
    has_mapping = re.search(r'\d+\s*(?:-\s*\d+)?\s*:\s*(左|右|单)', txt.replace('：', ':')) is not None
    if not has_mapping:
        # 视为整线统一侧别
        return txt if txt in ('左', '右') else None

    # 解析所有片段并匹配 tower_id
    ranges = parse_side_mapping(txt)
    for start, end, side in ranges:
        if start <= tower_id <= end:
            return side if side in ('左', '右') else None
    return None
