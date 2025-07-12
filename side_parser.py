import re
from typing import List, Tuple, Optional

RangeSide = Tuple[int, int, str]

def parse_side_mapping(raw: str) -> List[RangeSide]:
    """
    将类似 "1-15:左,16-20:单,21-25:右" 解析成 [(1,15,"左"), (16,20,"单"), (21,25,"右")].
    如果 raw 中不含数字-数字结构，返回空列表。
    """
    lst: List[RangeSide] = []
    # 先统一全角标点到半角
    s = raw.replace('：', ':').replace('，', ',').strip()
    # 拆分段
    for part in s.split(','):
        part = part.strip()
        m = re.match(r'(\d+)\s*-\s*(\d+)\s*:\s*(左|右|单)', part)
        if m:
            start, end, side = int(m.group(1)), int(m.group(2)), m.group(3)
            # 确保 start<=end
            if start <= end:
                lst.append((start, end, side))
    return lst

def get_expected_side_for_tower(raw: Optional[str], tower_id: int) -> Optional[str]:
    """
    返回该塔号的期望侧别：
      - raw 为 None 或 空 → None
      - raw 不含任何范围 → 直接返回 raw（“左”/“右”）
      - raw 含范围，解析后匹配到该塔 → 返回对应 side
      - 解析后若对应 side 为“单”或未匹配到 → 返回 None
    """
    if not raw:
        return None
    raw = raw.strip()
    # 如果根本没有“数字-数字”模式，当作整体侧别
    if not re.search(r'\d+\s*-\s*\d+', raw):
        return raw if raw in ('左', '右') else None

    # 否则解析所有段
    ranges = parse_side_mapping(raw)
    for start, end, side in ranges:
        if start <= tower_id <= end:
            return side if side in ('左', '右') else None
    return None
