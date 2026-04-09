from txt_parser import parse_course_txt
from chunking import chunk_course, count_tokens
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 测试解析
result = parse_course_txt("course_docs/COMP5422_20251.txt")
print(json.dumps(result, indent=2, ensure_ascii=False))

# 测试分块
chunks, parents = chunk_course(result)
print(f"\n共 {len(chunks)} 个 chunks，{len(parents)} 个 parents")
for c in chunks:
    m = c.metadata
    print(f"  [{m['section_type']:20s}] is_child={m['is_child']:<5} "
          f"parent_id={str(m['parent_id']):<25} "
          f"tokens={count_tokens(c.page_content):>4} "
          f"| {c.page_content[:60]}...")
