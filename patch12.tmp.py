"""Fixtures need the new right-rail inputs too."""
import pathlib
p = pathlib.Path("tools/preview_report.py"); s = p.read_text(encoding="utf-8")
old = """    render_html(WORKS, web, diagnostics=DIAGNOSTICS, problem_names=NAMES,
                library_size="4399 篇", window_days=30,
                coverage_warnings=["OpenAlex 查询轮换未覆盖全部追踪期刊，本轮覆盖不足"])"""
new = """    render_html(WORKS, web, diagnostics=DIAGNOSTICS, problem_names=NAMES,
                library_size="4399 篇", window_days=30,
                library_directions=LIBRARY_DIRECTIONS,
                coverage_warnings=["OpenAlex 查询轮换未覆盖全部追踪期刊，本轮覆盖不足"])"""
assert old in s
s = s.replace(old, new, 1)

old2 = 'DIAGNOSTICS = {'
new2 = '''# Real counts from the 4399-paper local profile, so the sidebar bars are honest.
LIBRARY_DIRECTIONS = [
    ("延性断裂与损伤演化", 1209), ("应力状态依赖塑性", 1161), ("温度—应变率耦合", 1092),
    ("冲击与结构验证", 950), ("增材组织—缺陷—失效", 382),
    ("数值实现与损伤正则化", 157), ("参数反演与硬化外推", 115),
]

DIAGNOSTICS = {'''
assert old2 in s
s = s.replace(old2, new2, 1)

# Give one paper a watched author so that card renders in the preview.
old3 = '''         "Ballistic perforation of thin titanium plates: failure mode transition with target thickness",
         "consider", "impact", ["R. Kumar", "S. Gupta"], "Thin-Walled Structures", 27,'''
assert old3 in s
p.write_text(s, encoding="utf-8")
print("preview fixtures updated")
