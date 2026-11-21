"""核心逻辑单测：切块、引用校验、评测指标。只用标准库，可直接跑。

  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chunk as chunk_mod  # noqa: E402
import eval as eval_mod  # noqa: E402
import generate as gen  # noqa: E402

HANDBOOK = """第一章 学籍管理
第一节 入学与注册
第一条 新生应当在规定期限内办理入学手续。
第二条 未按期报到且无正当理由的，视为放弃入学资格。
第二章 休学与复学
第十二条 学生休学一般以一年为期，累计不得超过两年。休学期间不享受在校学习学生待遇。
"""

DOC = {"doc_id": "handbook_2026", "title": "学生手册（2026版）",
       "cohort": "", "status": "active", "text": HANDBOOK}


class ChunkTest(unittest.TestCase):
    def test_article_split_keeps_article_no(self):
        chunks = chunk_mod.chunk_doc(DOC)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["article_no"], "第一条")
        self.assertEqual(chunks[2]["article_no"], "第十二条")
        self.assertEqual(chunks[2]["section_path"], "第二章 休学与复学")
        for c in chunks:
            self.assertTrue(c["chunk_id"].startswith("handbook_2026#c"))

    def test_long_article_is_windowed(self):
        long_doc = dict(DOC, text="第一条 " + "休学与复学的具体规定。" * 80)
        chunks = chunk_mod.chunk_doc(long_doc, max_chars=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"]) <= 200 for c in chunks))
        self.assertTrue(all(c["article_no"] == "第一条" for c in chunks))


class CitationTest(unittest.TestCase):
    chunks = [{"chunk_id": "d#c1", "title": "学生手册", "section_path": "第三章",
               "article_no": "第十二条",
               "text": "学生休学一般以一年为期，累计不得超过两年。"}]

    def test_table_chunk_skips_text_coverage(self):
        tbl = [{"chunk_id": "d#c2", "title": "2026版本科人才培养方案",
                "section_path": "数字经济专业 > 课程计划表", "article_no": "",
                "text": "体育 | 84 | 60 | 144 | 8 | 1—4"}]
        answer = ("数字经济专业体育课共8学分[1]。公共必修课中体育课总学时为144"
                  "（课堂84、实践60），学分为8，开课学期为1—4[1]。")
        self.assertEqual(gen.check_citations(answer, tbl, "数字经济专业体育课多少学分"), [])

    def test_table_still_checks_numbers(self):
        tbl = [{"chunk_id": "d#c2", "title": "t", "section_path": "p", "article_no": "",
                "text": "体育 | 84 | 60 | 144 | 8 | 1—4"}]
        self.assertEqual(gen.check_citations("体育课总学时为160[1]。", tbl, "体育课多少学分"), [1])

    def test_source_includes_section_path(self):
        # 模型写"2026级"，依据其实在 section_path 里，只比正文会误杀
        ch = [{"chunk_id": "d#c3", "title": "北京语言大学2026版本科人才培养方案",
               "section_path": "数字经济专业 > 课程计划表", "article_no": "",
               "text": "体育 | 84 | 60 | 144 | 8 | 1—4"}]
        self.assertEqual(
            gen.check_citations("2026级数字经济专业体育课为8学分[1]。", ch,
                                "数字经济专业体育课多少学分"), [])

    def test_trim_keeps_supported_sentence(self):
        text = "休学累计不得超过两年[1]。退学后不得复学[1]。"
        kept = gen.trim_unsupported(text, self.chunks)
        self.assertIn("休学累计不得超过两年", kept)
        self.assertNotIn("退学后不得复学", kept)

    def test_supported_answer_passes(self):
        self.assertEqual(gen.check_citations("休学累计不得超过两年[1]。", self.chunks), [])

    def test_wrong_number_is_caught(self):
        self.assertEqual(gen.check_citations("休学累计不得超过三年[1]。", self.chunks), [1])

    def test_dangling_index_is_caught(self):
        self.assertEqual(gen.check_citations("休学不得超过两年[3]。", self.chunks), [3])

    def test_abstain_text_passes(self):
        self.assertEqual(gen.check_citations(gen.ABSTAIN_TEXT, self.chunks), [])

    def test_sentence_citing_two_chunks_uses_union(self):
        """一句结论同时用两张卡时，要按两句合集判，不能逐张单独判。

        真实案例：数字经济专业「学制四年，授予经济学学士学位[1][3]」，
        第 1 张卡只有"授予经济学学士学位"，第 3 张卡只有"学制四年"，
        逐张判每张都凑不够 45% 重合，会把正确答案误杀成拒答。
        """
        two = [
            {"chunk_id": "d#c1", "text": "本专业学制四年，授予经济学学士学位。"},
            {"chunk_id": "d#c2", "text": "数字经济专业（020109T），授予经济学学士学位，学制四年。"},
        ]
        got = gen.check_citations("数字经济专业学制四年，授予经济学学士学位[1][2]。", two)
        self.assertEqual(got, [])

    def test_union_does_not_excuse_wrong_number(self):
        two = [
            {"chunk_id": "d#c1", "text": "本专业学制四年，授予经济学学士学位。"},
            {"chunk_id": "d#c2", "text": "数字经济专业，授予经济学学士学位。"},
        ]
        got = gen.check_citations("数字经济专业学制五年，授予经济学学士学位[1][2]。", two)
        self.assertEqual(got, [1, 2])

    def test_year_from_question_is_not_a_hallucination(self):
        """提问里的年份会出现在答案里，片段正文当然没有，不能判成编数字。"""
        chunks = [{"chunk_id": "d#c1",
                   "text": "（一）准予毕业总学分：149 学分。其中，课程学分141 学分，"
                           "专业及毕业实习2 学分，社会实践2 学分，毕业论文（设计）4 学分。"}]
        q = "2026 级数字经济专业准予毕业总学分是多少？"
        a = "2026级数字经济专业准予毕业总学分为149学分[1]。"
        self.assertEqual(gen.check_citations(a, chunks, question=q), [])
        # 但答案里编出来的数字仍然要被抓住
        self.assertEqual(gen.check_citations(a.replace("149", "151"), chunks, question=q), [1])

    def test_restating_subject_from_question_is_not_punished(self):
        """答案照抄提问里的主语（专业名），主语不在片段正文里，不能算它瞎编。"""
        chunks = [{"chunk_id": "d#c1",
                   "text": "（一）准予毕业总学分：149 学分。其中，课程学分141 学分，"
                           "专业及毕业实习2 学分，社会实践2 学分，毕业论文（设计）4 学分。"}]
        q = "金融学（数智金融）专业准予毕业总学分是多少？"
        a = "金融学（数智金融）专业准予毕业总学分为149学分[1]。"
        self.assertEqual(gen.check_citations(a, chunks, question=q), [])
        # 句子后半段凭空补的东西还是要被抓
        bad = "金融学（数智金融）专业准予毕业总学分为149学分，同时要求发表两篇论文[1]。"
        self.assertEqual(gen.check_citations(bad, chunks, question=q), [1])


class EvalMetricTest(unittest.TestCase):
    questions = [
        {"id": "a", "expect_abstain": False, "gold_chunk_ids": ["d1#c1"]},
        {"id": "b", "expect_abstain": False, "gold_chunk_ids": ["d2#c1", "d2#c2"]},
        {"id": "c", "expect_abstain": True, "gold_chunk_ids": []},
    ]
    preds = [
        {"id": "a", "abstained": False, "retrieved_chunk_ids": ["d1#c1"],
         "used_chunk_ids": ["d1#c1"]},
        {"id": "b", "abstained": True, "retrieved_chunk_ids": ["d9#c1"],
         "used_chunk_ids": []},
        {"id": "c", "abstained": True, "retrieved_chunk_ids": ["d3#c1"],
         "used_chunk_ids": []},
    ]

    def test_metrics(self):
        status = {"d1": "active", "d2": "active", "d9": "active"}
        m = eval_mod.evaluate(self.questions, self.preds, status, [1, 5])
        self.assertEqual(m["recall@1"], 0.5)      # 2 条有金标，命中 1 条
        self.assertEqual(m["mrr"], 0.5)
        self.assertEqual(m["false_abstain_rate"], 0.5)  # b 该答却拒答
        self.assertEqual(m["abstain_accuracy"], 1.0)
        self.assertEqual(m["stale_misuse_rate"], 0.0)

    def test_stale_misuse_detected(self):
        preds = [dict(self.preds[0], used_chunk_ids=["d1#c1", "old#c3"])]
        status = {"d1": "active", "old": "superseded"}
        m = eval_mod.evaluate(self.questions[:1], preds, status, [1])
        self.assertEqual(m["stale_misuse_rate"], 1.0)


class MajorSplitTest(unittest.TestCase):
    """培养方案没有条款号，靠「专业 > 小节」定位；表格行不能被劈开。"""

    PLAN = """北京语言大学本科人才培养方案
目录
人工智能专业……………………1
计算机科学与技术专业……………………9
人工智能专业（080717T）
一、培养目标
本专业培养具备人工智能基础理论、工程实践能力和跨学科视野的复合型人才。
二、核心课程
机器学习、深度学习、数据结构与算法、自然语言处理。
学制四年，授予工学学士学位。
计算机科学与技术专业（080901）
一、培养目标
本专业培养计算机系统与软件工程方向的高素质人才。
"""

    def test_auto_falls_back_to_major(self):
        chunks = chunk_mod.chunk_doc({"doc_id": "plan", "text": self.PLAN}, mode="auto")
        paths = [c["section_path"] for c in chunks]
        self.assertTrue(any("人工智能专业" in p for p in paths), paths)
        self.assertTrue(any("计算机科学与技术专业" in p for p in paths), paths)
        self.assertFalse(any("目录" in p for p in paths))

    def test_major_code_kept_in_path(self):
        chunks = chunk_mod.chunk_doc({"doc_id": "plan", "text": self.PLAN}, mode="major")
        self.assertTrue(any("080717T" in c["section_path"] for c in chunks))

    def test_table_rows_not_broken(self):
        rows = "\n".join("第%d学期 | 课程%d | 3.0 | 48 | 必修" % (i % 8, i) for i in range(30))
        text = "人工智能专业（080717T）\n一、课程计划\n【表格】\n" + rows
        chunks = chunk_mod.chunk_doc({"doc_id": "plan", "text": text},
                                     max_chars=120, overlap=20, mode="major")
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            for line in c["text"].splitlines():
                if "|" in line:
                    self.assertEqual(line.count("|"), 4, line)

    def test_window_baseline_is_worse_structured(self):
        """固定窗口基线会切断表格行，结构切块不会——这就是消融实验要报的差异。"""
        rows = "\n".join("第%d学期 | 课程%d | 3.0 | 48 | 必修" % (i % 8, i) for i in range(30))
        text = "人工智能专业\n【表格】\n" + rows
        win = chunk_mod.chunk_doc({"doc_id": "p", "text": text},
                                  max_chars=120, overlap=20, mode="window")
        cut = sum(1 for c in win for line in c["text"].splitlines()
                  if "|" in line and line.count("|") != 4)
        self.assertGreater(cut, 0)

    def test_page_header_becomes_major_marker(self):
        """培养方案的页眉带着专业名，是唯一靠得住的专业边界信号。"""
        text = "\n".join([
            "北京语言大学 2026 版本科人才培养方案 | 数字经济专业",
            "18",
            "四、培养目标",
            "培养数字经济领域的高素质人才。",
            "北京语言大学 2026 版本科人才培养方案 | 金融学（数智金融）专业",
            "22",
            "四、培养目标",
            "培养数智金融方向的复合型人才。",
        ])
        chunks = chunk_mod.chunk_doc({"doc_id": "plan", "text": text, "doc_type": "培养方案"})
        paths = [c["section_path"] for c in chunks]
        self.assertTrue(any("数字经济专业" in p and "四、培养目标" in p for p in paths), paths)
        self.assertTrue(any("金融学" in p for p in paths), paths)
        for c in chunks:
            self.assertNotIn("版本科人才培养方案", c["text"])
            self.assertNotIn("\n18\n", c["text"])

    def test_major_title_line_becomes_marker(self):
        """2025/2024 版没有逐页专业名，靠「XX专业培养方案」标题行定位专业。

        目录里也有同一行标题，所以要靠后面紧跟「一、专业名称」来区分正文标题。
        """
        text = "\n".join([
            "目 录",
            "经济学（数智分析）专业培养方案",
            "数字经济专业培养方案",
            "经济学（数智分析）专业培养方案",
            "一、专业名称",
            "经济学（数智分析），英文名称：Economics",
            "二、所属单位：商学院",
            "数字经济专业培养方案",
            "一、专业名称",
            "数字经济，英文名称：Digital Economic",
            "二、所属单位：商学院",
        ])
        chunks = chunk_mod.chunk_doc({"doc_id": "plan", "text": text, "doc_type": "培养方案"})
        paths = [c["section_path"] for c in chunks]
        self.assertTrue(any(p.startswith("经济学（数智分析）专业") and "专业名称" in p for p in paths), paths)
        self.assertTrue(any(p.startswith("数字经济专业") and "专业名称" in p for p in paths), paths)
        # 标题行本身不该混进正文
        for c in chunks:
            self.assertNotIn("培养方案", c["text"])

    def test_table_keeps_section_in_path(self):
        """表格块要同时带上级小节，否则「哪门课多少学分」没法回指。"""
        text = ("数字经济专业\n七、课程设置与学分要求\n【表格】\n"
                "数字经济概论 | 3.0 | 48 | 必修 | 1\n")
        chunks = chunk_mod.chunk_doc({"doc_id": "plan", "text": text, "doc_type": "培养方案"})
        paths = [c["section_path"] for c in chunks]
        self.assertTrue(all("七、课程设置与学分要求" in p for p in paths if p), paths)
        self.assertTrue(any("课程计划表" in p for p in paths), paths)

    def test_plan_doc_type_beats_embedded_regulation(self):
        """培养方案里嵌一份《…管理办法》时，不能被整本判成规章。"""
        text = ("第一章 总则\n第一条 为规范管理，制定本办法。\n第二条 本办法由教务处负责解释。\n"
                "数字经济专业\n一、培养目标\n培养数字经济人才。")
        chunks = chunk_mod.chunk_doc({"doc_id": "p", "text": text, "doc_type": "培养方案"})
        self.assertTrue(any("数字经济专业" in c["section_path"] for c in chunks))


if __name__ == "__main__":
    unittest.main()
