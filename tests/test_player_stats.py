"""
玩家数据统计与分析系统 - 验收测试
运行方式：python -m pytest tests/test_player_stats.py -q
共 14 个测试用例
"""
import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.player_stats import PlayerStats


class TestPlayerStats:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.tmpdir, "stats.json")
        self.system = PlayerStats({"storage_path": self.save_path})

    def _record(self, system=None, n=100, wins=30, hit=50, fired=100, seed=0):
        """默认记录 100 场：赢 30 场，命中率 50%。"""
        system = system or self.system
        for i in range(n):
            won = i < wins
            system.record_match(
                match_id=f"m{seed}-{i}",
                won=won,
                kills=5 if won else 2,
                deaths=3,
                damage=1000.0,
                shots_fired=fired,
                shots_hit=hit,
                timestamp=time.time() - (n - i) * 3600,
            )

    # --- 比赛记录 ---

    def test_case_01(self):
        """记录包含击杀、死亡、伤害、命中率等详细数据"""
        assert self.system.record_match(
            "match-1", won=True, kills=10, deaths=2, damage=3500.0,
            shots_fired=20, shots_hit=12,
        )
        m = self.system._matches[0]
        assert m["kills"] == 10
        assert m["deaths"] == 2
        assert m["damage"] == 3500.0
        assert m["shots_hit"] == 12
        assert m["shots_fired"] == 20

    def test_case_02(self):
        """同一 match_id 重复记录被拒绝"""
        assert self.system.record_match("dup", won=True) is True
        assert self.system.record_match("dup", won=False) is False
        assert len(self.system._matches) == 1
        assert self.system._matches[0]["won"] is True

    def test_case_03(self):
        """记录持久化，重启新实例后数据仍在"""
        self._record(n=5, wins=2)
        assert os.path.exists(self.save_path)
        reloaded = PlayerStats({"storage_path": self.save_path})
        assert len(reloaded._matches) == 5
        stats = reloaded.calculate_stats()
        assert stats["wins"] == 2
        assert stats["total_matches"] == 5

    # --- 统计计算 ---

    def test_case_04(self):
        """100 场赢 30 场，胜率必须是 30.0%，不能错算成 50%"""
        self._record(n=100, wins=30)
        stats = self.system.calculate_stats()
        assert stats["win_rate"] == 30.0
        assert stats["wins"] == 30
        assert stats["losses"] == 70

    def test_case_05(self):
        """命中率 50/100 必须是 50.0%，不能错算成 80%"""
        self._record(n=100, wins=30, hit=50, fired=100)
        stats = self.system.calculate_stats()
        assert stats["accuracy"] == 50.0
        assert stats["shots_hit"] == 5000
        assert stats["shots_fired"] == 10000

    def test_case_06(self):
        """考虑样本数量：1 场 100% 胜率不应等同于 100 场 50% 胜率（贝叶斯平滑）"""
        one = PlayerStats({"storage_path": os.path.join(self.tmpdir, "one.json")})
        hundred = PlayerStats({"storage_path": os.path.join(self.tmpdir, "hundred.json")})
        one.record_match("m1", won=True, timestamp=time.time())
        self._record(system=hundred, n=100, wins=50)
        s1 = one.calculate_stats()
        s100 = hundred.calculate_stats()
        assert s1["smoothed_win_rate"] < 100.0
        assert s100["smoothed_win_rate"] == pytest.approx(50.0, abs=1.0)
        assert s100["smoothed_win_rate"] < s1["smoothed_win_rate"]

    def test_case_07(self):
        """考虑时间：近期比赛权重高于远期比赛（时间衰减）"""
        now = time.time()
        self.system.record_match("old1", won=False, timestamp=now - 30 * 86400)
        self.system.record_match("old2", won=False, timestamp=now - 30 * 86400)
        self.system.record_match("new1", won=True, timestamp=now - 3600)
        self.system.record_match("new2", won=True, timestamp=now - 1800)
        stats = self.system.calculate_stats(now=now)
        # 原始胜率 50%，但近期连胜，时间加权胜率应明显偏高
        assert stats["win_rate"] == 50.0
        assert stats["weighted_win_rate"] > 80.0

    def test_case_08(self):
        """精确舍入：3 场胜 1 场 = 33.3%（ROUND_HALF_UP，无浮点误差）"""
        now = time.time()
        self.system.record_match("a", won=True, timestamp=now)
        self.system.record_match("b", won=False, timestamp=now)
        self.system.record_match("c", won=False, timestamp=now)
        assert self.system.calculate_stats(now=now)["win_rate"] == 33.3

    # --- 趋势分析 ---

    def test_case_09(self):
        """趋势方向：命中率从 30% 提升到 90%，应为 improving"""
        now = time.time()
        for i in range(4):
            self.system.record_match(
                f"lo{i}", won=False, shots_fired=100, shots_hit=30,
                timestamp=now - (8 - i) * 86400,
            )
        for i in range(4):
            self.system.record_match(
                f"hi{i}", won=True, shots_fired=100, shots_hit=90,
                timestamp=now - (4 - i) * 86400,
            )
        trend = self.system.trend_analysis("accuracy", days=30, now=now)
        assert trend["direction"] == "improving"
        assert trend["change"] > 0

    def test_case_10(self):
        """时间范围：7 天外的旧比赛不计入趋势窗口"""
        now = time.time()
        self.system.record_match("old", won=False, shots_fired=100, shots_hit=10,
                                 timestamp=now - 20 * 86400)
        for i in range(2):
            self.system.record_match(
                f"new{i}", won=True, shots_fired=100, shots_hit=80,
                timestamp=now - (2 - i) * 86400,
            )
        trend7 = self.system.trend_analysis("accuracy", days=7, now=now)
        assert trend7["matches_in_window"] == 2
        assert trend7["recent"] == 80.0
        trend30 = self.system.trend_analysis("accuracy", days=30, now=now)
        assert trend30["matches_in_window"] == 3

    def test_case_11(self):
        """玩家对比：可以和其他玩家比较，给出均值差与百分位"""
        now = time.time()
        weak = PlayerStats({"storage_path": os.path.join(self.tmpdir, "weak.json")})
        strong = PlayerStats({"storage_path": os.path.join(self.tmpdir, "strong.json")})
        self._record(system=weak, n=20, wins=2, seed=1)
        self._record(system=strong, n=20, wins=19, seed=2)
        self._record(system=self.system, n=20, wins=10, seed=3)
        trend = self.system.trend_analysis(
            "win_rate", days=60, compare_with=[weak, strong], now=now
        )
        cmp = trend["comparison"]
        assert cmp is not None
        assert cmp["others_average"] == pytest.approx(52.5, abs=0.1)
        assert cmp["better_than_average"] is False
        assert 0 < cmp["percentile"] < 100

    # --- 成就追踪 ---

    def test_case_12(self):
        """成就解锁：达到条件后自动解锁"""
        now = time.time()
        for i in range(5):
            self.system.record_match(f"w{i}", won=True, timestamp=now + i)
        unlocked = {a["id"]: a for a in self.system.achievement_tracking(now=now)}
        assert unlocked["first_win"]["unlocked"] is True
        assert unlocked["winner_10"]["unlocked"] is False
        assert unlocked["streak_5"]["unlocked"] is True

    def test_case_13(self):
        """成就进度与稀有度：显示 progress/target/百分比，且稀有度分级"""
        now = time.time()
        self.system.record_match("w", won=True, timestamp=now)
        achievements = {a["id"]: a for a in
                        self.system.achievement_tracking(now=now)}
        ten = achievements["winner_10"]
        assert ten["progress"] == 1
        assert ten["target"] == 10
        assert ten["progress_pct"] == 10.0
        assert ten["rarity"] == "common"
        assert achievements["veteran_100"]["rarity"] == "rare"
        assert achievements["sharpshooter"]["rarity"] == "epic"
        assert achievements["legend_1000"]["rarity"] == "legendary"

    # --- 统计导出 ---

    def test_case_14(self):
        """导出支持 JSON/CSV/文本，并能生成社交分享内容"""
        self._record(n=5, wins=2)
        data = json.loads(self.system.stats_export("json"))
        assert data["stats"]["total_matches"] == 5
        assert len(data["matches"]) == 5

        csv_text = self.system.stats_export("csv")
        assert "match_id,won,kills,deaths,damage,shots_fired,shots_hit,timestamp" in csv_text
        assert csv_text.count("m0-") == 5

        text = self.system.stats_export("text")
        assert "胜率: 40.0%" in text

        share = self.system.share("twitter")
        assert "胜率 40.0%" in share["text"]
        assert share["url"].startswith("https://twitter.com/intent/tweet?text=")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
