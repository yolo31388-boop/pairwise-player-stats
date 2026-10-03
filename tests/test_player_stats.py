"""
玩家数据统计与分析系统 - 验收测试
运行方式：python -m pytest tests/test_player_stats.py -q
共 14 个测试用例
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.player_stats import PlayerStats

NOW = 1_700_000_000.0
DAY = 86400.0


def make_match(match_id, win=False, kills=0, deaths=0, damage=0.0,
               hits=0, shots=0, timestamp=NOW):
    return {
        "match_id": match_id,
        "win": win,
        "kills": kills,
        "deaths": deaths,
        "damage": damage,
        "hits": hits,
        "shots": shots,
        "timestamp": timestamp,
    }


class TestPlayerStats:
    def setup_method(self, tmp_path=None):
        self.system = PlayerStats()

    # 1. 比赛记录包含完整详细数据
    def test_case_01(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        system.record_match(make_match(
            "m1", win=True, kills=10, deaths=2, damage=1500.0,
            hits=30, shots=60, timestamp=NOW))
        record = system._history[0]
        assert record["win"] is True
        assert record["kills"] == 10
        assert record["deaths"] == 2
        assert record["damage"] == 1500.0
        assert record["hits"] == 30
        assert record["shots"] == 60
        assert record["timestamp"] == NOW

    # 2. 重复比赛不重复记录
    def test_case_02(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        assert system.record_match(make_match("m1", win=True)) is True
        assert system.record_match(make_match("m1", win=False, kills=99)) is False
        assert len(system._history) == 1
        assert system._history[0]["win"] is True

    # 3. 记录持久化，重启后数据不丢失
    def test_case_03(self, tmp_path):
        data_file = str(tmp_path / "data.json")
        system = PlayerStats({"data_file": data_file})
        for i in range(5):
            system.record_match(make_match(f"m{i}", win=i % 2 == 0, kills=i))
        # 模拟重启：新实例加载同一数据文件
        restarted = PlayerStats({"data_file": data_file})
        assert len(restarted._history) == 5
        stats = restarted.calculate_stats()
        assert stats["matches"] == 5
        assert stats["wins"] == 3

    # 4. 胜率精确计算：100 局赢 30 局必须是 30%
    def test_case_04(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        for i in range(100):
            system.record_match(make_match(f"m{i}", win=i < 30))
        stats = system.calculate_stats()
        assert stats["win_rate"] == pytest.approx(0.3)
        assert stats["wins"] == 30
        assert stats["losses"] == 70

    # 5. 命中率精确计算：命中 50 / 射击 100 必须是 50%
    def test_case_05(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        system.record_match(make_match("m1", hits=50, shots=100))
        stats = system.calculate_stats()
        assert stats["accuracy"] == pytest.approx(0.5)

    # 6. 统计考虑样本数量：1 局全胜与 100 局全胜权重不同
    def test_case_06(self, tmp_path):
        small = PlayerStats({"data_file": str(tmp_path / "small.json")})
        small.record_match(make_match("m1", win=True, hits=10, shots=10))
        large = PlayerStats({"data_file": str(tmp_path / "large.json")})
        for i in range(100):
            large.record_match(make_match(f"m{i}", win=True, hits=10, shots=10))
        small_stats = small.calculate_stats()
        large_stats = large.calculate_stats()
        # 小样本的调整后胜率应向 50% 收缩，且置信度更低
        assert small_stats["adjusted_win_rate"] < large_stats["adjusted_win_rate"]
        assert small_stats["adjusted_win_rate"] < 1.0
        assert small_stats["confidence"] < large_stats["confidence"]
        assert large_stats["adjusted_win_rate"] == pytest.approx(1.0, abs=0.05)

    # 7. 统计考虑时间：最近的比赛权重高于一个月前的比赛
    def test_case_07(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json"),
                              "half_life_days": 7})
        # 30 天前全输，最近全赢
        for i in range(10):
            system.record_match(make_match(f"old{i}", win=False,
                                           timestamp=NOW - 30 * DAY))
        for i in range(10):
            system.record_match(make_match(f"new{i}", win=True, timestamp=NOW))
        stats = system.calculate_stats(now=NOW)
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["weighted_win_rate"] > 0.9

    # 8. 趋势分析能识别命中率上升
    def test_case_08(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        # 前半段命中率 20%，后半段 80%
        for i in range(5):
            system.record_match(make_match(f"a{i}", hits=20, shots=100,
                                           timestamp=NOW - (10 - i) * DAY))
        for i in range(5):
            system.record_match(make_match(f"b{i}", hits=80, shots=100,
                                           timestamp=NOW - (5 - i) * DAY))
        trend = system.trend_analysis(metric="accuracy", days=30, now=NOW)
        assert trend["direction"] == "improving"
        assert trend["change"] > 0
        assert trend["second_half"] > trend["first_half"]

    # 9. 趋势分析支持时间范围（7 天 vs 30 天）
    def test_case_09(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        for i in range(5):
            system.record_match(make_match(f"old{i}", timestamp=NOW - 20 * DAY))
        for i in range(3):
            system.record_match(make_match(f"new{i}", timestamp=NOW - 2 * DAY))
        trend_7d = system.trend_analysis(days=7, now=NOW)
        trend_30d = system.trend_analysis(days=30, now=NOW)
        assert trend_7d["days"] == 7
        assert trend_7d["matches"] == 3
        assert trend_30d["matches"] == 8

    # 10. 趋势分析支持和其他玩家对比
    def test_case_10(self, tmp_path):
        me = PlayerStats({"data_file": str(tmp_path / "me.json")})
        other = PlayerStats({"data_file": str(tmp_path / "other.json")})
        for i in range(10):
            me.record_match(make_match(f"me{i}", hits=80, shots=100, timestamp=NOW))
            other.record_match(make_match(f"ot{i}", hits=40, shots=100, timestamp=NOW))
        trend = me.trend_analysis(metric="accuracy", days=30,
                                  compare_with=other, now=NOW)
        comparison = trend["comparison"]
        assert comparison["my_value"] == pytest.approx(0.8)
        assert comparison["others_average"] == pytest.approx(0.4)
        assert comparison["rank"] == 1
        assert comparison["players"] == 2
        assert comparison["better_than_percent"] == pytest.approx(100.0)

    # 11. 成就达成后解锁
    def test_case_11(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        system.record_match(make_match("m1", win=True))
        achievements = {a["id"]: a for a in system.achievement_tracking()}
        assert achievements["first_win"]["unlocked"] is True
        assert achievements["ten_wins"]["unlocked"] is False

    # 12. 成就有进度显示
    def test_case_12(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        for i in range(5):
            system.record_match(make_match(f"m{i}", win=True, kills=10))
        achievements = {a["id"]: a for a in system.achievement_tracking()}
        assert achievements["ten_wins"]["progress"] == pytest.approx(0.5)
        assert achievements["ten_wins"]["current"] == 5
        assert achievements["ten_wins"]["target"] == 10
        assert achievements["killer"]["progress"] == pytest.approx(0.5)

    # 13. 成就有稀有度区分
    def test_case_13(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        achievements = system.achievement_tracking()
        rarities = {a["rarity"] for a in achievements}
        assert {"common", "rare", "epic", "legendary"} <= rarities
        by_id = {a["id"]: a for a in achievements}
        assert by_id["first_win"]["rarity"] == "common"
        assert by_id["legend"]["rarity"] == "legendary"

    # 14. 统计导出支持多种格式和分享
    def test_case_14(self, tmp_path):
        system = PlayerStats({"data_file": str(tmp_path / "data.json")})
        system.record_match(make_match("m1", win=True, kills=5, hits=50, shots=100))
        # JSON 格式
        json_export = system.stats_export(format="json")
        parsed = json.loads(json_export["data"])
        assert parsed["stats"]["matches"] == 1
        assert len(parsed["matches"]) == 1
        # CSV 格式
        csv_export = system.stats_export(format="csv")
        assert "match_id" in csv_export["data"]
        assert "m1" in csv_export["data"]
        # 文本格式
        text_export = system.stats_export(format="text")
        assert "胜率" in text_export["data"]
        # 分享
        share_export = system.stats_export(format="json", share=True)
        assert "share_text" in share_export
        assert "share_url" in share_export
        assert "胜率" in share_export["share_text"]
        # 非法格式报错
        with pytest.raises(ValueError):
            system.stats_export(format="xml")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
