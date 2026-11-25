"""The cloud GPU hours table: the sums, without reading any job record."""
import unittest

from lir import cloudhours


class Summary(unittest.TestCase):
    def test_hours_and_cost_by_gpu_type_with_failed_jobs_counted(self):
        jobs = [{"flavor": "l4x1", "stage": "COMPLETED", "running_secs": 3600}, {"flavor": "l4x1", "stage": "ERROR", "running_secs": 1800},
                {"flavor": "l40sx1", "stage": "COMPLETED", "running_secs": 7200}, {"flavor": "other", "stage": "COMPLETED", "running_secs": None}]
        res = cloudhours.summary(jobs)
        self.assertEqual(res["by_gpu"]["l4x1"], {"jobs": 2, "completed": 1, "running_hours": 1.5, "list_price_dollars": 1.2})
        self.assertEqual(res["by_gpu"]["l40sx1"]["list_price_dollars"], 3.6)
        self.assertIsNone(res["by_gpu"]["other"]["list_price_dollars"])
        self.assertEqual((res["jobs"], res["running_hours"], res["list_price_dollars"]), (4, 3.5, 4.8))


if __name__ == "__main__":
    unittest.main()
