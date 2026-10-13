"""Unit tests for EchoGuard Predictive Maintenance Advisor module."""

import unittest
from src.advisor import (
    query_knowledge_base,
    generate_maintenance_advisory,
    MAINTENANCE_KNOWLEDGE_BASE
)


class TestAdvisor(unittest.TestCase):
    def test_knowledge_base_populated(self):
        self.assertGreaterEqual(len(MAINTENANCE_KNOWLEDGE_BASE), 5)
        for item in MAINTENANCE_KNOWLEDGE_BASE:
            self.assertIn("id", item)
            self.assertIn("title", item)
            self.assertIn("sop_steps", item)
            self.assertIn("root_causes", item)
            self.assertIn("iso_standard", item)

    def test_query_knowledge_base_outer_race(self):
        results = query_knowledge_base("outer race defect spalling", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "KB-FAULT-BPFO")

    def test_query_knowledge_base_lubrication(self):
        results = query_knowledge_base("grease starvation lubrication", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "KB-FAULT-LUB")

    def test_generate_maintenance_advisory_healthy(self):
        adv = generate_maintenance_advisory(
            prediction="HEALTHY",
            confidence=0.92,
            features={
                "time_domain": {"kurtosis": 2.9, "crest_factor": 2.8, "rms": 0.05},
                "frequency_domain": {"dominant_frequency_hz": 33.3},
                "iso_severity": {"zone": "Zone A"}
            }
        )
        self.assertEqual(adv["severity_level"], "NORMAL")
        self.assertEqual(adv["urgency"], "LOW")
        self.assertIn("Healthy", adv["primary_diagnosis"])
        self.assertGreater(len(adv["sop_action_plan"]), 0)

    def test_generate_maintenance_advisory_faulty_bpfo(self):
        adv = generate_maintenance_advisory(
            prediction="FAULTY",
            confidence=0.15,
            features={
                "time_domain": {"kurtosis": 6.8, "crest_factor": 5.2, "rms": 0.22},
                "frequency_domain": {"dominant_frequency_hz": 236.4},
                "iso_severity": {"zone": "Zone D"}
            }
        )
        self.assertEqual(adv["severity_level"], "CRITICAL")
        self.assertIn("BPFO", adv["primary_diagnosis"])
        self.assertIn("ISO 15243", adv["iso_standard_reference"])
        self.assertGreater(len(adv["root_cause_hypotheses"]), 0)
        self.assertGreater(len(adv["sop_action_plan"]), 0)


if __name__ == "__main__":
    unittest.main()
