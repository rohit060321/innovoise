import sys
from pathlib import Path
import unittest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from fastapi.testclient import TestClient
from morphe.api.main import app, get_system


class TestMorpheAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Ensure system is initialized
        get_system()

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["system"], "MORPHE")
        self.assertIn("active_version", data)
        self.assertGreater(data["module_count"], 0)

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})

    def test_query_endpoint(self):
        payload = {
            "query": "How does Raft elect a leader and maintain log replication?",
            "top_k": 3,
        }
        resp = self.client.post("/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("records", data)
        self.assertIn("queried_modules", data)
        self.assertIn("latency_ms", data)
        self.assertGreater(len(data["records"]), 0)

    def test_graph_endpoint(self):
        resp = self.client.get("/graph")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("modules", data)
        self.assertIn("edges", data)

    def test_modules_endpoint(self):
        resp = self.client.get("/modules")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_candidates_endpoint(self):
        resp = self.client.get("/adapt/candidates")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), list)

    def test_sandbox_trial(self):
        sys_inst = get_system()
        mods = list(sys_inst.graph.modules.keys())
        payload = {
            "operation_type": "split",
            "module_id": mods[0],
        }
        resp = self.client.post("/adapt/trial", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("decision", data)
        self.assertFalse(data["production_memory_modified"])

    def test_harmful_rollback_demo(self):
        resp = self.client.post("/adapt/rollback_demo")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["validator_decision"], "REJECTED")
        self.assertTrue(data["sandbox_discarded"])
        self.assertTrue(data["production_state_intact"])

    def test_versions_endpoint(self):
        resp = self.client.get("/versions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)


if __name__ == "__main__":
    unittest.main()
