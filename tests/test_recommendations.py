from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np


class FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        documents,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
        single = isinstance(documents, str)
        items = [documents] if single else documents
        vectors = np.array([self._vectorize(item) for item in items], dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        return vectors[0] if single else vectors

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        buckets = {
            "criminal": ("อาญา", "ทำร้าย", "ประกันตัว", "ข่มขู่", "ทะเลาะ"),
            "family": ("ครอบครัว", "หย่า", "บุตร", "ค่าเลี้ยงดู", "มรดก"),
            "cyber": ("ไซเบอร์", "ออนไลน์", "ฉ้อโกง", "โอนเงิน", "บัญชีม้า", "ดิจิทัล"),
        }
        lowered = text.lower()
        scores = []
        for words in buckets.values():
            scores.append(float(sum(lowered.count(word.lower()) for word in words)))
        scores.append(1.0)
        return scores


fake_module = types.ModuleType("sentence_transformers")
fake_module.SentenceTransformer = FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", fake_module)

from app import create_app
from app.config import AppConfig
from app.routes import _case_to_prompt
from app.services.lawyer_ranker import LawyerRanker


class RecommendationApiTest(unittest.TestCase):
    def setUp(self) -> None:
        data_path = Path(__file__).resolve().parents[1] / "data" / "lawyers.test.json"
        app = create_app(
            AppConfig(
                model_name="fake-test-model",
                lawyer_data_source=str(data_path),
                default_top_k=2,
                cors_origins=["*"],
                debug=False,
                port=8000,
            )
        )
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health_reports_model(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["model"], "fake-test-model")

    def test_embeddings_returns_vector_for_lawyer_text(self):
        response = self.client.post(
            "/embeddings",
            json={
                "lawyer": {
                    "slogan": "เชี่ยวชาญคดีออนไลน์",
                    "summary": "ช่วยผู้เสียหายถูกหลอกโอนเงิน",
                    "description": "ตรวจพยานหลักฐานดิจิทัลและบัญชีม้า",
                    "ประเภทงานที่เชี่ยวชาญ": ["คดีอาญา", "อาชญากรรมไซเบอร์"],
                }
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["model"], "fake-test-model")
        self.assertEqual(payload["dimension"], 4)
        self.assertEqual(len(payload["embedding"]), 4)

    def test_recommendations_rank_best_lawyer_for_cyber_case(self):
        response = self.client.post(
            "/recommendations",
            json={
                "case": {
                    "ประเภทของงาน": "คดีอาญา",
                    "หัวข้อบริการ": "อาชญากรรมไซเบอร์",
                    "หัวข้องาน": "ถูกหลอกโอนเงินออนไลน์",
                    "รายละเอียดงาน": "ลูกค้าถูกฉ้อโกงออนไลน์ มีหลักฐานแชทและสลิปโอนเงิน",
                    "หมายเหตุเพิ่มเติม": "ต้องการตรวจพยานหลักฐานดิจิทัล",
                },
                "top_k": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["data"][0]["lawyer_id"], "test-cyber-001")
        self.assertGreaterEqual(payload["data"][0]["score"], payload["data"][1]["score"])

    def test_recommendations_can_use_stored_lawyer_embeddings(self):
        response = self.client.post(
            "/recommendations",
            json={
                "case": {
                    "ประเภทของงาน": "คดีอาญา",
                    "หัวข้อบริการ": "อาชญากรรมไซเบอร์",
                    "หัวข้องาน": "ถูกหลอกโอนเงินออนไลน์",
                    "รายละเอียดงาน": "ลูกค้าถูกฉ้อโกงออนไลน์",
                },
                "lawyers": [
                    {
                        "lawyer_id": "stored-cyber",
                        "slogan": "ข้อความนี้ไม่ควรถูกใช้",
                        "lawyer_embedding": [0.0, 0.0, 1.0, 0.0],
                        "ประเภทงานที่เชี่ยวชาญ": ["คดีอาญา"],
                    },
                    {
                        "lawyer_id": "stored-family",
                        "slogan": "ข้อความนี้ไม่ควรถูกใช้",
                        "lawyer_embedding": [0.0, 1.0, 0.0, 0.0],
                        "ประเภทงานที่เชี่ยวชาญ": ["คดีอาญา"],
                    },
                ],
                "top_k": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["data"][0]["lawyer_id"], "stored-cyber")

    def test_recommendations_can_build_prompt_from_case_payload(self):
        response = self.client.post(
            "/recommendations",
            json={
                "case_id": "case-family-001",
                "case": {
                    "หัวข้องาน": "ข้อพิพาทค่าเลี้ยงดู",
                    "ประเภทของงาน": "คดีแพ่ง",
                    "หัวข้อบริการ": "หย่าร้าง",
                    "รายละเอียดงาน": "ต้องการฟ้องหย่าและขออำนาจปกครองบุตร",
                    "หมายเหตุเพิ่มเติม": "แยกกันอยู่แล้ว 6 เดือน",
                    "client": {"name": "Test Client", "tel": "0800000000"},
                    "timelines": [{"title": "แยกกันอยู่แล้ว 6 เดือน"}],
                    "appointments": [],
                },
                "top_k": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["case_id"], "case-family-001")
        self.assertEqual(payload["data"][0]["lawyer_id"], "test-family-001")

    def test_recommendations_filter_by_case_category_before_ranking(self):
        response = self.client.post(
            "/recommendations",
            json={
                "case": {
                    "ประเภทของงาน": "คดีแพ่ง",
                    "หัวข้อบริการ": "หย่าร้าง",
                    "หัวข้องาน": "ขอคำปรึกษาคดีหย่า",
                    "รายละเอียดงาน": "ต้องการฟ้องหย่าและตกลงค่าเลี้ยงดู",
                    "หมายเหตุเพิ่มเติม": "มีเรื่องทรัพย์สินและบุตร",
                },
                "top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["data"][0]["lawyer_id"], "test-family-001")

    def test_recommendations_filter_requires_only_one_expertise_overlap(self):
        response = self.client.post(
            "/recommendations",
            json={
                "case": {
                    "ประเภทของงาน": "คดีแพ่ง",
                    "หัวข้อบริการ": "อาชญากรรมไซเบอร์",
                    "หัวข้องาน": "ถูกหลอกออนไลน์",
                    "รายละเอียดงาน": "มีหลักฐานแชทและสลิปโอนเงินจากการหลอกออนไลน์",
                },
                "top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        lawyer_ids = {item["lawyer_id"] for item in payload["data"]}
        self.assertIn("test-family-001", lawyer_ids)
        self.assertIn("test-cyber-001", lawyer_ids)

    def test_recommendations_return_empty_when_filter_has_no_matching_lawyers(self):
        response = self.client.post(
            "/recommendations",
            json={
                "case": {
                    "ประเภทของงาน": "คดีปกครอง/รัฐ",
                    "หัวข้อบริการ": "สิ่งแวดล้อม",
                    "หัวข้องาน": "ข้อพิพาทสิ่งแวดล้อม",
                    "รายละเอียดงาน": "ต้องการทนายด้านสิ่งแวดล้อม",
                },
                "top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["count"], 0)
        self.assertEqual(payload["data"], [])

    def test_recommendations_reject_empty_description(self):
        response = self.client.post("/recommendations", json={})

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("`description` or `case` data is required", payload["message"])

    def test_recommendations_reject_invalid_lawyers_payload(self):
        response = self.client.post(
            "/recommendations",
            json={"description": "คดีทั่วไป", "lawyers": {"lawyer_id": "bad"}},
        )

        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertIn("`lawyers` must be an array", payload["message"])

    def test_lawyer_score_document_uses_only_allowed_lawyer_fields(self):
        document = LawyerRanker._lawyer_to_document(
            {
                "lawyer_id": "not-for-score",
                "name": "ไม่ควรใช้ชื่อ",
                "slogan": "สโลแกนสำหรับคำนวณ",
                "summary": "สรุปสำหรับคำนวณ",
                "description": "รายละเอียดสำหรับคำนวณ",
                "ประเภทงานที่เชี่ยวชาญ": ["คดีอาญา", "อาชญากรรมไซเบอร์"],
                "languages": ["ไทย"],
                "law_firm": "ไม่ควรใช้สำนักงาน",
                "avg_rating": 5.0,
            }
        )

        self.assertIn("สโลแกนสำหรับคำนวณ", document)
        self.assertIn("สรุปสำหรับคำนวณ", document)
        self.assertIn("รายละเอียดสำหรับคำนวณ", document)
        self.assertIn("อาชญากรรมไซเบอร์", document)
        self.assertNotIn("ไม่ควรใช้ชื่อ", document)
        self.assertNotIn("ไม่ควรใช้สำนักงาน", document)
        self.assertNotIn("ไทย", document)
        self.assertNotIn("5.0", document)

    def test_case_score_document_uses_only_allowed_case_fields(self):
        document = _case_to_prompt(
            {
                "ประเภทของงาน": "คดีอาญา",
                "หัวข้อบริการ": "อาชญากรรมไซเบอร์",
                "หัวข้องาน": "ถูกหลอกโอนเงิน",
                "รายละเอียดงาน": "มีแชทและสลิปโอนเงิน",
                "หมายเหตุเพิ่มเติม": "ต้องการดำเนินคดีเร็ว",
                "client": {"name": "ไม่ควรใช้ชื่อลูกความ"},
                "case_id": "not-for-score",
            },
            root_payload={},
        )

        self.assertIn("คดีอาญา", document)
        self.assertIn("อาชญากรรมไซเบอร์", document)
        self.assertIn("ถูกหลอกโอนเงิน", document)
        self.assertIn("มีแชทและสลิปโอนเงิน", document)
        self.assertIn("ต้องการดำเนินคดีเร็ว", document)
        self.assertNotIn("ไม่ควรใช้ชื่อลูกความ", document)
        self.assertNotIn("not-for-score", document)


if __name__ == "__main__":
    unittest.main()
