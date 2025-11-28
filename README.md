# Lawyer Matching Microservice

A lightweight Python/FastAPI service that scores and ranks lawyers against a given case so that your existing Express backend can request the top `k` candidates. The service combines character-level TF-IDF text similarity (good for Thai and English) with metadata boosts (specializations, licensing status, verification, ratings) to produce transparent scores and explanations.

## Repository Layout

```
/workspace/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .env.example             # Sample configuration for the service
└── lawyer_matching_service/
    ├── lawyer_matching_service/
    │   ├── app.py           # FastAPI entrypoint
    │   ├── config.py        # Environment-driven settings
    │   ├── matcher.py       # TF-IDF ranking logic
    │   ├── providers.py     # Fetches lawyers from Express or local file
    │   ├── schemas.py       # Pydantic models / API contracts
    │   └── data/
    │       └── sample_lawyers.json
    └── __init__.py
```

## Quick Start

1. **Python environment**
   ```bash
   cd /workspace
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure settings**
   ```bash
   cp .env.example .env
   ```
   Update `.env` with:
   - `LAWYER_SOURCE_URL`: HTTP endpoint exposed by your Express API that returns lawyer profiles (`GET /lawyers` or similar).
   - `LAWYER_SOURCE_TOKEN`: Optional Bearer/JWT token if your Express route is protected.
   - `FALLBACK_DATASET_PATH`: Leave empty to use the bundled `sample_lawyers.json`, or point to another JSON file.

3. **Run the matcher API**
   ```bash
   uvicorn lawyer_matching_service.app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Health check**
   ```bash
   curl http://localhost:8000/health
   ```

## Matching Endpoint

- **URL**: `POST /match-lawyers`
- **Body**:
  ```json
  {
    "case": {
      "case_id": "2ed91761-dd1d-458e-aba3-6ffed46f8482",
      "title": "โดนข่มขู่คำร้ายร่างกาย",
      "description": "รายละเอียดคดี...",
      "category": "CRIMINAL",
      "note": "ข้อมูลเพิ่มเติม"
    },
    "top_k": 3,
    "lawyers": []
  }
  ```
  - `lawyers` (optional): Provide candidate lawyers directly if you already filtered them on the Express side. When omitted, the service fetches from `LAWYER_SOURCE_URL`.

- **Response**:
  ```json
  {
    "success": true,
    "count": 3,
    "case_id": "2ed91761-dd1d-458e-aba3-6ffed46f8482",
    "lawyers": [
      {
        "lawyer_id": "lawyer-criminal-001",
        "score": 0.83,
        "reasons": [
          "Text similarity score 0.78",
          "Specialized in criminal cases",
          "Verified by court"
        ],
        "lawyer": { "name": "...", "summary": "...", "criminal_case_specialization": ["ทำร้ายร่างกาย"] }
      }
    ],
    "metadata": {
      "top_k_requested": 3,
      "lawyer_source": "provider",
      "total_candidates": 15
    }
  }
  ```

## Integrating with your Express Backend

1. **Create/Reuse an endpoint** in Express that collects the case payload, calls the Python service, and returns the ranked lawyers to the client.

   ```ts
   import axios from 'axios'

   export const matchLawyers = async (req, res) => {
     const payload = {
       case: req.body.case,
       top_k: req.body.top_k ?? 5,
       // Optionally pass filtered lawyers directly:
       lawyers: req.body.lawyers ?? undefined,
     }

     const matcherHost = process.env.MATCHER_BASE_URL ?? 'http://localhost:8000'
     const response = await axios.post(`${matcherHost}/match-lawyers`, payload, {
       timeout: 5000,
     })

     res.status(200).json(response.data)
   }
   ```

2. **Fetch candidates from Express**: Expose a `GET /lawyers` route that returns the fields required by `LawyerProfile` (summary, description, specializations, pricing, verification flags). The Python service automatically normalizes Prisma-style (`civil_case_specialization`) and Mongoose-style (`civilCase_specialized`) payloads.

3. **Deployment**: Containerize or run the Python service alongside Express. Point `MATCHER_BASE_URL` in Express to the deployed Python host (`http://matcher:8000`). Consider using Docker Compose or Kubernetes to keep both services healthy.

## Data Requirements

Each lawyer record should include:
- Identifier (`lawyer_id`, `_id`, or `user_id`).
- `summary` and `description` text (used for TF-IDF similarity).
- Specialization arrays (`civil_case_specialization`, `criminal_case_specialization`) or camelCase equivalents.
- Optional quality signals: `has_lawyer_license`, `is_verified_by_court`, `avg_rating`, `review_count`, pricing fields.

If no Express endpoint is available yet, edit `lawyer_matching_service/data/sample_lawyers.json` or provide another JSON file via `FALLBACK_DATASET_PATH`.

## How the Ranking Works

1. Build documents for the case and every lawyer (summary + description + specializations).
2. Use character n-gram TF-IDF (`char_wb`, n=3..5) so Thai text and typos still match.
3. Compute cosine similarity between the case vector and each lawyer vector.
4. Apply light bonuses for:
   - Matching category/specialization (criminal vs civil).
   - Verified license / court verification.
   - High rating with enough reviews.
5. Return the top `k` lawyers with a bounded score (0-1) and human-readable reasons.

You can extend `LawyerMatcher._metadata_bonus` to include geography, price ranges, or availability filters.

## Testing the Service

```bash
uvicorn lawyer_matching_service.app:app --port 8000 --reload

curl -X POST http://localhost:8000/match-lawyers \
  -H 'Content-Type: application/json' \
  -d '{
        "case": {
          "case_id": "demo",
          "title": "โดนข่มขู่คำร้ายร่างกาย",
          "description": "รายละเอียดคดี...",
          "category": "CRIMINAL"
        },
        "top_k": 2
      }'
```

The response will use the bundled sample lawyers if `LAWYER_SOURCE_URL` is not set.

## Next Steps & Customization Ideas

- Swap TF-IDF for a transformer embedding model (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) if you need deeper semantics.
- Store lawyer embeddings in a vector database (Faiss, Pinecone) for quicker lookup.
- Add caching or periodic background refresh for large lawyer pools.
- Track feedback to retrain or adjust bonuses automatically.
