#!/usr/bin/env bash
# Curl examples against the GLM-OCR FastAPI service.
# Assumes the server is running at http://localhost:8889.
set -euo pipefail
BASE="${BASE:-http://localhost:8889}"

echo "# Health"
curl -sS "$BASE/health" | python3 -m json.tool
echo

echo "# Parse a single image (task=text)"
curl -sS -X POST "$BASE/ocr/parse" \
  -F "files=@samples/invoice.png" \
  -F "task=text" | python3 -m json.tool
echo

echo "# Parse a PDF (each page becomes its own result; pdf2image splits them)"
curl -sS -X POST "$BASE/ocr/parse" \
  -F "files=@samples/document.pdf" \
  -F "task=text" | python3 -m json.tool
echo

echo "# Table recognition"
curl -sS -X POST "$BASE/ocr/parse" \
  -F "files=@samples/table.png" \
  -F "task=table" | python3 -m json.tool
echo

echo "# Formula recognition"
curl -sS -X POST "$BASE/ocr/parse" \
  -F "files=@samples/formula.png" \
  -F "task=formula" | python3 -m json.tool
echo

echo "# Custom prompt (targeted extraction)"
curl -sS -X POST "$BASE/ocr/parse" \
  -F "files=@samples/invoice.png" \
  -F "task=custom" \
  -F "prompt=Extract the invoice number and total amount." | python3 -m json.tool
echo

echo "# Parse by URL (JSON body)"
curl -sS -X POST "$BASE/ocr/parse/url" \
  -H "Content-Type: application/json" \
  -d '{"images": ["https://raw.githubusercontent.com/zai-org/GLM-OCR/main/examples/source/code.png"], "task": "text"}' \
  | python3 -m json.tool
