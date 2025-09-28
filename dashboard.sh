#!/bin/bash

echo "=========================================="
echo "🧠 SEKAI MEMORY SYSTEM - LIVE DASHBOARD"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""

echo "📊 SYSTEM HEALTH STATUS:"
echo "--------------------------"
curl -s http://localhost:8000/health | jq '{
  database_status: .database_status,
  embedding_status: .embedding_service_status,
  total_memories: .total_memories,
  consistency_score: .consistency_score,
  last_update: .last_update
}'
echo ""

echo "📈 MEMORY STORE STATISTICS:"
echo "----------------------------"
curl -s http://localhost:8000/stats | jq '{
  total_memories: .total_memories,
  average_importance: .avg_importance_score,
  memory_types: .memories_by_type,
  top_3_characters: (.memories_by_character | to_entries | sort_by(.value) | reverse | .[0:3] | from_entries)
}'
echo ""

echo "🔍 SEARCH CAPABILITY DEMO:"
echo "---------------------------"
echo "Query: 'Byleth Dimitri relationship'"
curl -s -X POST http://localhost:8000/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "Byleth Dimitri relationship", "limit": 2}' | jq '{
  total_found: .total_found,
  results: [.results[] | {
    similarity_score: .similarity_score,
    content_preview: (.memory.content[:80] + "..."),
    character_id: .memory.character_id,
    memory_type: .memory.memory_type
  }]
}'
echo ""

echo "⚡ SYSTEM PERFORMANCE:"
echo "----------------------"
start_time=$(date +%s%3N)
curl -s http://localhost:8000/api/v1/characters/1/memories?limit=5 > /dev/null
end_time=$(date +%s%3N)
response_time=$((end_time - start_time))
echo "API Response Time: ${response_time}ms"
echo ""

echo "📋 EVALUATION SUMMARY:"
echo "-----------------------"
if [ -f "eval/results/evaluation_20250928_094606.json" ]; then
    echo "Latest Evaluation Results:"
    cat eval/results/evaluation_20250928_094606.json | jq '{
      consistency_score: .consistency.overall_consistency_score,
      total_issues: .consistency.total_issues,
      evaluation_date: .timestamp
    }'
else
    echo "No recent evaluation results found. Run: python eval/runner.py"
fi

echo ""
echo "=========================================="
echo "✅ Dashboard Complete - System Operational"
echo "=========================================="