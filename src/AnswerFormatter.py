class AnswerFormatter:

    def __init__(self):
        pass

    def format_json(self, query, results):
        formatted = {
            "query": query,
            "top_k": len(results),
            "answers": []
        }

        for item in results:
            meta = item["metadata"]

            formatted["answers"].append({
                "score": item["final_score"],
                "source": meta["source"],
                "section": meta["section"],
                "chunk_id": meta["chunk_id"],
                "text": item["chunk"]
            })

        return formatted

    def format_pretty(self, query, results):
        lines = []
        lines.append("=" * 100)
        lines.append(f"QUERY: {query}")
        lines.append("=" * 100)

        for i, item in enumerate(results, 1):
            meta = item["metadata"]

            lines.append(f"\n[{i}]  Score: {item['final_score']:.4f}")
            lines.append(f"     Source : {meta['source']}")
            lines.append(f"     Section: {meta['section']}")
            lines.append(f"     ChunkID: {meta['chunk_id']}")
            lines.append("-" * 100)
            lines.append(item["chunk"][:600] + " ...")

        lines.append("\n" + "=" * 100)
        lines.append("CITED SOURCES:")
        cited = set()
        for item in results:
            meta = item["metadata"]
            cited.add(f"{meta['source']} (section={meta['section']}, chunk={meta['chunk_id']})")

        for c in cited:
            lines.append(" - " + c)

        lines.append("=" * 100)

        return "\n".join(lines)
