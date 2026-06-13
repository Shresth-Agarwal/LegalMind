import json
from pathlib import Path
from src.retriever import load_indexes, retrieve

chunks, bm25, faiss_index, model = load_indexes()

# 25 test questions — one per IPC topic
QUESTIONS = [
    "What is the rarest of rare doctrine for imposing death penalty?",
    "What are the criteria laid down in Machhi Singh for death penalty?",
    "What is the difference between culpable homicide and murder?",
    "What was the Nanavati case about and what did the court decide?",
    "How does circumstantial evidence apply in murder cases?",
    "What is the punishment prescribed under Section 302 IPC?",
    "What are the guidelines for arrest under Section 498A?",
    "What constitutes cruelty under Section 498A IPC?",
    "What is criminal conspiracy under Section 120B IPC?",
    "What are the rights of rape victims under Indian law?",
    "What did the court say about sexual assault in Krishnappa case?",
    "What is the legal definition of cheating under Section 420 IPC?",
    "What is attempt to murder under Section 307 IPC?",
    "What is the punishment for dacoity under Section 395?",
    "What is outraging the modesty of a woman under Section 354?",
    "What did Shreya Singhal case decide about free speech?",
    "What is the basic structure doctrine from Kesavananda Bharati?",
    "What rights does Article 21 protect according to Maneka Gandhi?",
    "What is the definition of theft under Section 379 IPC?",
    "What is culpable homicide not amounting to murder under Section 304?",
    "What evidence is needed to prove fraud under IPC?",
    "What are the procedural safeguards before arresting someone?",
    "What is the doctrine of mens rea in Indian criminal law?",
    "How does the court determine sentence in heinous crimes?",
    "What is the significance of motive in circumstantial evidence cases?"
]

dataset = []

for q in QUESTIONS:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print(f"{'='*60}")

    results = retrieve(q, chunks, bm25, faiss_index, model, top_k=10)

    for i, r in enumerate(results[:5], 1):
        print(f"\n[{i}] chunk_index={r['chunk_index']} | {r['source']}")
        print(f"    {r['text'][:300]}...")

    print("\nEnter the chunk_index of the BEST answer (or 's' to skip): ", end="")
    ans = input().strip()

    if ans == 's':
        print("Skipped.")
        continue

    dataset.append({
        "question": q,
        "relevant_chunk_index": int(ans)
    })
    print(f"Saved: chunk {ans}")

# Save dataset
with open("eval_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"\n Saved {len(dataset)} questions to eval_dataset.json")