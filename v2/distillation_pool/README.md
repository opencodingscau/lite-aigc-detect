# Distillation pool (Pilot B)

1. Inventory candidate reals + fakes (licenses OK).  
2. Hash all paths → `dedup/pool_sha256.txt`.  
3. Subtract paper test manifest IDs/hashes → `dedup/blocked_from_paper_test.txt`.  
4. Emit `../manifests/pilot_b/distill_pool.jsonl`.  
5. Teachers (NPR, UnivFD) write soft labels to `../outputs/pilot_b/teacher_soft/` (gitignored).

Do not use paper UFD/DALL·E test images for distillation.
