# FlashInfer AI Kernel Generation Contest @ MLSys 2026 — Technical Write-up Requirements

Congratulations on your performance in the contest\! To be eligible for awards, you must submit a technical write-up describing your solution.

## Deadline

**May 1, 2026, 11:59 PM AoE (Anywhere on Earth)**

## Format

- **Max 4 pages** for a single track, PDF format, free layout (no template required)
- If covering **multiple tracks** in one report: max 4 \+ 2 per additional track (2 tracks \= 6 pages, 3 tracks \= 8 pages)
- Reasonable font size (11pt or larger)
- References do not count toward the page limit
- If you participated in **both** Full-Agent and Agent-Assisted approaches, please submit **separate** write-ups for each

## Submission

Submit your write-up (PDF) via this Google Form:

**https://docs.google.com/forms/d/e/1FAIpQLSdeffx8ZEtd5Hud6wstynRNEP8hHlP45n7oq-kb4w3SsbxFMA/viewform**

## Content Guidelines

Your write-up should cover the following areas. Not every section is required — focus on the parts most relevant to your solution.

### For All Submissions

1. **Solution Overview** — High-level description of your approach and key ideas
2. **Kernel Design & Optimizations** — Core techniques used (e.g., tiling strategy, memory access patterns, warp specialization, use of Blackwell-specific features like UMMA, TMA, etc.)
3. **Performance Analysis** — Profiling results, bottleneck analysis, and how you addressed them. Include speedup numbers across workloads if possible
4. **Tools & Languages** — What you used (Triton, CUDA, CuTe-DSL, CuTile, Tilelang, etc.) and why

### Additional for Full-Agent Submissions

5. **Agent Architecture & Workflow** — Describe the agent system: prompt design, search/evolution strategy, feedback loop, how correctness and performance are verified
6. **Reproducibility** (optional but recommended) — Instructions or code to reproduce the kernel generation process

### Additional for Agent-Assisted Submissions

5. **Human vs. Agent Contributions** — What parts were human-designed vs. agent-generated? How did the collaboration work?

## Notes

- Award decisions will be announced by **May 11, 2026**. Teams without a submitted write-up will not be considered for awards
- Write-ups may be published on the contest website with your permission
