# DualGraph DeepResearch

Interactive visualization and demo for the **DualGraph** deep research framework.

**Paper:** [DualGraph](https://arxiv.org/abs/2602.13830)

## How to Use

- **Demo mode**: Type `demo` to replay a pre-generated research report (Case 1). No LLM calls required.
- **Live mode**: Enter any research question to generate a new report in real-time.

## What You'll See

1. **Outline** — The research outline at each iteration, shown as expandable steps
2. **Search Queries** — The queries generated per iteration
3. **Report** — The final deep-research report in Markdown
4. **References** — All cited sources with clickable links

## Settings

Click the gear icon to adjust:
- **Language**: English / Chinese
- **Max Iterations**: Number of search-update cycles (default: 5)

## Architecture

The system follows an iterative **Outline-guided + Knowledge Graph (OG+KG)** approach:

1. Generate an initial outline from the query
2. Produce search queries from the outline and knowledge graph
3. Search the web, extract evidence, and build/update the knowledge graph
4. Update the outline with new findings
5. Repeat until convergence or max iterations
6. Generate the final report section by section

## Links

- [GitHub Repository](https://github.com/inclusionAI/AWorld)
- [arXiv Paper](https://arxiv.org/abs/2602.13830)
