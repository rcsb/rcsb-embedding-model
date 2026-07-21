# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, breaking changes are released in a minor bump.

## [0.8.0] - 2026-07-21

This release reworks how `fm-search` reports results and makes its alignment
statistics calibrated. **It contains breaking changes to every `fm-search query`
command and to `cluster` / `similarity-graph`** — see *Migration* below.

### Added

- **Selectable output columns.** Sequence-identity (Stage-2) searches accept
  `--format-output`, a comma-separated list of columns, e.g.
  `--format-output "query,target,evalue,qaln,taln"`. Twenty-five columns are
  supported: `query`, `target`, `embscore`, `evalue`, `gapopen`, `pident`,
  `fident`, `nident`, `qstart`, `qend`, `qlen`, `tstart`, `tend`, `tlen`,
  `alnlen`, `raw`, `bits`, `cigar`, `qseq`, `tseq`, `qaln`, `taln`, `mismatch`,
  `qcov`, `tcov`. An unknown column fails immediately with the supported list.
- **`embscore` column**, carrying the Stage-1 embedding similarity score through
  to the Stage-2 output so both stages' files line up.
- **Calibrated alignment significance.** E-values and bit scores are now computed
  from published Karlin-Altschul constants with a finite-size-corrected search
  space, making them exact and reproducible rather than sampled. Selected with
  `--significance-mode`: `default` (calibrated; requires the default gap
  penalties 11/1) or `sampled` (any gap penalties, magnitudes relative-only).
  `--significance-sample-size` tunes the `sampled` estimator.
- **Bit score** (`bits`) and raw alignment score (`raw`) as reportable columns.
- **Faster Stage-2 alignment.** Work is parallelized over (query, subject)
  *candidate pairs* rather than whole queries, so a single query with many
  candidates now uses the whole node. `--align-workers` sets the pool size and
  defaults to all CPUs, widening any scheduler CPU pinning.
- The significance pass is skipped automatically when neither `evalue` nor
  `bits` is requested, which also lifts the 11/1 gap-penalty restriction for
  those runs.

### Changed

- **BREAKING — `--output-csv` is now `--output-file`, and it is required** on
  `query structure`, `query embedding`, `query sequences`, and `query db`.
- **BREAKING — all result files are now tab-separated with no header row**
  (previously comma-separated with a header).
- **BREAKING — results are always written to a file.** Printing results to
  stdout when no output path was given has been removed.
- **BREAKING — the Stage-2 column set changed.** Output is now the
  `--format-output` columns, defaulting to
  `query,target,embscore,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits`.
  The previous `Query,Rank,Match,EmbScore,SeqIdentity_aln,...` columns are gone.
- **BREAKING — an embedding-only search emits three columns**,
  `query`, `target`, `embscore`. The former `Rank` column is gone; hits are
  ordered best-first, so rank is the row order.
- **BREAKING — `cluster --output` is now `--output-file`**, and its output is
  tab-separated with no header: `chain_id`, `cluster_id`, `cluster_size`.
  The default filename changed from `clusters.csv` to `clusters.tsv`.
- **BREAKING — `similarity-graph --output` is now `--output-file`.** The file is
  still GraphML.
- Relicensed from the EvolutionaryScale Cambrian Non-Commercial License
  Agreement to the **MIT License**.

### Removed

- **BREAKING — cluster JSON output.** `cluster` previously emitted JSON when the
  output filename ended in `.json`; output is always TSV now.
- **BREAKING — `EmbeddingDatabase.print_results()` and
  `EmbeddingDatabase.export_results()`** (Python API). Use
  `foldmatch.search.output.write_embedding_results(results, output_file)`.

### Internal

- New `foldmatch.search.output` module owning the tabular output contract for
  every command — column sets, per-field renderers, and the shared writer.
  `alignment.py` and `clustering.py` are now computation-only.

### Migration

| Before | After |
|---|---|
| `fm-search query structure --db-path db --query-structure q.cif` | add `--output-file hits.tsv` (now required) |
| `--output-csv results.csv` | `--output-file results.tsv` |
| `fm-search cluster --output clusters.csv` | `fm-search cluster --output-file clusters.tsv` |
| `fm-search cluster --output clusters.json` | no JSON equivalent; output is TSV |
| `fm-search similarity-graph --output g.graphml` | `--output-file g.graphml` |
| parsing results with `csv.reader(f)` after skipping a header | split on `\t`; there is no header row |
| `db.export_results(results, path)` | `output.write_embedding_results(results, path)` |

Scripts that read the Stage-2 columns positionally must be updated: pass an
explicit `--format-output` to pin the column order you expect.

## 0.7.3 and earlier

Not tracked in this file; see the git history and release tags.

[0.8.0]: https://github.com/rcsb/foldmatch/releases/tag/v0.8.0
