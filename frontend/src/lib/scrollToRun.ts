/** "Về lượt chat tạo kết quả" (plan section 8.2.1): focuses the originating
 * run's card in the transcript. `RunBlock`/`AnalysisSystemCard` both render
 * `id="run-anchor-<run_id>"`; a run not yet loaded into the transcript
 * (message history not paginated in yet) simply has no anchor to jump to —
 * a documented backlog gap (section 10.2), not a crash. */
export function scrollToRun(runId: string): void {
  document.getElementById(`run-anchor-${runId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
}
