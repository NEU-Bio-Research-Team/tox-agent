export function Footer() {
  return (
    <footer className="border-t py-8" style={{ borderColor: 'var(--border)' }}>
      <div
        className="mx-auto flex max-w-6xl flex-col items-center gap-2 px-6 text-xs md:flex-row md:justify-between"
        style={{ color: 'var(--text-faint)' }}
      >
        <span>© {new Date().getFullYear()} ToxAgent — evidence-and-decision-support control plane</span>
        <span>hERG, Tox21 và ClinTox là ba phép đo độc lập. Không có aggregate toxicity score.</span>
      </div>
    </footer>
  );
}
