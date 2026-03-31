const tasks = [
  { name: 'NR-AR', score: 0.21 },
  { name: 'NR-AR-LBD', score: 0.42 },
  { name: 'NR-AhR', score: 0.63 },
  { name: 'NR-Aromatase', score: 0.18 },
  { name: 'NR-ER', score: 0.31 },
  { name: 'NR-ER-LBD', score: 0.19 },
  { name: 'NR-PPAR-gamma', score: 0.24 },
  { name: 'SR-ARE', score: 0.81 },
  { name: 'SR-ATAD5', score: 0.33 },
  { name: 'SR-HSE', score: 0.89 },
  { name: 'SR-MMP', score: 0.71 },
  { name: 'SR-p53', score: 0.78 },
].sort((a, b) => b.score - a.score);

export function MechanismProfilingSection() {
  const activeTasks = tasks.filter(t => t.score > 0.5);
  const highestRisk = tasks[0];

  return (
    <section id="mechanism">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        §2 Mechanism Profiling
      </h2>

      <div className="mb-4 flex items-center gap-6">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          Highest Risk: <span className="font-semibold" style={{ color: 'var(--accent-red)' }}>{highestRisk.name}</span> ({highestRisk.score.toFixed(2)})
        </p>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          Active: <span className="font-semibold" style={{ color: 'var(--text)' }}>{activeTasks.length}/12</span> tasks
        </p>
      </div>

      <div className="rounded-xl p-6 space-y-2" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        {tasks.map((task, idx) => {
          const isActive = task.score > 0.5;
          const isHighest = task.name === highestRisk.name;
          const barColor = task.score >= 0.7 ? 'var(--accent-red)' : task.score >= 0.3 ? 'var(--accent-yellow)' : 'var(--accent-green)';

          return (
            <div key={idx} className="flex items-center gap-3">
              <div className="w-32 text-sm font-medium" style={{ color: 'var(--text)' }}>
                {task.name}
              </div>
              <div className="flex-1 h-5 rounded relative" style={{ backgroundColor: 'var(--border)' }}>
                <div
                  className="h-full rounded"
                  style={{
                    width: `${task.score * 100}%`,
                    backgroundColor: barColor,
                    transition: 'width 600ms cubic-bezier(0.34, 1.56, 0.64, 1)',
                    transitionDelay: `${idx * 50}ms`
                  }}
                />
              </div>
              <div className="w-12 text-sm font-mono text-right" style={{ color: 'var(--text)' }}>
                {task.score.toFixed(2)}
              </div>
              <div className="w-32 text-xs">
                {isHighest && (
                  <span className="font-bold uppercase" style={{ color: 'var(--accent-red)' }}>
                    ★ HIGHEST RISK
                  </span>
                )}
                {isActive && !isHighest && (
                  <span className="font-bold uppercase" style={{ color: 'var(--accent-yellow)' }}>
                    ★ ACTIVE
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4">
        <p className="text-sm mb-2" style={{ color: 'var(--text-muted)' }}>
          Active tasks (score &gt; 0.5): {activeTasks.length}/12
        </p>
        <div className="flex flex-wrap gap-2">
          {activeTasks.map((task, idx) => (
            <span
              key={idx}
              className="px-3 py-1 rounded-full text-xs font-semibold"
              style={{
                backgroundColor: task.name === highestRisk.name ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)',
                color: task.name === highestRisk.name ? 'var(--accent-red)' : 'var(--accent-yellow)',
                border: `1px solid ${task.name === highestRisk.name ? 'var(--accent-red)' : 'var(--accent-yellow)'}`
              }}
            >
              {task.name}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
