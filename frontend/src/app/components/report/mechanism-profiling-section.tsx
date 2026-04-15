import type { MechanismSection } from '../../../lib/api';

interface MechanismProfilingSectionProps {
  data: MechanismSection;
  language: 'vi' | 'en';
}

export function MechanismProfilingSection({ data, language }: MechanismProfilingSectionProps) {
  const taskScores = data?.task_scores ?? {};
  const tasks = Object.entries(taskScores)
    .map(([name, score]) => ({ name, score: Number(score ?? 0) }))
    .sort((a, b) => b.score - a.score);

  const activeTasks = tasks.filter((task) => task.score > 0.5);
  const highestRiskName = data?.highest_risk || tasks[0]?.name || 'N/A';
  const highestRisk = tasks.find((task) => task.name === highestRiskName) || tasks[0];
  const highestRiskScore = highestRisk?.score ?? 0;

  return (
    <section id="mechanism" className="scroll-mt-24 lg:scroll-mt-20">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        {language === 'vi' ? '§2 Hồ sơ cơ chế độc tính' : '§2 Mechanism Profiling'}
      </h2>

      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:gap-6">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {language === 'vi' ? 'Rủi ro cao nhất' : 'Highest Risk'}:{' '}
          <span className="font-semibold" style={{ color: 'var(--accent-red)' }}>
            {highestRiskName}
          </span>{' '}
          ({highestRiskScore.toFixed(2)})
        </p>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          {language === 'vi' ? 'Đang active' : 'Active'}:{' '}
          <span className="font-semibold" style={{ color: 'var(--text)' }}>
            {activeTasks.length}/{tasks.length || 0}
          </span>{' '}
          {language === 'vi' ? 'task' : 'tasks'}
        </p>
      </div>

      <div className="rounded-xl p-6 space-y-2" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        {tasks.length === 0 && (
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Không có dữ liệu task_scores.' : 'No task_scores data available.'}
          </p>
        )}

        {tasks.map((task, idx) => {
          const isActive = task.score > 0.5;
          const isHighest = task.name === highestRiskName;
          const barColor =
            task.score >= 0.7
              ? 'var(--accent-red)'
              : task.score >= 0.3
                ? 'var(--accent-yellow)'
                : 'var(--accent-green)';

          return (
            <div key={task.name || idx} className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
              <div className="text-sm font-medium sm:w-32" style={{ color: 'var(--text)' }}>
                {task.name}
              </div>
              <div className="flex-1 h-5 rounded relative" style={{ backgroundColor: 'var(--border)' }}>
                <div
                  className="h-full rounded"
                  style={{
                    width: `${Math.min(Math.max(task.score, 0), 1) * 100}%`,
                    backgroundColor: barColor,
                    transition: 'width 600ms cubic-bezier(0.34, 1.56, 0.64, 1)',
                    transitionDelay: `${idx * 40}ms`,
                  }}
                />
              </div>
              <div className="text-sm font-mono sm:w-12 sm:text-right" style={{ color: 'var(--text)' }}>
                {task.score.toFixed(2)}
              </div>
              <div className="text-xs sm:w-32">
                {isHighest && (
                  <span className="font-bold uppercase" style={{ color: 'var(--accent-red)' }}>
                    {language === 'vi' ? '★ RỦI RO CAO NHẤT' : '★ HIGHEST RISK'}
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
          {language === 'vi'
            ? `Task active (score > 0.5): ${activeTasks.length}/${tasks.length || 0}`
            : `Active tasks (score > 0.5): ${activeTasks.length}/${tasks.length || 0}`}
        </p>
        <div className="flex flex-wrap gap-2">
          {activeTasks.map((task) => (
            <span
              key={task.name}
              className="px-3 py-1 rounded-full text-xs font-semibold"
              style={{
                backgroundColor:
                  task.name === highestRiskName ? 'rgba(239,68,68,0.15)' : 'rgba(245,158,11,0.15)',
                color: task.name === highestRiskName ? 'var(--accent-red)' : 'var(--accent-yellow)',
                border: `1px solid ${task.name === highestRiskName ? 'var(--accent-red)' : 'var(--accent-yellow)'}`,
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
