import { useState } from 'react';

const sections = [
  { id: 'clinical', label: 'Clinical Toxicity' },
  { id: 'mechanism', label: 'Mechanism' },
  { id: 'structural', label: 'Structural' },
  { id: 'literature', label: 'Literature' },
  { id: 'recommendations', label: 'AI Recommendations' },
];

export function ReportSidebar() {
  const [activeSection, setActiveSection] = useState('clinical');

  return (
    <aside className="sticky top-16 h-[calc(100vh-4rem)] p-6 border-r overflow-y-auto" style={{ borderColor: 'var(--border)' }}>
      <div className="mb-6">
        <h3 className="text-sm font-semibold uppercase mb-4" style={{ color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
          Nội dung báo cáo
        </h3>
        <nav className="space-y-2">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors"
              style={{
                backgroundColor: activeSection === section.id ? 'var(--accent-blue-muted)' : 'transparent',
                color: activeSection === section.id ? 'var(--accent-blue)' : 'var(--text-muted)',
              }}
            >
              {activeSection === section.id ? '●' : '○'} {section.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="border-t pt-6" style={{ borderColor: 'var(--border)' }}>
        <h3 className="text-xs font-semibold uppercase mb-3" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
          Thông tin nhanh
        </h3>
        <div className="space-y-3">
          <div>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>p_toxic</p>
            <div className="w-full h-1.5 rounded-full mb-1" style={{ backgroundColor: 'var(--border)' }}>
              <div className="h-full rounded-full" style={{ width: '23%', backgroundColor: 'var(--accent-green)' }} />
            </div>
            <p className="font-mono text-xs font-semibold" style={{ color: 'var(--text)' }}>0.23</p>
          </div>
          <div>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Label</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--accent-green)' }}>NON-TOXIC</p>
          </div>
          <div>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>MW</p>
            <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>180.16</p>
          </div>
          <div>
            <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>Formula</p>
            <p className="font-mono text-sm font-semibold" style={{ color: 'var(--text)' }}>C9H8O4</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
