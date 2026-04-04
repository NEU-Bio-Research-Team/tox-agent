import { Github } from 'lucide-react';

export function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer 
      className="border-t py-12 mt-20 transition-colors duration-300"
      style={{ 
        borderColor: 'var(--border)',
        backgroundColor: 'var(--bg-secondary)' 
      }}
    >
      <div className="max-w-[1400px] mx-auto px-6">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-12 mb-10">
          
          {/* Column 1: Brand & Mission */}
          <div className="md:col-span-2 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-blue-500/20">
                T
              </div>
              <span className="font-bold text-xl tracking-tight" style={{ color: 'var(--text)' }}>
                ToxAgent
              </span>
            </div>
            <p className="text-sm leading-relaxed max-w-sm" style={{ color: 'var(--text-muted)' }}>
              A cutting-edge Multi-Agent Molecular Toxicity Analysis System. 
              Leveraging Gemini 2.0 Flash to accelerate bio-research safety and discovery.
            </p>
          </div>

          {/* Column 2: Resources */}
          <div className="flex flex-col gap-4">
            <h4 className="text-sm font-semibold uppercase tracking-wider" style={{ color: 'var(--text)' }}>
              Resources
            </h4>
            <nav className="flex flex-col gap-2 text-sm">
              <a href="#" className="hover:text-blue-500 transition-colors" style={{ color: 'var(--text-muted)' }}>Documentation</a>
              <a href="#" className="hover:text-blue-500 transition-colors" style={{ color: 'var(--text-muted)' }}>Research Papers</a>
              <a href="#" className="hover:text-blue-500 transition-colors" style={{ color: 'var(--text-muted)' }}>Methodology</a>
            </nav>
          </div>

          {/* Column 3: Legal & Contact */}
          <div className="flex flex-col gap-4">
            <h4 className="text-sm font-semibold uppercase tracking-wider" style={{ color: 'var(--text)' }}>
              Legal
            </h4>
            <nav className="flex flex-col gap-2 text-sm">
              <a href="#" className="hover:text-blue-500 transition-colors" style={{ color: 'var(--text-muted)' }}>Privacy Policy</a>
              <a href="#" className="hover:text-blue-500 transition-colors" style={{ color: 'var(--text-muted)' }}>Terms of Service</a>
              <a href="mailto:contact@toxagent.ai" className="hover:text-blue-500 transition-colors" style={{ color: 'var(--text-muted)' }}>Contact Us</a>
            </nav>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t flex flex-col md:flex-row justify-between items-center gap-6" style={{ borderColor: 'var(--border-faint)' }}>
          
          {/* Copyright & Team */}
          <div className="flex flex-col items-center md:items-start gap-1">
            <div className="text-xs" style={{ color: 'var(--text-faint)' }}>
              © {currentYear} <span className="font-medium text-blue-500/80">ToxAgent Project</span>. All rights reserved.
            </div>
            <div className="text-[10px] font-medium uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
              NEU Bio Research Team
            </div>
          </div>

          {/* Socials & Build Info */}
          <div className="flex items-center gap-6">
            <a 
              href="https://github.com/NEU-Bio-Research-Team/tox-agent.git" 
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-800 transition-all group"
              style={{ color: 'var(--text-muted)' }}
            >
              <Github className="w-5 h-5 group-hover:scale-110 transition-transform" />
            </a>
            
            <div className="flex items-center gap-3 border-l pl-6" style={{ borderColor: 'var(--border)' }}>
              <span className="px-2 py-0.5 rounded bg-blue-500/10 text-blue-500 text-[10px] font-mono border border-blue-500/20">
                v1.0.0-stable
              </span>
              <span className="text-[11px] font-medium" style={{ color: 'var(--text-faint)' }}>
                Powered by <span className="text-blue-500">NEU Bio Research Team</span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}