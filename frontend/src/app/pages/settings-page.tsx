import { useState } from 'react';
import { Navbar } from '../components/navbar';
import { Button } from '../components/ui/button';
import { Bell, Database, Shield, Palette, Zap, Globe } from 'lucide-react';

export function SettingsPage() {
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [autoSave, setAutoSave] = useState(true);
  const [defaultThreshold, setDefaultThreshold] = useState(0.5);
  const [language, setLanguage] = useState('vi');

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: 'var(--bg)',
      fontFamily: 'Inter, sans-serif'
    }}>
      <Navbar />
      
      <main className="max-w-4xl mx-auto px-6 py-12">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2" style={{ color: 'var(--text)' }}>
            Settings
          </h1>
          <p className="text-base" style={{ color: 'var(--text-muted)' }}>
            Customize your ToxAgent experience
          </p>
        </div>

        <div className="space-y-6">
          {/* General Settings */}
          <section className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="flex items-center gap-3 mb-6">
              <Zap className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
              <h2 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                General
              </h2>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between py-3 border-b" style={{ borderColor: 'var(--border)' }}>
                <div>
                  <h3 className="font-medium mb-1" style={{ color: 'var(--text)' }}>Language</h3>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Choose your preferred language</p>
                </div>
                <select 
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="px-4 py-2 rounded-lg border"
                  style={{ 
                    backgroundColor: 'var(--surface-alt)', 
                    borderColor: 'var(--border)',
                    color: 'var(--text)'
                  }}
                >
                  <option value="vi">Tiếng Việt</option>
                  <option value="en">English</option>
                </select>
              </div>

              <div className="flex items-center justify-between py-3">
                <div>
                  <h3 className="font-medium mb-1" style={{ color: 'var(--text)' }}>Auto-save results</h3>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Automatically save analysis results</p>
                </div>
                <button
                  onClick={() => setAutoSave(!autoSave)}
                  className="relative w-12 h-6 rounded-full transition-colors"
                  style={{ backgroundColor: autoSave ? 'var(--accent-blue)' : 'var(--border)' }}
                >
                  <span 
                    className="absolute top-1 w-4 h-4 rounded-full bg-white transition-transform"
                    style={{ transform: autoSave ? 'translateX(26px)' : 'translateX(4px)' }}
                  />
                </button>
              </div>
            </div>
          </section>

          {/* Analysis Settings */}
          <section className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="flex items-center gap-3 mb-6">
              <Database className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
              <h2 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                Analysis
              </h2>
            </div>

            <div className="space-y-4">
              <div className="py-3">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-medium" style={{ color: 'var(--text)' }}>Default Toxicity Threshold</h3>
                  <span className="font-mono font-semibold" style={{ color: 'var(--accent-blue)' }}>
                    {defaultThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={defaultThreshold}
                  onChange={(e) => setDefaultThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 rounded-full appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, var(--accent-green) 0%, var(--accent-yellow) 50%, var(--accent-red) 100%)`
                  }}
                />
                <div className="flex justify-between text-xs mt-2" style={{ color: 'var(--text-faint)' }}>
                  <span>Non-toxic</span>
                  <span>Warning</span>
                  <span>Toxic</span>
                </div>
              </div>

              <div className="flex items-center justify-between py-3 border-t" style={{ borderColor: 'var(--border)' }}>
                <div>
                  <h3 className="font-medium mb-1" style={{ color: 'var(--text)' }}>Default Analysis Mode</h3>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Choose between Full or Quick analysis</p>
                </div>
                <select 
                  className="px-4 py-2 rounded-lg border"
                  style={{ 
                    backgroundColor: 'var(--surface-alt)', 
                    borderColor: 'var(--border)',
                    color: 'var(--text)'
                  }}
                >
                  <option value="full">Full Analysis</option>
                  <option value="quick">Quick Analysis</option>
                </select>
              </div>
            </div>
          </section>

          {/* Notifications */}
          <section className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="flex items-center gap-3 mb-6">
              <Bell className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
              <h2 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                Notifications
              </h2>
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between py-3">
                <div>
                  <h3 className="font-medium mb-1" style={{ color: 'var(--text)' }}>Enable notifications</h3>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Receive alerts when analysis completes</p>
                </div>
                <button
                  onClick={() => setNotificationsEnabled(!notificationsEnabled)}
                  className="relative w-12 h-6 rounded-full transition-colors"
                  style={{ backgroundColor: notificationsEnabled ? 'var(--accent-blue)' : 'var(--border)' }}
                >
                  <span 
                    className="absolute top-1 w-4 h-4 rounded-full bg-white transition-transform"
                    style={{ transform: notificationsEnabled ? 'translateX(26px)' : 'translateX(4px)' }}
                  />
                </button>
              </div>
            </div>
          </section>

          {/* Privacy & Security */}
          <section className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="flex items-center gap-3 mb-6">
              <Shield className="w-5 h-5" style={{ color: 'var(--accent-blue)' }} />
              <h2 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                Privacy & Security
              </h2>
            </div>

            <div className="space-y-4">
              <div className="py-3">
                <h3 className="font-medium mb-1" style={{ color: 'var(--text)' }}>Data Storage</h3>
                <p className="text-sm mb-3" style={{ color: 'var(--text-muted)' }}>
                  All analysis data is stored locally in your browser. No data is sent to external servers.
                </p>
                <Button 
                  variant="outline" 
                  className="text-sm"
                  style={{ borderColor: 'var(--accent-red)', color: 'var(--accent-red)' }}
                >
                  Clear All Data
                </Button>
              </div>
            </div>
          </section>

          {/* Save Button */}
          <div className="flex justify-end gap-3 pt-4">
            <Button
              variant="outline"
              className="px-6"
            >
              Reset to Defaults
            </Button>
            <Button
              className="px-6"
              style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
            >
              Save Changes
            </Button>
          </div>
        </div>
      </main>
    </div>
  );
}
