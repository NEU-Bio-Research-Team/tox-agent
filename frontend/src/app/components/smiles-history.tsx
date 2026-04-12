import { useEffect, useState } from 'react';
import { Clock, Trash2, TrendingUp } from 'lucide-react';
import { Button } from './ui/button';
import {
  deleteAnalysisFromFirestore,
  loadAnalysesFromFirestore,
  saveAnalysisToFirestore,
  type AnalysisMeta,
  type HistoryEntry,
} from '@/lib/firestore-history';

interface SmilesHistoryProps {
  onSelectSmiles: (smiles: string) => void;
  uid: string | null; //null = guest, dùng localStorage fallback
}

export function SmilesHistory({ onSelectSmiles, uid }: SmilesHistoryProps) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    void loadHistory();
  }, [uid]);

  const loadHistory = async () => {
    try {
      if (uid) {
        // Logged in -> Firestore
        const entries = await loadAnalysesFromFirestore(uid);
        setHistory(entries);
      } else {
        // Guest -> localStorage fallback
        const saved = localStorage.getItem('toxagent_smiles_history');
        setHistory(saved ? JSON.parse(saved) : []);
      }
    } catch (error) {
      console.warn('Failed to load history:', error);
      setHistory([]);
    }
  };


  const deleteEntry = async (id: string) => {
    if (uid) {
      await deleteAnalysisFromFirestore(uid, id);
      setHistory((prev) => prev.filter((entry) => entry.id !== id));
      return;
    }

    const updated = history.filter((entry) => entry.id !== id);
    setHistory(updated);
    localStorage.setItem('toxagent_smiles_history', JSON.stringify(updated));
  };

  const clearAll = async () => {
    if (uid) {
      await Promise.all(history.map((entry) => deleteAnalysisFromFirestore(uid, entry.id)));
      setHistory([]);
      return;
    }

    setHistory([]);
    localStorage.removeItem('toxagent_smiles_history');
  };

  const displayedHistory = showAll ? history : history.slice(0, 5);

  if (history.length === 0) {
    return (
      <div 
        className="rounded-xl p-8 text-center"
        style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
      >
        <Clock className="w-12 h-12 mx-auto mb-3" style={{ color: 'var(--text-faint)' }} />
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          No search history yet. Start analyzing molecules to see them here.
        </p>
      </div>
    );
  }

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'toxic': return 'var(--accent-red)';
      case 'warning': return 'var(--accent-yellow)';
      case 'non-toxic': return 'var(--accent-green)';
      default: return 'var(--text-muted)';
    }
  };

  return (
    <div 
      className="rounded-xl overflow-hidden"
      style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
    >
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4" style={{ color: 'var(--accent-blue)' }} />
          <h3 className="font-semibold" style={{ color: 'var(--text)' }}>Recent Searches</h3>
          <span 
            className="text-xs px-2 py-0.5 rounded-full"
            style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}
          >
            {history.length}
          </span>
        </div>
        {history.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              void clearAll();
            }}
            className="text-xs"
            style={{ color: 'var(--accent-red)' }}
          >
            Clear all
          </Button>
        )}
      </div>

      {/* History List */}
      <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
        {displayedHistory.map((entry) => (
          <div
            key={entry.id}
            className="p-4 hover:bg-[var(--surface-alt)] transition-colors group"
          >
            <div className="flex items-start justify-between gap-3">
              <button
                onClick={() => onSelectSmiles(entry.smiles)}
                className="flex-1 text-left"
              >
                <div className="flex items-center gap-2 mb-1">
                  <code 
                    className="text-sm font-mono"
                    style={{ color: 'var(--text)' }}
                  >
                    {entry.smiles}
                  </code>
                  <span
                    className="text-xs px-2 py-0.5 rounded-full font-medium capitalize"
                    style={{ 
                      backgroundColor: `${getVerdictColor(entry.verdict)}15`,
                      color: getVerdictColor(entry.verdict)
                    }}
                  >
                    {entry.verdict}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                  <span>{new Date(entry.timestamp).toLocaleDateString()}</span>
                  <span>•</span>
                  <div className="flex items-center gap-1">
                    <TrendingUp className="w-3 h-3" />
                    <span>{(entry.score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </button>
              
              <button
                onClick={() => {
                  void deleteEntry(entry.id);
                }}
                className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-[var(--border)] rounded"
                title="Delete"
              >
                <Trash2 className="w-3.5 h-3.5" style={{ color: 'var(--accent-red)' }} />
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Show More/Less */}
      {history.length > 5 && (
        <div className="p-3 border-t" style={{ borderColor: 'var(--border)' }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAll(!showAll)}
            className="w-full text-sm"
            style={{ color: 'var(--accent-blue)' }}
          >
            {showAll ? 'Show less' : `Show ${history.length - 5} more`}
          </Button>
        </div>
      )}
    </div>
  );
}

// Helper function to save to history
export async function addToHistory(
  smiles: string,
  verdict: 'toxic' | 'warning' | 'non-toxic',
  score: number,
  uid?: string,
  meta?: AnalysisMeta,
) {
  if (uid) {
    // Save to Firestore
    await saveAnalysisToFirestore(uid, smiles, verdict, score, meta);
  } else {
    const saved = localStorage.getItem('toxagent_smiles_history');
    const history: HistoryEntry[] = saved ? JSON.parse(saved) : [];

    // Check if this SMILES already exists
    const existingIndex = history.findIndex((entry) => entry.smiles === smiles);

    const newEntry: HistoryEntry = {
      id: crypto.randomUUID(),
      smiles,
      timestamp: Date.now(),
      verdict,
      score,
    };

    // Remove existing entry if found
    if (existingIndex !== -1) {
      history.splice(existingIndex, 1);
    }

    // Add to beginning of array (most recent first)
    history.unshift(newEntry);

    // Keep only last 50 entries
    const trimmedHistory = history.slice(0, 50);

    localStorage.setItem('toxagent_smiles_history', JSON.stringify(trimmedHistory));
}
}
