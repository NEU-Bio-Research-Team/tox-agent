import { createContext, useContext, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type { AgentAnalyzeResponse } from './api';

interface ReportContextValue {
  report: AgentAnalyzeResponse | null;
  setReport: (nextReport: AgentAnalyzeResponse | null) => void;
  isLoading: boolean;
  setIsLoading: (next: boolean) => void;
  error: string | null;
  setError: (next: string | null) => void;
}

const ReportContext = createContext<ReportContextValue | undefined>(undefined);

export function ReportProvider({ children }: { children: ReactNode }) {
  const [report, setReport] = useState<AgentAnalyzeResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const value = useMemo(
    () => ({
      report,
      setReport,
      isLoading,
      setIsLoading,
      error,
      setError,
    }),
    [error, isLoading, report],
  );

  return <ReportContext.Provider value={value}>{children}</ReportContext.Provider>;
}

export function useReport() {
  const context = useContext(ReportContext);
  if (!context) {
    throw new Error('useReport must be used within a ReportProvider');
  }
  return context;
}
