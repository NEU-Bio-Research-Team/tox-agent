import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Network, CheckCircle2, Clock, AlertCircle } from 'lucide-react';
import { Progress } from './ui/progress';

interface OrchestratorCardProps {
  smilesInput: string;
  isAnalyzing: boolean;
}

export function OrchestratorCard({ smilesInput, isAnalyzing }: OrchestratorCardProps) {
  const agents = [
    { name: 'Screening Agent', status: 'completed', progress: 100 },
    { name: 'Explainer Agent', status: 'completed', progress: 100 },
    { name: 'Researcher Agent', status: 'completed', progress: 100 },
    { name: 'Report Writer', status: 'completed', progress: 100 },
  ];

  return (
    <Card className="bg-gradient-to-br from-blue-600 to-indigo-600 text-white border-0">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-white/20 backdrop-blur-md rounded-lg flex items-center justify-center">
              <Network className="w-6 h-6 text-white" />
            </div>
            <div>
              <CardTitle className="text-white">Orchestrator Agent</CardTitle>
              <CardDescription className="text-blue-100">
                Gemini 2.0 Flash - Root Coordinator
              </CardDescription>
            </div>
          </div>
          <Badge variant="secondary" className="bg-white/20 text-white border-white/30">
            Active
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Molecule Info */}
        <div className="bg-white/10 backdrop-blur-md rounded-lg p-4">
          <p className="text-xs text-blue-100 mb-1">Analyzing Molecule</p>
          <p className="font-mono text-sm break-all">{smilesInput}</p>
        </div>

        {/* Agent Status Grid */}
        <div className="grid grid-cols-2 gap-3">
          {agents.map((agent, idx) => (
            <div key={idx} className="bg-white/10 backdrop-blur-md rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium">{agent.name}</p>
                {agent.status === 'completed' ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-300" />
                ) : agent.status === 'running' ? (
                  <Clock className="w-4 h-4 text-amber-300 animate-pulse" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-slate-300" />
                )}
              </div>
              <Progress value={agent.progress} className="h-1.5 bg-white/20" />
            </div>
          ))}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-white/10 backdrop-blur-md rounded-lg p-3 text-center">
            <p className="text-2xl font-bold">4/4</p>
            <p className="text-xs text-blue-100">Agents Complete</p>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-lg p-3 text-center">
            <p className="text-2xl font-bold">2.3s</p>
            <p className="text-xs text-blue-100">Total Time</p>
          </div>
          <div className="bg-white/10 backdrop-blur-md rounded-lg p-3 text-center">
            <p className="text-2xl font-bold">High</p>
            <p className="text-xs text-blue-100">Confidence</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
