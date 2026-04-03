import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Atom, Zap, Activity } from 'lucide-react';

interface ExplainerCardProps {
  isAnalyzing: boolean;
  expanded?: boolean;
}

export function ExplainerCard({ isAnalyzing, expanded = false }: ExplainerCardProps) {
  const result = {
    top_atoms: [
      { index: 3, importance: 0.92, element: 'O' },
      { index: 7, importance: 0.87, element: 'C' },
      { index: 2, importance: 0.76, element: 'C' },
    ],
    top_bonds: [
      { atoms: [2, 3], importance: 0.89, type: 'C=O' },
      { atoms: [7, 8], importance: 0.81, type: 'C-O' },
    ],
    interpretation: 'Carbonyl group (C=O) shows highest attribution for toxicity prediction',
  };

  return (
    <Card className="bg-white/80 backdrop-blur-md border-slate-200 hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg flex items-center justify-center">
              <Atom className="w-5 h-5 text-white" />
            </div>
            <div>
              <CardTitle>Explainer Agent</CardTitle>
              <CardDescription>Gemini 2.0 Flash - GNN Explainer</CardDescription>
            </div>
          </div>
          <Badge variant="outline" className="border-purple-300 text-purple-700">
            Attribution
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Heatmap Visualization */}
        <div className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-purple-600" />
            <span className="text-sm font-semibold text-slate-900">Molecular Heatmap</span>
          </div>
          <div className="bg-white rounded-lg h-32 flex items-center justify-center border border-purple-100">
            <div className="text-center">
              <Zap className="w-8 h-8 text-purple-400 mx-auto mb-2" />
              <p className="text-xs text-slate-500">GNN Attribution Heatmap</p>
            </div>
          </div>
        </div>

        {/* Top Atoms */}
        <div className="space-y-2">
          <p className="text-sm font-semibold text-slate-900">Top Contributing Atoms</p>
          <div className="space-y-2">
            {result.top_atoms.map((atom, idx) => (
              <div key={idx} className="flex items-center gap-3 bg-slate-50 rounded-lg p-2">
                <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                  <span className="text-xs font-bold text-purple-700">{atom.element}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-slate-600">Atom #{atom.index}</span>
                    <span className="text-xs font-semibold text-slate-900">
                      {(atom.importance * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                      style={{ width: `${atom.importance * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {expanded && (
          <>
            {/* Top Bonds */}
            <div className="space-y-2">
              <p className="text-sm font-semibold text-slate-900">Top Contributing Bonds</p>
              <div className="space-y-2">
                {result.top_bonds.map((bond, idx) => (
                  <div key={idx} className="flex items-center justify-between bg-slate-50 rounded-lg p-3">
                    <div>
                      <p className="text-sm font-medium text-slate-900">{bond.type}</p>
                      <p className="text-xs text-slate-500">Atoms {bond.atoms.join('-')}</p>
                    </div>
                    <Badge variant="secondary">{(bond.importance * 100).toFixed(0)}%</Badge>
                  </div>
                ))}
              </div>
            </div>

            {/* Chemical Interpretation */}
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-3">
              <p className="text-xs font-semibold text-purple-900 mb-1">Chemical Interpretation</p>
              <p className="text-sm text-slate-700">{result.interpretation}</p>
            </div>

            {/* Note */}
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
              <p className="text-xs text-amber-800">
                <span className="font-semibold">Note:</span> GNNExplainer provides attribution analysis. 
                Prediction instability is expected due to optimization noise.
              </p>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
