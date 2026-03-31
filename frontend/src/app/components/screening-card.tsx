import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Shield, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import { Progress } from './ui/progress';

interface ScreeningCardProps {
  isAnalyzing: boolean;
  expanded?: boolean;
}

export function ScreeningCard({ isAnalyzing, expanded = false }: ScreeningCardProps) {
  const result = {
    p_toxic: 0.23,
    label: 'NON_TOXIC',
    confidence: 0.89,
  };

  return (
    <Card className="bg-white/80 backdrop-blur-md border-slate-200 hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-green-600 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <CardTitle>Screening Agent</CardTitle>
              <CardDescription>Gemini 2.0 Flash - Toxicity Prediction</CardDescription>
            </div>
          </div>
          <Badge 
            variant={result.label === 'TOXIC' ? 'destructive' : 'default'}
            className={result.label === 'TOXIC' ? '' : 'bg-emerald-500 hover:bg-emerald-600'}
          >
            {result.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Main Result */}
        <div className="bg-gradient-to-br from-emerald-50 to-green-50 border border-emerald-200 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-emerald-600" />
              <span className="font-semibold text-slate-900">Prediction Result</span>
            </div>
            <span className="text-3xl font-bold text-emerald-600">
              {(result.p_toxic * 100).toFixed(1)}%
            </span>
          </div>
          <p className="text-sm text-slate-600">Toxicity Probability</p>
        </div>

        {/* Confidence Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Confidence Level</span>
            <span className="font-semibold text-slate-900">{(result.confidence * 100).toFixed(0)}%</span>
          </div>
          <Progress value={result.confidence * 100} className="h-2" />
        </div>

        {expanded && (
          <>
            {/* Tool Call Details */}
            <div className="bg-slate-50 rounded-lg p-3 space-y-2">
              <p className="text-xs font-semibold text-slate-700">Tool Call</p>
              <div className="font-mono text-xs text-slate-600 space-y-1">
                <p>predict_toxicity(smiles)</p>
                <p className="text-emerald-600">→ FastAPI /predict endpoint</p>
              </div>
            </div>

            {/* Model Info */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-xs text-slate-500 mb-1">Model</p>
                <p className="text-sm font-semibold text-slate-900">GNN Classifier</p>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <p className="text-xs text-slate-500 mb-1">Validation</p>
                <p className="text-sm font-semibold text-slate-900">RDKit Passed</p>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
