import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { FlaskConical, Play, Loader2 } from 'lucide-react';

interface SmilesMoleculeInputProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
}

export function SmilesMoleculeInput({ value, onChange, onAnalyze, isAnalyzing }: SmilesMoleculeInputProps) {
  return (
    <Card className="bg-white/80 backdrop-blur-md border-slate-200">
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-lg flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-white" />
          </div>
          <div>
            <CardTitle>SMILES Input</CardTitle>
            <CardDescription>Enter a SMILES string to analyze molecular toxicity</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="CC(=O)Oc1ccccc1C(=O)O"
          className="font-mono text-sm min-h-[100px]"
          disabled={isAnalyzing}
        />
        <div className="flex items-center justify-between">
          <p className="text-xs text-slate-500">
            Example: Aspirin - CC(=O)Oc1ccccc1C(=O)O
          </p>
          <Button 
            onClick={onAnalyze} 
            disabled={isAnalyzing || !value.trim()}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Analyze Toxicity
              </>
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
