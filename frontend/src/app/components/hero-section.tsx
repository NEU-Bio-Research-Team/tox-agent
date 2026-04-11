import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Loader2, Zap, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';

interface HeroSectionProps {
  value: string;
  onChange: (value: string) => void;
  onAnalyze: (opts: { binaryModel: string; toxTypeModel: string}) => void; //allow changes will be applied
  isAnalyzing: boolean;
}

const exampleMolecules = [
  { name: 'Caffeine', icon: '☕', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
  { name: 'Aspirin', icon: '💊', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
  { name: 'Ethanol', icon: '⚗', smiles: 'CCO' },
  { name: 'Benzene', icon: '⚠', smiles: 'c1ccccc1' },
];

export function HeroSection({ value, onChange, onAnalyze, isAnalyzing }: HeroSectionProps) {
  const [validationState, setValidationState] = useState<'idle' | 'checking' | 'valid' | 'invalid'>('idle');
  const [validationMessage, setValidationMessage] = useState('');
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [threshold, setThreshold] = useState(0.5);

  // Add options for users to choose which model will use for 2 predict tasks
  const [binaryModel, setBinaryModel] = useState<string>('pretrained_2head_herg_chemberta_model');
  const [toxTypeModel, setToxTypeModel] = useState<string>('tox21_gatv2_model');

  useEffect(() => {
    if (!value.trim()) {
      setValidationState('idle');
      setValidationMessage('');
      return;
    }

    setValidationState('valid');
    setValidationMessage('Ready to call API. Detailed validation will be handled by the backend.');
  }, [value]);

  const getButtonState = () => {
    if (isAnalyzing) return { text: 'Analyzing...', icon: Loader2, disabled: true, className: 'animate-spin' };
    if (!value.trim()) return { text: 'Analyze', icon: Zap, disabled: true, className: '' };
    return { text: 'Analyze', icon: Zap, disabled: false, className: '' };
  };

  const buttonState = getButtonState();
  const ButtonIcon = buttonState.icon;

  return (
    <section className="pt-20 pb-12">
      <div className="text-center mb-10">
        <h1 
          className="text-5xl font-bold mb-3 bg-gradient-to-r from-[var(--accent-blue)] to-[var(--accent-green)] bg-clip-text"
          style={{ 
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text'
          }}
        >
          Drug Toxicity Analysis
        </h1>
        <p className="text-base mb-1" style={{ color: 'var(--text-muted)' }}>
          Drug Toxicity Analysis using Multi-Agent AI System
        </p>
        <p className="text-sm" style={{ color: 'var(--text-faint)' }}>
          Enter a SMILES string or molecule name to start the analysis
        </p>
      </div>

      {/* Input Container */}
      <div className="rounded-2xl p-8 shadow-lg" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
        {/* SMILES Input */}
        <div className="flex flex-col md:flex-row gap-3 mb-3">
          <div className="flex-1 relative">
            <div className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-faint)' }}>
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v6m0 6v6m0-6h6m-6 0H6" />
              </svg>
            </div>
            <Textarea
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder="CC(=O)Oc1ccccc1C(=O)O  hoặc  aspirin"
              className="pl-12 pr-10 py-4 min-h-[56px] font-mono text-sm resize-none"
              style={{
                backgroundColor: 'var(--surface-alt)',
                border: validationState === 'valid' ? '1.5px solid var(--accent-green)' : validationState === 'invalid' ? '1.5px solid var(--accent-red)' : '1.5px solid var(--border)',
                color: 'var(--text)',
                letterSpacing: '0.02em'
              }}
              disabled={isAnalyzing}
              spellCheck={false}
              autoCorrect="off"
              autoComplete="off"
            />
            {value && !isAnalyzing && (
              <button
                onClick={() => onChange('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-[var(--surface)] transition-colors"
                style={{ color: 'var(--text-faint)' }}
              >
                ×
              </button>
            )}
          </div>
          
          <Button
            onClick={() => onAnalyze({ binaryModel, toxTypeModel})}
            disabled={buttonState.disabled}
            className="md:w-auto w-full h-[56px] px-7 text-base font-semibold rounded-lg"
            style={{
              backgroundColor: buttonState.disabled ? 'var(--border)' : 'var(--accent-blue)',
              color: '#ffffff',
              opacity: buttonState.disabled ? 0.5 : 1
            }}
          >
            <ButtonIcon className={`w-4 h-4 mr-2 ${buttonState.className}`} />
            {buttonState.text}
          </Button>
        </div>

        {/* Validation Feedback */}
        {validationState !== 'idle' && (
          <div 
            className="px-4 py-2 rounded-lg text-sm mb-4"
            style={{
              backgroundColor: validationState === 'valid' ? 'rgba(34,197,94,0.08)' : validationState === 'invalid' ? 'rgba(239,68,68,0.08)' : 'transparent',
              borderLeft: validationState === 'valid' ? '2px solid var(--accent-green)' : validationState === 'invalid' ? '2px solid var(--accent-red)' : 'none',
              color: validationState === 'valid' ? 'var(--accent-green)' : validationState === 'invalid' ? 'var(--accent-red)' : 'var(--text-faint)'
            }}
          >
            <div className="flex items-center gap-2">
              {validationState === 'checking' && <RefreshCw className="w-4 h-4 animate-spin" />}
              {validationState === 'valid' && <CheckCircle className="w-4 h-4" />}
              {validationState === 'invalid' && <XCircle className="w-4 h-4" />}
              <span>{validationState === 'checking' ? 'Validating SMILES...' : validationMessage}</span>
            </div>
          </div>
        )}

        {/* Example Molecules */}
        <div className="mb-4">
          <p className="text-xs uppercase mb-2" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
            Quick Examples:
          </p>
          <div className="flex flex-wrap gap-2">
            {exampleMolecules.map((mol, idx) => (
              <button
                key={idx}
                onClick={() => onChange(mol.smiles)}
                className="px-4 py-2 rounded-full text-xs font-medium transition-all"
                style={{
                  backgroundColor: 'var(--surface-alt)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-muted)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--surface)';
                  e.currentTarget.style.borderColor = 'var(--accent-blue)';
                  e.currentTarget.style.color = 'var(--text)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--surface-alt)';
                  e.currentTarget.style.borderColor = 'var(--border)';
                  e.currentTarget.style.color = 'var(--text-muted)';
                }}
              >
                {mol.icon} {mol.name}
              </button>
            ))}
          </div>
        </div>

        {/* Advanced Options */}
        <Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen}>
          <CollapsibleTrigger className="text-sm flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
            <span>{isAdvancedOpen ? '▼' : '▶'}</span>
            Advanced options
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-4 space-y-4">
            <div>
              <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>
                Toxicity Threshold
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="flex-1"
                  style={{
                    background: `linear-gradient(to right, var(--accent-green) 0%, var(--accent-yellow) 50%, var(--accent-red) 100%)`
                  }}
                />
                <span className="font-mono text-sm w-12" style={{ color: 'var(--text)' }}>
                  {threshold.toFixed(2)}
                </span>
              </div>
            </div>
            <div>
              <label className="text-sm mb-2 block" style={{ color: 'var(--text-muted)' }}>
                Analysis Mode:
              </label>
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm" style={{ color: 'var(--text)' }}>
                  <input type="radio" name="mode" defaultChecked />
                  Full (ScreeningAgent + ResearcherAgent + Writer)
                </label>
                <label className="flex items-center gap-2 text-sm" style={{ color: 'var(--text)' }}>
                  <input type="radio" name="mode" />
                  Quick (ScreeningAgent only)
                </label>
              </div>
            </div>

            {/* GNN Binary Toxicity Model */}
            <div>
              <label className="text-sm mb-1 block" style={{ color: 'var(--text-muted)' }}>
                Binary Toxicity Model (GNN):
              </label>
              <p className="text-xs mb-2" style={{ color: 'var(--text-faint' }}>
                Chọn GNN backbone để predict xác suất độc tính nhị phân
              </p>
              <select
                value={binaryModel}
                onChange={(e) => setBinaryModel(e.target.value)}
                disabled={isAnalyzing}
                className="w-full px-3 py-2 rounded-lg text-sm"
                style={{
                  backgroundColor: "var(--surface-alt)",
                  border: '1px solid var(--border)',
                  color: 'var(--text)',
                }}
              >
                <option value="pretrained_2head_herg_chemberta_model">ChemBERTa Dual-Head · Full · Recommended</option>
                <option value="pretrained_2head_herg_chemberta_quick">ChemBERTa Dual-Head · Quick</option>
                <option value="pretrained_2head_herg_molformer_model">MolFormer Dual-Head · Full</option>
                <option value="pretrained_2head_herg_molformer_quick">MolFormer Dual-Head · Quick</option>
              </select>
            </div>

            {/* GNN Toxicity Type Model */}
            <div>
              <label className="text-sm mb-1 block" style={{ color: 'var(--text-muted)' }}>
                Toxicity Type Model (GNN):
              </label>
              <p className='text-xs mb-2' style={{ color: 'ver(--text-faint)' }}>
                Chọn model để profile 12 Tox21 assay tasks
              </p>
              <select 
              value={toxTypeModel}
              onChange={(e) => setToxTypeModel(e.target.value)}
              disabled={isAnalyzing}
              className='w-full px-3 py-2 rounded-lg text-sm'
              style={{
                backgroundColor: 'var(--surface-alt)',
                border: '1px solid var(--border)',
                color: 'var(--text)', 
              }}
              >
                <option value="tox21_gatv2_model">GATv2 Tox21 · 12 assays · Recommended</option>
                <option value="pretrained_2head_herg_chemberta_model">ChemBERTa Dual-Head · Tox21 head (Full)</option>
                <option value="pretrained_2head_herg_chemberta_quick">ChemBERTa Dual-Head · Tox21 head (Quick)</option>
                <option value="pretrained_2head_herg_molformer_model">MolFormer Dual-Head · Tox21 head (Full)</option>
                <option value="pretrained_2head_herg_molformer_quick">MolFormer Dual-Head · Tox21 head (Quick)</option>
              </select>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    </section>
  );
}
