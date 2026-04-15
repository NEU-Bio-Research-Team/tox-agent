import { Suspense, lazy, useCallback, useEffect, useRef, useState, type ComponentType } from 'react';
import { Loader2, RefreshCw, CheckCircle, XCircle, Zap } from 'lucide-react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from './ui/collapsible';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { previewSmiles, SmilesPreviewError, type SmilesPreviewResponse } from '../../lib/api';

const SmilesImageUploadPanel = lazy(async () => {
	const module = await import('./smiles-image-upload-panel');
	return { default: module.SmilesImageUploadPanel };
});

interface HeroSectionProps {
	value: string;
	onChange: (value: string) => void;
	onAnalyze: (opts: { binaryModel: string; toxTypeModel: string }) => void;
	isAnalyzing: boolean;
}

type InputMode = 'text' | 'draw' | 'image';

type SmilesDrawingPanelComponent = ComponentType<{
	currentSmiles: string;
	onSmilesExtracted: (smiles: string) => void;
	disabled?: boolean;
	onReady?: () => void;
}>;

const exampleMolecules = [
	{ name: 'Caffeine', icon: 'C', smiles: 'Cn1cnc2c1c(=O)n(c(=O)n2C)C' },
	{ name: 'Aspirin', icon: 'A', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
	{ name: 'Ethanol', icon: 'E', smiles: 'CCO' },
	{ name: 'Haloperidol', icon: 'H', smiles: 'O=C(CCCN1CCC(c2ccc(Cl)cc2)(O)CC1)c1ccc(F)cc1' },
];

export function HeroSection({ value, onChange, onAnalyze, isAnalyzing }: HeroSectionProps) {
	const trimmedSmiles = value.trim();
	const [inputMode, setInputMode] = useState<InputMode>('text');
	const [validationState, setValidationState] = useState<'idle' | 'checking' | 'valid' | 'invalid'>('idle');
	const [validationMessage, setValidationMessage] = useState('');
	const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
	const [threshold, setThreshold] = useState(0.5);
	const [binaryModel, setBinaryModel] = useState<string>('pretrained_2head_herg_chemberta_model');
	const [toxTypeModel, setToxTypeModel] = useState<string>('tox21_gatv2_model');
	const [DrawPanelComponent, setDrawPanelComponent] = useState<SmilesDrawingPanelComponent | null>(null);
	const [drawEditorRequested, setDrawEditorRequested] = useState(false);
	const [drawEditorLoading, setDrawEditorLoading] = useState(false);
	const [drawEditorSlow, setDrawEditorSlow] = useState(false);
	const [drawEditorError, setDrawEditorError] = useState<string | null>(null);
	const [smilesPreview, setSmilesPreview] = useState<SmilesPreviewResponse | null>(null);
	const [smilesPreviewLoading, setSmilesPreviewLoading] = useState(false);
	const [smilesPreviewError, setSmilesPreviewError] = useState<string | null>(null);
	const drawEditorImportRef = useRef<Promise<typeof import('./smiles-drawing-panel')> | null>(null);

	const importDrawEditor = useCallback(() => {
		if (!drawEditorImportRef.current) {
			drawEditorImportRef.current = import('./smiles-drawing-panel');
		}
		return drawEditorImportRef.current;
	}, []);

	const loadDrawEditor = useCallback(async () => {
		if (DrawPanelComponent || drawEditorLoading) {
			return;
		}

		setDrawEditorError(null);
		setDrawEditorSlow(false);
		setDrawEditorLoading(true);

		const slowTimer = window.setTimeout(() => {
			setDrawEditorSlow(true);
		}, 8000);

		try {
			const module = await importDrawEditor();
			setDrawPanelComponent(() => module.SmilesDrawingPanel as SmilesDrawingPanelComponent);
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Unknown error while loading Ketcher.';
			setDrawEditorError(`Unable to load drawing editor. ${message}`);
		} finally {
			window.clearTimeout(slowTimer);
			setDrawEditorLoading(false);
		}
	}, [DrawPanelComponent, drawEditorLoading, importDrawEditor]);

	useEffect(() => {
		if (DrawPanelComponent) {
			return;
		}

		const connection = (navigator as unknown as { connection?: { saveData?: boolean } }).connection;
		if (connection?.saveData) {
			return;
		}

		const requestIdle = (window as unknown as { requestIdleCallback?: (cb: () => void, opts?: { timeout?: number }) => number })
			.requestIdleCallback;
		const cancelIdle = (window as unknown as { cancelIdleCallback?: (id: number) => void }).cancelIdleCallback;

		if (requestIdle && cancelIdle) {
			const idleId = requestIdle(() => {
				void importDrawEditor();
			}, { timeout: 2500 });
			return () => cancelIdle(idleId);
		}

		const timer = window.setTimeout(() => {
			void importDrawEditor();
		}, 2000);
		return () => window.clearTimeout(timer);
	}, [DrawPanelComponent, importDrawEditor]);

	useEffect(() => {
		if (!trimmedSmiles) {
			setValidationState('idle');
			setValidationMessage('');
			return;
		}

		setValidationState('valid');
		setValidationMessage('Ready to call API. Detailed validation will be handled by the backend.');
	}, [trimmedSmiles]);

	useEffect(() => {
		if (!trimmedSmiles || isAnalyzing) {
			setSmilesPreview(null);
			setSmilesPreviewLoading(false);
			setSmilesPreviewError(null);
			return;
		}

		let cancelled = false;
		const timer = window.setTimeout(async () => {
			setSmilesPreviewLoading(true);
			setSmilesPreviewError(null);
			try {
				const preview = await previewSmiles(trimmedSmiles);
				if (!cancelled) {
					setSmilesPreview(preview);
				}
			} catch (error) {
				if (cancelled) {
					return;
				}

				setSmilesPreview(null);
				if (error instanceof SmilesPreviewError) {
					setSmilesPreviewError(error.message);
				} else {
					const message = error instanceof Error ? error.message : 'Failed to render SMILES preview.';
					setSmilesPreviewError(message);
				}
			} finally {
				if (!cancelled) {
					setSmilesPreviewLoading(false);
				}
			}
		}, 400);

		return () => {
			cancelled = true;
			window.clearTimeout(timer);
		};
	}, [trimmedSmiles, isAnalyzing]);

	const getButtonState = () => {
		if (isAnalyzing) {
			return { text: 'Analyzing...', icon: Loader2, disabled: true, className: 'animate-spin' };
		}
		if (!value.trim()) {
			return { text: 'Analyze', icon: Zap, disabled: true, className: '' };
		}
		return { text: 'Analyze', icon: Zap, disabled: false, className: '' };
	};

	const buttonState = getButtonState();
	const ButtonIcon = buttonState.icon;

	const tabFallback = (
		<div className="rounded-xl border p-4 text-sm" style={{ borderColor: 'var(--border)', color: 'var(--text-muted)' }}>
			Loading input tools...
		</div>
	);

	return (
		<section className="pt-20 pb-12">
			<div className="text-center mb-10">
				<h1
					className="text-5xl font-bold mb-3 bg-gradient-to-r from-[var(--accent-blue)] to-[var(--accent-green)] bg-clip-text"
					style={{
						WebkitBackgroundClip: 'text',
						WebkitTextFillColor: 'transparent',
						backgroundClip: 'text',
					}}
				>
					Drug Toxicity Analysis
				</h1>
				<p className="text-base mb-1" style={{ color: 'var(--text-muted)' }}>
					Drug Toxicity Analysis using Multi-Agent AI System
				</p>
				<p className="text-sm" style={{ color: 'var(--text-faint)' }}>
					Type SMILES, draw a molecule, or upload a structure image to start analysis
				</p>
			</div>

			<div className="rounded-2xl p-8 shadow-lg" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
				<Tabs
					value={inputMode}
					onValueChange={(next) => {
						const nextMode = next as InputMode;
						setInputMode(nextMode);
						if (nextMode === 'draw' && !DrawPanelComponent && !drawEditorLoading) {
							setDrawEditorRequested(true);
							void loadDrawEditor();
						}
					}}
					className="mb-4"
				>
					<TabsList className="grid w-full grid-cols-3" style={{ backgroundColor: 'var(--surface-alt)' }}>
						<TabsTrigger value="text">Type SMILES</TabsTrigger>
						<TabsTrigger value="draw">Draw Molecule</TabsTrigger>
						<TabsTrigger value="image">Upload Image</TabsTrigger>
					</TabsList>

					<TabsContent value="text" className="space-y-4 pt-3">
						<div className="relative">
							<div className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-faint)' }}>
								<svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
									<circle cx="12" cy="12" r="3" />
									<path d="M12 1v6m0 6v6m0-6h6m-6 0H6" />
								</svg>
							</div>
							<Textarea
								value={value}
								onChange={(event) => onChange(event.target.value)}
								placeholder="CC(=O)Oc1ccccc1C(=O)O"
								className="pl-12 pr-10 py-4 min-h-[80px] font-mono text-sm resize-none"
								style={{
									backgroundColor: 'var(--surface-alt)',
									border:
										validationState === 'valid'
											? '1.5px solid var(--accent-green)'
											: validationState === 'invalid'
												? '1.5px solid var(--accent-red)'
												: '1.5px solid var(--border)',
									color: 'var(--text)',
									letterSpacing: '0.02em',
								}}
								disabled={isAnalyzing}
								spellCheck={false}
								autoCorrect="off"
								autoComplete="off"
							/>
							{value && !isAnalyzing && (
								<button
									type="button"
									onClick={() => onChange('')}
									className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-[var(--surface)] transition-colors"
									style={{ color: 'var(--text-faint)' }}
								>
									x
								</button>
							)}
						</div>

						<div>
							<p className="text-xs uppercase mb-2" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
								Quick Examples:
							</p>
							<div className="flex flex-wrap gap-2">
								{exampleMolecules.map((mol) => (
									<button
										type="button"
										key={mol.name}
										onClick={() => onChange(mol.smiles)}
										className="px-4 py-2 rounded-full text-xs font-medium transition-all"
										style={{
											backgroundColor: 'var(--surface-alt)',
											border: '1px solid var(--border)',
											color: 'var(--text-muted)',
										}}
										onMouseEnter={(event) => {
											event.currentTarget.style.backgroundColor = 'var(--surface)';
											event.currentTarget.style.borderColor = 'var(--accent-blue)';
											event.currentTarget.style.color = 'var(--text)';
										}}
										onMouseLeave={(event) => {
											event.currentTarget.style.backgroundColor = 'var(--surface-alt)';
											event.currentTarget.style.borderColor = 'var(--border)';
											event.currentTarget.style.color = 'var(--text-muted)';
										}}
									>
										{mol.icon} {mol.name}
									</button>
								))}
							</div>
						</div>
					</TabsContent>

					<TabsContent value="draw" className="pt-3">
						<div className="space-y-3">
							{!DrawPanelComponent && !drawEditorRequested && (
								<div
									className="rounded-xl border p-4 text-sm"
									style={{ borderColor: 'var(--border)', color: 'var(--text-muted)', backgroundColor: 'var(--surface-alt)' }}
								>
									The drawing editor will load when needed to keep the main UI responsive.
								</div>
							)}

							{drawEditorRequested && drawEditorLoading && tabFallback}

							{drawEditorRequested && drawEditorLoading && drawEditorSlow && (
								<div
									className="rounded-xl border p-4 text-sm"
									style={{ borderColor: 'var(--border)', color: 'var(--text-muted)', backgroundColor: 'var(--surface-alt)' }}
								>
									First-time Ketcher load can take 10-30 seconds on slower networks or CPUs.
								</div>
							)}

							{drawEditorRequested && drawEditorError && (
								<div
									className="rounded-xl border p-4 text-sm"
									style={{ borderColor: 'rgba(239,68,68,0.35)', color: 'var(--accent-red)', backgroundColor: 'rgba(239,68,68,0.08)' }}
								>
									<div>{drawEditorError}</div>
									<div className="mt-2">
										<Button
											type="button"
											variant="outline"
											onClick={() => {
												setDrawEditorRequested(true);
												void loadDrawEditor();
											}}
										>
											Retry Loading Editor
										</Button>
									</div>
								</div>
							)}

							{DrawPanelComponent && !drawEditorError && (
								<DrawPanelComponent
									currentSmiles={value}
									onSmilesExtracted={onChange}
									disabled={isAnalyzing}
									onReady={() => {
										setDrawEditorSlow(false);
										setDrawEditorError(null);
									}}
								/>
							)}
						</div>
					</TabsContent>

					<TabsContent value="image" className="pt-3">
						<Suspense fallback={tabFallback}>
							<SmilesImageUploadPanel onSmilesExtracted={onChange} disabled={isAnalyzing} />
						</Suspense>
					</TabsContent>
				</Tabs>

				{validationState !== 'idle' && (
					<div
						className="px-4 py-2 rounded-lg text-sm mb-4"
						style={{
							backgroundColor:
								validationState === 'valid'
									? 'rgba(34,197,94,0.08)'
									: validationState === 'invalid'
										? 'rgba(239,68,68,0.08)'
										: 'transparent',
							borderLeft:
								validationState === 'valid'
									? '2px solid var(--accent-green)'
									: validationState === 'invalid'
										? '2px solid var(--accent-red)'
										: 'none',
							color:
								validationState === 'valid'
									? 'var(--accent-green)'
									: validationState === 'invalid'
										? 'var(--accent-red)'
										: 'var(--text-faint)',
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

				<div className="mb-4 rounded-lg border px-4 py-3" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' }}>
					<p className="text-xs uppercase mb-1" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
						Current SMILES for Analysis
					</p>
					<p className="font-mono text-sm break-all" style={{ color: 'var(--text)' }}>
						{trimmedSmiles || 'No SMILES selected yet.'}
					</p>
				</div>

				<div className="mb-4 rounded-lg border px-4 py-3" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' }}>
					<div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
						<div className="flex-1">
							<p className="text-xs uppercase mb-1" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
								RDKit Molecule Preview
							</p>

							{!trimmedSmiles && (
								<p className="text-sm" style={{ color: 'var(--text-muted)' }}>
									Enter or extract a SMILES string to render a structure preview.
								</p>
							)}

							{trimmedSmiles && smilesPreviewLoading && (
								<p className="text-sm" style={{ color: 'var(--text-muted)' }}>
									Rendering preview...
								</p>
							)}

							{trimmedSmiles && smilesPreviewError && (
								<p className="text-sm" style={{ color: 'var(--accent-red)' }}>
									{smilesPreviewError}
								</p>
							)}

							{trimmedSmiles && !smilesPreviewLoading && smilesPreview?.molecule_png_base64 && (
								<div className="mt-3 overflow-hidden rounded-xl border" style={{ borderColor: 'var(--border)', backgroundColor: '#ffffff' }}>
									<img
										src={`data:image/png;base64,${smilesPreview.molecule_png_base64}`}
										alt="RDKit molecule preview"
										className="w-full object-contain"
										style={{ maxHeight: 280 }}
									/>
								</div>
							)}

							{trimmedSmiles && !smilesPreviewLoading && smilesPreview?.canonical_smiles && (
								<p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
									Canonical SMILES: <span className="font-mono" style={{ color: 'var(--text)' }}>{smilesPreview.canonical_smiles}</span>
								</p>
							)}
						</div>

						<div className="flex justify-end">
							<Button
								type="button"
								variant="outline"
								onClick={() => onChange('')}
								disabled={isAnalyzing || !trimmedSmiles}
							>
								<RefreshCw className="mr-2 h-4 w-4" />
								Retry
							</Button>
						</div>
					</div>
				</div>

				<div className="mb-4 flex items-center justify-end">
					<Button
						onClick={() => onAnalyze({ binaryModel, toxTypeModel })}
						disabled={buttonState.disabled}
						className="h-[48px] px-7 text-base font-semibold rounded-lg"
						style={{
							backgroundColor: buttonState.disabled ? 'var(--border)' : 'var(--accent-blue)',
							color: '#ffffff',
							opacity: buttonState.disabled ? 0.5 : 1,
						}}
					>
						<ButtonIcon className={`w-4 h-4 mr-2 ${buttonState.className}`} />
						{buttonState.text}
					</Button>
				</div>

				<Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen}>
					<CollapsibleTrigger className="text-sm flex items-center gap-1" style={{ color: 'var(--text-muted)' }}>
						<span>{isAdvancedOpen ? 'v' : '>'}</span>
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
									onChange={(event) => setThreshold(parseFloat(event.target.value))}
									className="flex-1"
									style={{
										background:
											'linear-gradient(to right, var(--accent-green) 0%, var(--accent-yellow) 50%, var(--accent-red) 100%)',
									}}
								/>
								<span className="font-mono text-sm w-12" style={{ color: 'var(--text)' }}>
									{threshold.toFixed(2)}
								</span>
							</div>
						</div>

						<div>
							<label className="text-sm mb-1 block" style={{ color: 'var(--text-muted)' }}>
								Binary Toxicity Model (GNN):
							</label>
							<p className="text-xs mb-2" style={{ color: 'var(--text-faint)' }}>
								Choose a backbone model or ensemble for binary toxicity probability prediction.
							</p>
							<select
								value={binaryModel}
								onChange={(event) => setBinaryModel(event.target.value)}
								disabled={isAnalyzing}
								className="w-full px-3 py-2 rounded-lg text-sm"
								style={{
									backgroundColor: 'var(--surface-alt)',
									border: '1px solid var(--border)',
									color: 'var(--text)',
								}}
							>
								<option value="dualhead_ensemble6_simple">Ensemble-6 Simple · joint_auc_beta3 0.8467 · Recommended</option>
								<option value="dualhead_ensemble3_weighted">Ensemble-3 Weighted · joint_auc_beta3 0.8466</option>
								<option value="dualhead_ensemble3_simple">Ensemble-3 Simple · joint_auc_beta3 0.8455</option>
								<option value="dualhead_ensemble5_simple">Ensemble-5 Simple · joint_auc_beta3 0.8451</option>
								<option value="tox21_ensemble_3_best">Legacy Ensemble-3 Alias (backward compatibility)</option>
								<option value="pretrained_2head_herg_molformer_model">MolFormer Dual-Head · Full</option>
								<option value="pretrained_2head_herg_chemberta_model">ChemBERTa Dual-Head · Full</option>
								<option value="pretrained_2head_herg_pubchem_model">PubChem Dual-Head · Full</option>
								<option value="pretrained_2head_herg_chemberta_quick">ChemBERTa Dual-Head · Quick</option>
								<option value="pretrained_2head_herg_molformer_quick">MolFormer Dual-Head · Quick</option>
								<option value="pretrained_2head_herg_pubchem_quick">PubChem Dual-Head · Quick</option>
							</select>
						</div>

						<div>
							<label className="text-sm mb-1 block" style={{ color: 'var(--text-muted)' }}>
								Toxicity Type Model (GNN):
							</label>
							<p className="text-xs mb-2" style={{ color: 'var(--text-faint)' }}>
								Choose a model or ensemble to profile all 12 Tox21 assay tasks.
							</p>
							<select
								value={toxTypeModel}
								onChange={(event) => setToxTypeModel(event.target.value)}
								disabled={isAnalyzing}
								className="w-full px-3 py-2 rounded-lg text-sm"
								style={{
									backgroundColor: 'var(--surface-alt)',
									border: '1px solid var(--border)',
									color: 'var(--text)',
								}}
							>
								<option value="pretrained_2head_herg_chemberta_model">ChemBERTa Dual-Head · Tox21 head (Full) · Recommended</option>
								<option value="dualhead_ensemble3_weighted">Ensemble-3 Weighted · task-wise weighted (ChemBERTa + MolFormer + Pretrained-GIN)</option>
								<option value="dualhead_ensemble3_simple">Ensemble-3 Simple · mean (ChemBERTa + MolFormer + Pretrained-GIN)</option>
								<option value="dualhead_ensemble5_simple">Ensemble-5 Simple · mean (ChemBERTa + MolFormer + AFP + XGB + GPS)</option>
								<option value="tox21_ensemble_3_best">Legacy Ensemble-3 Alias (backward compatibility)</option>
								<option value="tox21_pretrained_gin_model">Pretrained-GIN (Hu et al.) · Tox21 task engine</option>
								<option value="tox21_gatv2_model">GATv2 Tox21 · 12 assays</option>
								<option value="pretrained_2head_herg_molformer_model">MolFormer Dual-Head · Tox21 head (Full)</option>
								<option value="dualhead_ensemble6_simple">Ensemble-6 Simple · Tox21 mix (Dual-Head + AFP + XGB + GPS + Pretrained-GIN)</option>
								<option value="pretrained_2head_herg_pubchem_model">PubChem Dual-Head · Tox21 head (Full)</option>
								<option value="pretrained_2head_herg_chemberta_quick">ChemBERTa Dual-Head · Tox21 head (Quick)</option>
								<option value="pretrained_2head_herg_molformer_quick">MolFormer Dual-Head · Tox21 head (Quick)</option>
								<option value="pretrained_2head_herg_pubchem_quick">PubChem Dual-Head · Tox21 head (Quick)</option>
							</select>
						</div>
					</CollapsibleContent>
				</Collapsible>
			</div>
		</section>
	);
}
