import { useMemo, useRef, useState } from 'react';
import { Editor } from 'ketcher-react';
import { StandaloneStructServiceProvider } from 'ketcher-standalone';
import { AlertCircle, PenTool, WandSparkles } from 'lucide-react';
import { toast } from 'sonner';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Button } from './ui/button';
import 'ketcher-react/dist/index.css';

type KetcherLike = {
	getSmiles: (isExtended?: boolean) => Promise<string>;
	setMolecule?: (structure: string) => Promise<void>;
};

interface SmilesDrawingPanelProps {
	currentSmiles: string;
	onSmilesExtracted: (smiles: string) => void;
	disabled?: boolean;
}

export function SmilesDrawingPanel({
	currentSmiles,
	onSmilesExtracted,
	disabled = false,
}: SmilesDrawingPanelProps) {
	const editorRef = useRef<KetcherLike | null>(null);
	const [extracting, setExtracting] = useState(false);
	const [syncing, setSyncing] = useState(false);
	const [editorError, setEditorError] = useState<string | null>(null);

	const structServiceProvider = useMemo(() => new StandaloneStructServiceProvider(), []);

	const handleExtractSmiles = async () => {
		if (disabled || !editorRef.current) {
			return;
		}

		setEditorError(null);
		setExtracting(true);
		try {
			const smiles = (await editorRef.current.getSmiles()).trim();
			if (!smiles) {
				setEditorError('No valid structure found. Draw a molecule first.');
				return;
			}

			onSmilesExtracted(smiles);
			toast.success('SMILES extracted from drawing.');
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Failed to export SMILES from drawing.';
			setEditorError(message);
		} finally {
			setExtracting(false);
		}
	};

	const handleSyncFromInput = async () => {
		if (disabled || !editorRef.current || !editorRef.current.setMolecule) {
			return;
		}

		const incoming = currentSmiles.trim();
		if (!incoming) {
			setEditorError('Current input is empty. Type or extract SMILES first.');
			return;
		}

		setEditorError(null);
		setSyncing(true);
		try {
			await editorRef.current.setMolecule(incoming);
			toast.success('Drawing canvas synced with current SMILES.');
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Failed to load SMILES into drawing editor.';
			setEditorError(message);
		} finally {
			setSyncing(false);
		}
	};

	return (
		<div className="space-y-3">
			<div className="flex flex-col gap-2 rounded-lg border p-3 md:flex-row md:items-center md:justify-between" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' }}>
				<div className="text-sm" style={{ color: 'var(--text-muted)' }}>
					Use Ketcher to draw a structure, then export canonical SMILES for analysis.
				</div>
				<div className="flex flex-wrap gap-2">
					<Button
						type="button"
						variant="outline"
						onClick={handleSyncFromInput}
						disabled={disabled || syncing}
					>
						<PenTool className="mr-2 h-4 w-4" />
						{syncing ? 'Syncing...' : 'Load Current SMILES'}
					</Button>
					<Button
						type="button"
						onClick={handleExtractSmiles}
						disabled={disabled || extracting}
						className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700"
					>
						<WandSparkles className="mr-2 h-4 w-4" />
						{extracting ? 'Extracting...' : 'Use Drawn SMILES'}
					</Button>
				</div>
			</div>

			{editorError && (
				<Alert variant="destructive">
					<AlertCircle className="h-4 w-4" />
					<AlertTitle>Drawing Error</AlertTitle>
					<AlertDescription>{editorError}</AlertDescription>
				</Alert>
			)}

			<div className="overflow-hidden rounded-xl border" style={{ borderColor: 'var(--border)' }}>
				<div style={{ minHeight: 420 }}>
					<Editor
						staticResourcesUrl="/"
						structServiceProvider={structServiceProvider}
						errorHandler={(message) => setEditorError(message)}
						onInit={(ketcher) => {
							editorRef.current = ketcher as unknown as KetcherLike;
						}}
						disableMacromoleculesEditor
					/>
				</div>
			</div>
		</div>
	);
}
