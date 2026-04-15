import { useCallback, useEffect, useMemo, useState } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { AlertCircle, CheckCircle2, ImagePlus, Loader2, UploadCloud } from 'lucide-react';
import { toast } from 'sonner';
import {
	extractSmilesFromImage,
	SmilesImageExtractionError,
	type SmilesImageExtractionResponse,
} from '../../lib/api';
import { Alert, AlertDescription, AlertTitle } from './ui/alert';
import { Button } from './ui/button';

const MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024;

const ACCEPTED_TYPES: Record<string, string[]> = {
	'image/png': ['.png'],
	'image/jpg': ['.jpg'],
	'image/jpeg': ['.jpeg', '.jpg'],
	'image/webp': ['.webp'],
};

interface SmilesImageUploadPanelProps {
	onSmilesExtracted: (smiles: string) => void;
	disabled?: boolean;
}

function mapErrorMessage(code: string, fallback: string): string {
	switch (code) {
		case 'unsupported_image_format':
			return 'Unsupported image format. Please upload PNG/JPG/JPEG/WebP.';
		case 'image_too_large':
			return 'Image is too large. The maximum upload size is 5MB.';
		case 'smiles_not_detected':
			return 'No molecule text was detected from this image. Try a cleaner image.';
		case 'invalid_smiles_from_image':
			return 'Image was parsed, but extracted SMILES is invalid. Please re-upload a clearer structure image.';
		case 'extraction_service_unavailable':
			return 'Image extraction service is temporarily unavailable. Please try again later.';
		default:
			return fallback;
	}
}

function parseRejection(rejections: FileRejection[]): { code: string; message: string } | null {
	const firstError = rejections[0]?.errors?.[0]?.code;
	if (!firstError) {
		return null;
	}

	if (firstError === 'file-too-large') {
		return {
			code: 'image_too_large',
			message: 'Image is too large. The maximum upload size is 5MB.',
		};
	}

	return {
		code: 'unsupported_image_format',
		message: 'Unsupported image format. Please upload PNG/JPG/JPEG/WebP.',
	};
}

export function SmilesImageUploadPanel({ onSmilesExtracted, disabled = false }: SmilesImageUploadPanelProps) {
	const [file, setFile] = useState<File | null>(null);
	const [previewUrl, setPreviewUrl] = useState<string | null>(null);
	const [isExtracting, setIsExtracting] = useState(false);
	const [errorCode, setErrorCode] = useState<string | null>(null);
	const [errorMessage, setErrorMessage] = useState<string | null>(null);
	const [result, setResult] = useState<SmilesImageExtractionResponse | null>(null);

	const clearMessages = () => {
		setErrorCode(null);
		setErrorMessage(null);
		setResult(null);
	};

	const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
		clearMessages();

		if (rejectedFiles.length > 0) {
			const parsed = parseRejection(rejectedFiles);
			if (parsed) {
				setErrorCode(parsed.code);
				setErrorMessage(parsed.message);
			}
			return;
		}

		const selected = acceptedFiles[0];
		if (!selected) {
			return;
		}

		setFile(selected);
	}, []);

	const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
		onDrop,
		accept: ACCEPTED_TYPES,
		maxSize: MAX_IMAGE_SIZE_BYTES,
		multiple: false,
		noClick: true,
		disabled: disabled || isExtracting,
	});

	useEffect(() => {
		if (!file) {
			setPreviewUrl(null);
			return;
		}

		const nextUrl = URL.createObjectURL(file);
		setPreviewUrl(nextUrl);
		return () => {
			URL.revokeObjectURL(nextUrl);
		};
	}, [file]);

	const canExtract = useMemo(() => Boolean(file) && !disabled && !isExtracting, [file, disabled, isExtracting]);

	const handleExtract = async () => {
		if (!file || !canExtract) {
			return;
		}

		setIsExtracting(true);
		clearMessages();

		try {
			const extracted = await extractSmilesFromImage(file);
			setResult(extracted);
			const smiles = (extracted.canonical_smiles ?? extracted.smiles ?? '').trim();

			if (!smiles) {
				setErrorCode('smiles_not_detected');
				setErrorMessage('No SMILES sequence was detected from this image.');
				return;
			}

			onSmilesExtracted(smiles);
			toast.success('SMILES extracted from uploaded image.');
			if (Array.isArray(extracted.warnings) && extracted.warnings.length > 0) {
				toast.message(`Extraction warnings: ${extracted.warnings.join(', ')}`);
			}
		} catch (error) {
			if (error instanceof SmilesImageExtractionError) {
				setErrorCode(error.code);
				setErrorMessage(mapErrorMessage(error.code, error.message));
				return;
			}

			setErrorCode('extraction_service_unavailable');
			setErrorMessage('Image extraction failed due to an unexpected error.');
		} finally {
			setIsExtracting(false);
		}
	};

	return (
		<div className="space-y-3">
			<div
				{...getRootProps()}
				className="cursor-pointer rounded-xl border-2 border-dashed p-6 transition-colors"
				style={{
					borderColor: isDragActive ? 'var(--accent-blue)' : 'var(--border)',
					backgroundColor: isDragActive ? 'rgba(59,130,246,0.08)' : 'var(--surface-alt)',
				}}
			>
				<input {...getInputProps()} />
				<div className="flex flex-col items-center gap-2 text-center">
					<UploadCloud className="h-8 w-8" style={{ color: 'var(--accent-blue)' }} />
					<p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
						Drag and drop a molecule image here
					</p>
					<p className="text-xs" style={{ color: 'var(--text-muted)' }}>
						Allowed: PNG/JPG/JPEG/WebP, max 5MB
					</p>
					<Button type="button" variant="outline" onClick={open} disabled={disabled || isExtracting}>
						<ImagePlus className="mr-2 h-4 w-4" />
						Choose Image
					</Button>
				</div>
			</div>

			{previewUrl && (
				<div className="overflow-hidden rounded-xl border" style={{ borderColor: 'var(--border)' }}>
					<img src={previewUrl} alt="Molecule upload preview" className="max-h-[280px] w-full object-contain bg-white" />
				</div>
			)}

			<div className="flex items-center gap-2">
				<Button type="button" onClick={handleExtract} disabled={!canExtract}>
					{isExtracting ? (
						<>
							<Loader2 className="mr-2 h-4 w-4 animate-spin" />
							Extracting...
						</>
					) : (
						<>
							<CheckCircle2 className="mr-2 h-4 w-4" />
							Extract SMILES from Image
						</>
					)}
				</Button>
				{file && (
					<p className="text-xs" style={{ color: 'var(--text-muted)' }}>
						{file.name}
					</p>
				)}
			</div>

			{errorCode && errorMessage && (
				<Alert variant="destructive">
					<AlertCircle className="h-4 w-4" />
					<AlertTitle>{errorCode}</AlertTitle>
					<AlertDescription>{errorMessage}</AlertDescription>
				</Alert>
			)}

			{result?.canonical_smiles && (
				<Alert>
					<CheckCircle2 className="h-4 w-4" />
					<AlertTitle>Extraction Success</AlertTitle>
					<AlertDescription>
						<p>Canonical SMILES: {result.canonical_smiles}</p>
						{typeof result.confidence === 'number' && (
							<p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
						)}
					</AlertDescription>
				</Alert>
			)}
		</div>
	);
}
