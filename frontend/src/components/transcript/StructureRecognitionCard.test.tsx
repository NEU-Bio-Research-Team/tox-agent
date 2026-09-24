import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { StructureRecognitionCard } from './StructureRecognitionCard';

describe('StructureRecognitionCard', () => {
  it('shows OCR confidence neutrally and only prefills a new analysis', () => {
    const onUseSmiles = vi.fn();
    render(
      <StructureRecognitionCard
        content={{
          code: 'structure_recognized',
          smiles: 'CC(=O)Oc1ccccc1C(=O)O',
          canonical_smiles: 'CC(=O)Oc1ccccc1C(=O)O',
          confidence: 0.91,
        }}
        onUseSmiles={onUseSmiles}
      />,
    );

    expect(screen.getByText('Độ tin cậy nhận diện: 91%')).toBeInTheDocument();
    expect(screen.getByText(/không phải độ tin cậy dự đoán độc tính/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Chỉnh sửa SMILES/i }));
    expect(onUseSmiles).toHaveBeenCalledOnce();
    expect(onUseSmiles).toHaveBeenCalledWith('CC(=O)Oc1ccccc1C(=O)O');
  });

  it('states when the OCR contract did not provide confidence', () => {
    render(
      <StructureRecognitionCard
        content={{
          code: 'structure_recognized',
          smiles: 'CCO',
          canonical_smiles: 'CCO',
        }}
        onUseSmiles={vi.fn()}
      />,
    );

    expect(screen.getByText(/không cung cấp độ tin cậy/i)).toBeInTheDocument();
  });
});
