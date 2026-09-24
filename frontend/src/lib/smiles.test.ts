/**
 * I04's two opposite failures, and the cases that must keep working.
 *
 * The old heuristic accepted any whitespace-free run of SMILES-ish characters,
 * so `hello` became a molecule and the user's text was discarded, while
 * `Phân tích CCO giúp tôi` matched nothing and reached the backend with no
 * subject at all.
 */
import { describe, expect, it } from 'vitest';

import { findSmilesCandidates, looksLikeSmiles, suggestMolecule } from './smiles';

describe('looksLikeSmiles', () => {
  it.each([
    ['CCO', 'ethanol'],
    ['CC(=O)Oc1ccccc1C(=O)O', 'aspirin'],
    ['c1ccccc1', 'benzene'],
    ['C[C@H](N)C(=O)O', 'alanine with stereochemistry'],
    ['[Na+].[Cl-]', 'a disconnected salt'],
    ['C1CCCCC1', 'a ring closure'],
    ['C%10CCCCC%10', 'a two-digit ring closure'],
  ])('accepts %s (%s)', (smiles) => {
    expect(looksLikeSmiles(smiles)).toBe(true);
  });

  it.each([
    ['hello', 'the regression this function exists for'],
    ['aspirin', 'a compound name is not a structure'],
    ['toxicity', 'an ordinary English word'],
    ['SMILES', 'the word itself'],
    ['thalidomide', 'a longer name'],
    ['what', 'a question word'],
  ])('rejects %s (%s)', (word) => {
    expect(looksLikeSmiles(word)).toBe(false);
  });

  it('rejects a two-atom letter pair that is more likely a word', () => {
    // `No` tokenises cleanly as N + o. Requiring structure or three atoms is
    // what keeps it out; someone who means it has the SMILES field.
    expect(looksLikeSmiles('No')).toBe(false);
    expect(looksLikeSmiles('CN')).toBe(false);
  });

  it('rejects malformed brackets rather than treating them as structure', () => {
    expect(looksLikeSmiles('C[NH3')).toBe(false);
    expect(looksLikeSmiles('C[]C')).toBe(false);
  });

  it('rejects anything containing whitespace', () => {
    expect(looksLikeSmiles('CCO CCO')).toBe(false);
  });

  it('rejects the empty string', () => {
    expect(looksLikeSmiles('')).toBe(false);
    expect(looksLikeSmiles('   ')).toBe(false);
  });
});

describe('findSmilesCandidates', () => {
  it('finds a molecule inside a Vietnamese sentence', () => {
    // The exact case that used to reach the backend with no subject.
    expect(findSmilesCandidates('Phân tích CCO giúp tôi')).toEqual(['CCO']);
  });

  it('finds a molecule inside an English sentence', () => {
    expect(findSmilesCandidates('What is the hERG risk for CC(=O)Oc1ccccc1C(=O)O?')).toEqual([
      'CC(=O)Oc1ccccc1C(=O)O',
    ]);
  });

  it('trims a trailing full stop rather than reading it as a disconnection', () => {
    expect(findSmilesCandidates('Check CCO.')).toEqual(['CCO']);
  });

  it('returns every candidate when a sentence names more than one', () => {
    expect(findSmilesCandidates('Compare CCO and c1ccccc1')).toEqual(['CCO', 'c1ccccc1']);
  });

  it('does not repeat the same candidate', () => {
    expect(findSmilesCandidates('CCO vs CCO')).toEqual(['CCO']);
  });

  it('finds nothing in a question with no molecule', () => {
    expect(findSmilesCandidates('What does this hERG number mean?')).toEqual([]);
  });
});

describe('suggestMolecule', () => {
  it('reports a bare molecule as exactly that', () => {
    const suggestion = suggestMolecule('CCO');
    expect(suggestion.smiles).toBe('CCO');
    expect(suggestion.isBareMolecule).toBe(true);
  });

  it('keeps a sentence a sentence even when it contains a molecule', () => {
    const suggestion = suggestMolecule('Phân tích CCO giúp tôi');
    expect(suggestion.smiles).toBe('CCO');
    // The caller must send the text as well; this flag is what tells it to.
    expect(suggestion.isBareMolecule).toBe(false);
  });

  it('refuses to choose when a sentence names two molecules', () => {
    const suggestion = suggestMolecule('Compare CCO and c1ccccc1');
    expect(suggestion.smiles).toBeNull();
    expect(suggestion.candidates).toHaveLength(2);
  });

  it('suggests nothing for an ordinary question', () => {
    const suggestion = suggestMolecule('hello');
    expect(suggestion.smiles).toBeNull();
    expect(suggestion.candidates).toEqual([]);
  });
});
