/**
 * Suggesting a molecule from what the user typed — and never asserting one.
 *
 * The previous heuristic (I04) accepted any whitespace-free run of characters
 * SMILES happens to use, which made two opposite mistakes at once:
 *
 * - `hello` and `aspirin` matched, and the composer then replaced the user's
 *   question with them as a molecule, losing the text entirely.
 * - `Phân tích CCO giúp tôi` did not match, because it contains spaces, so a
 *   question with a molecule in it reached the backend with no subject.
 *
 * What is here does two things differently. It tokenises against the SMILES
 * grammar's actual atom symbols, so a word made of letters that are not
 * element or aromatic symbols is rejected. And it scans inside a sentence,
 * returning every candidate rather than one, so the caller can ask when there
 * is more than one instead of guessing.
 *
 * This remains a suggestion. RDKit on the server is the only thing that
 * decides whether a string is a molecule; nothing here should be described to
 * the user as validation.
 */

/** Organic-subset atoms writable without brackets, longest first. */
const ORGANIC = ['Cl', 'Br', 'B', 'C', 'N', 'O', 'P', 'S', 'F', 'I'];
/** Lowercase aromatic atoms writable without brackets. */
const AROMATIC = ['b', 'c', 'n', 'o', 'p', 's'];
/** Bonds, branches, ring closures and disconnections. */
const STRUCTURAL = new Set(['-', '=', '#', '$', ':', '/', '\\', '(', ')', '.', '*']);

export interface SmilesScan {
  /** True when every character belongs to the SMILES grammar. */
  plausible: boolean;
  /** Atoms found outside brackets, plus one per bracket atom. */
  atomCount: number;
  /** Whether any bond, branch, ring closure or bracket appeared. */
  hasStructure: boolean;
}

/**
 * Walk `token` as SMILES. Structural only — this knows nothing about valence,
 * ring closure pairing or stereochemistry, and is not meant to.
 */
export function scanSmiles(token: string): SmilesScan {
  let index = 0;
  let atomCount = 0;
  let hasStructure = false;

  while (index < token.length) {
    const character = token[index];

    if (character === '[') {
      const close = token.indexOf(']', index);
      // An unclosed bracket is not SMILES, and refusing here is what keeps
      // `[` from silently behaving like a structural character.
      if (close === -1) return { plausible: false, atomCount, hasStructure };
      if (close === index + 1) return { plausible: false, atomCount, hasStructure };
      atomCount += 1;
      hasStructure = true;
      index = close + 1;
      continue;
    }

    const twoLetter = token.slice(index, index + 2);
    if (ORGANIC.includes(twoLetter)) {
      atomCount += 1;
      index += 2;
      continue;
    }
    if (ORGANIC.includes(character) || AROMATIC.includes(character)) {
      atomCount += 1;
      index += 1;
      continue;
    }
    if (STRUCTURAL.has(character)) {
      hasStructure = true;
      index += 1;
      continue;
    }
    if (character === '%') {
      // Two-digit ring closure.
      if (!/^\d\d/.test(token.slice(index + 1))) {
        return { plausible: false, atomCount, hasStructure };
      }
      hasStructure = true;
      index += 3;
      continue;
    }
    if (/\d/.test(character)) {
      hasStructure = true;
      index += 1;
      continue;
    }
    return { plausible: false, atomCount, hasStructure };
  }

  return { plausible: true, atomCount, hasStructure };
}

/**
 * Could this whole string be a molecule?
 *
 * Deliberately conservative about short all-letter tokens: `No` and `CN` both
 * tokenise cleanly, so requiring either some structure or at least three
 * atoms keeps ordinary two-letter words out while still accepting `CCO`.
 * A user who means a two-atom molecule has the dedicated SMILES field.
 */
export function looksLikeSmiles(text: string): boolean {
  const token = text.trim();
  if (token.length === 0 || /\s/.test(token)) return false;
  const scan = scanSmiles(token);
  if (!scan.plausible || scan.atomCount === 0) return false;
  return scan.hasStructure || scan.atomCount >= 3;
}

/**
 * Every token in a sentence that could be a molecule.
 *
 * Splits on whitespace and on punctuation that cannot appear in SMILES, so
 * `Phân tích CCO nhé.` yields `CCO` — the trailing full stop is a valid
 * SMILES character (disconnection), which is why a bare trailing `.` is
 * trimmed rather than tokenised.
 */
export function findSmilesCandidates(text: string): string[] {
  const seen = new Set<string>();
  const candidates: string[] = [];
  for (const raw of text.split(/[\s,;?!"'“”‘’]+/)) {
    const token = raw.replace(/\.+$/, '');
    if (!looksLikeSmiles(token)) continue;
    if (seen.has(token)) continue;
    seen.add(token);
    candidates.push(token);
  }
  return candidates;
}

export interface MoleculeSuggestion {
  /** The single candidate, when there is exactly one. */
  smiles: string | null;
  /** All candidates, so the caller can ask rather than pick. */
  candidates: string[];
  /** True when the message is nothing but the molecule. */
  isBareMolecule: boolean;
}

/**
 * What the composer should do with what was typed.
 *
 * The rule that matters: when the text is a *question* containing a molecule,
 * both survive. Replacing the question with the molecule is what made
 * `research_subject_missing` and silent text loss possible.
 */
export function suggestMolecule(text: string): MoleculeSuggestion {
  const trimmed = text.trim();
  const candidates = findSmilesCandidates(trimmed);
  return {
    smiles: candidates.length === 1 ? candidates[0] : null,
    candidates,
    isBareMolecule: candidates.length === 1 && candidates[0] === trimmed,
  };
}
