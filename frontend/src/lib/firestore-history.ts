// frontend/src/lib/firestore-history.ts
import {
    collection,
    addDoc,
    getDocs,
    deleteDoc,
    doc,
    query,
    orderBy,
    limit,
    serverTimestamp,
} from 'firebase/firestore';
import { db } from '../firebase-config';

export interface AnalysisMeta {
    language?: string;
    inferenceBackend?: string;
    binaryModel?: string;
    toxTypeModel?: string;
    sessionId?: string;
    riskLevel?: string;
}

export interface HistoryEntry {
    id: string;
    smiles: string;
    timestamp: number;
    verdict: 'toxic' | 'warning' | 'non-toxic';
    score: number;
    language?: string;
    inferenceBackend?: string;
    binaryModel?: string;
    toxTypeModel?: string;
    sessionId?: string;
    riskLevel?: string;
}

// Sub-collection: users/{uid}/analyses
function analysesRef(uid: string) {
    return collection(db, 'users', uid, 'analyses');
}

export async function saveAnalysisToFirestore (
    uid: string,
    smiles: string,
    verdict: 'toxic' | 'warning' | 'non-toxic',
    score: number,
    meta: AnalysisMeta = {},
): Promise<void> {
    await addDoc(analysesRef(uid), {
        smiles,
        verdict,
        score,
        ...meta,
        createdAt: serverTimestamp(),
    });
}

export async function loadAnalysesFromFirestore(
    uid: string,
    maxResults = 50,
): Promise<HistoryEntry[]> {
    const q = query(analysesRef(uid), orderBy('createdAt', 'desc'), limit(maxResults));
    const snapshot = await getDocs(q);

    return snapshot.docs.map((docSnap) => {
        const data = docSnap.data();
        return {
            id: docSnap.id,
            smiles: String(data.smiles ?? ''),
            timestamp: data.createdAt?.toMillis() ?? Date.now(),
            verdict: (data.verdict as HistoryEntry['verdict']) ?? 'non-toxic',
            score: typeof data.score === 'number' ? data.score : 0,
            language: typeof data.language === 'string' ? data.language : undefined,
            inferenceBackend: typeof data.inferenceBackend === 'string' ? data.inferenceBackend : undefined,
            binaryModel: typeof data.binaryModel === 'string' ? data.binaryModel : undefined,
            toxTypeModel: typeof data.toxTypeModel === 'string' ? data.toxTypeModel : undefined,
            sessionId: typeof data.sessionId === 'string' ? data.sessionId : undefined,
            riskLevel: typeof data.riskLevel === 'string' ? data.riskLevel : undefined,
        };
    });
}

export async function deleteAnalysisFromFirestore (
    uid: string,
    entryId: string,
): Promise<void> {
    await deleteDoc(doc(db, 'users', uid, 'analyses', entryId));
}