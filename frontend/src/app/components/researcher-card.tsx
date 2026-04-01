import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { BookOpen, Database, FileText, ExternalLink } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface ResearcherCardProps {
  isAnalyzing: boolean;
  expanded?: boolean;
}

export function ResearcherCard({ isAnalyzing, expanded = false }: ResearcherCardProps) {
  const result = {
    known_toxicity: 'Low toxicity profile in clinical use',
    confidence: 'high',
    similar_compounds: [
      { name: 'Salicylic acid', cid: '338', similarity: 0.87 },
      { name: 'Acetaminophen', cid: '1983', similarity: 0.72 },
    ],
    literature: [
      { 
        title: 'Aspirin toxicity profile in cardiovascular applications',
        source: 'PubMed',
        relevance: 0.94,
      },
      { 
        title: 'NSAID safety and mechanism of action',
        source: 'ChEMBL',
        relevance: 0.89,
      },
    ],
  };

  return (
    <Card className="bg-white/80 backdrop-blur-md border-slate-200 hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-lg flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <CardTitle>Researcher Agent</CardTitle>
              <CardDescription>Gemini 2.0 Flash - Literature & Database Search</CardDescription>
            </div>
          </div>
          <Badge 
            variant="outline" 
            className={
              result.confidence === 'high' 
                ? 'border-emerald-300 text-emerald-700'
                : 'border-amber-300 text-amber-700'
            }
          >
            {result.confidence} confidence
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Known Toxicity */}
        <div className="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Database className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-semibold text-slate-900">Known Toxicity Data</span>
          </div>
          <p className="text-sm text-slate-700">{result.known_toxicity}</p>
        </div>

        {!expanded && (
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-bold text-blue-600">{result.similar_compounds.length}</p>
              <p className="text-xs text-slate-600">Similar Compounds</p>
            </div>
            <div className="bg-slate-50 rounded-lg p-3 text-center">
              <p className="text-2xl font-bold text-blue-600">{result.literature.length}</p>
              <p className="text-xs text-slate-600">Literature Sources</p>
            </div>
          </div>
        )}

        {expanded && (
          <Tabs defaultValue="compounds" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="compounds">Similar Compounds</TabsTrigger>
              <TabsTrigger value="literature">Literature</TabsTrigger>
            </TabsList>
            
            <TabsContent value="compounds" className="space-y-2 mt-4">
              {result.similar_compounds.map((compound, idx) => (
                <div key={idx} className="bg-slate-50 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Database className="w-4 h-4 text-blue-600" />
                      <span className="text-sm font-semibold text-slate-900">{compound.name}</span>
                    </div>
                    <Badge variant="secondary">
                      {(compound.similarity * 100).toFixed(0)}% match
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-slate-500">PubChem CID: {compound.cid}</p>
                    <button className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1">
                      View <ExternalLink className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
            </TabsContent>
            
            <TabsContent value="literature" className="space-y-2 mt-4">
              {result.literature.map((paper, idx) => (
                <div key={idx} className="bg-slate-50 rounded-lg p-3">
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-start gap-2 flex-1">
                      <FileText className="w-4 h-4 text-blue-600 mt-0.5" />
                      <p className="text-sm font-medium text-slate-900 leading-snug">{paper.title}</p>
                    </div>
                    <Badge variant="secondary" className="shrink-0">
                      {(paper.relevance * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-slate-500">{paper.source}</p>
                    <button className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1">
                      Read <ExternalLink className="w-3 h-3" />
                    </button>
                  </div>
                </div>
              ))}
            </TabsContent>
          </Tabs>
        )}

        {/* Tools Used */}
        <div className="bg-slate-50 rounded-lg p-3">
          <p className="text-xs font-semibold text-slate-700 mb-2">Tools Used</p>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className="text-xs">lookup_pubchem()</Badge>
            <Badge variant="outline" className="text-xs">search_chembl()</Badge>
            <Badge variant="outline" className="text-xs">search_literature()</Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
