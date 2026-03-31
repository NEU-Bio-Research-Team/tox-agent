import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { 
  Network, 
  Shield, 
  Atom, 
  BookOpen, 
  FileText, 
  ArrowDown,
  CheckCircle2,
  Clock,
} from 'lucide-react';
import { motion } from 'motion/react';

interface AgentPipelineProps {
  isActive: boolean;
}

export function AgentPipeline({ isActive }: AgentPipelineProps) {
  const agents = [
    {
      name: 'Orchestrator Agent',
      icon: Network,
      model: 'Gemini 2.0 Flash',
      description: 'Root coordinator managing parallel dispatch',
      color: 'from-blue-600 to-indigo-600',
      status: 'active',
      behavior: 'Parallel dispatch + Reflection loop',
    },
    {
      name: 'Screening Agent',
      icon: Shield,
      model: 'Gemini 2.0 Flash',
      description: 'Toxicity prediction via FastAPI endpoint',
      color: 'from-emerald-500 to-green-600',
      status: 'completed',
      tool: 'predict_toxicity(smiles)',
      output: 'p_toxic, label, confidence',
    },
    {
      name: 'Explainer Agent',
      icon: Atom,
      model: 'Gemini 2.0 Flash',
      description: 'GNN-based molecular attribution analysis',
      color: 'from-purple-500 to-pink-600',
      status: 'completed',
      tool: 'explain_molecule(smiles, epochs)',
      output: 'top_atoms, top_bonds, heatmap',
    },
    {
      name: 'Researcher Agent',
      icon: BookOpen,
      model: 'Gemini 2.0 Flash',
      description: 'Literature and database search via RAG',
      color: 'from-blue-500 to-cyan-600',
      status: 'completed',
      tool: 'lookup_pubchem, search_chembl, search_literature',
      output: 'known_toxicity, similar_compounds',
    },
    {
      name: 'Report Writer Agent',
      icon: FileText,
      model: 'Gemini 1.5 Pro',
      description: 'Comprehensive structured report generation',
      color: 'from-amber-500 to-orange-600',
      status: 'completed',
      output: 'Markdown + JSON report',
      prompt: 'Structured output, temperature=0.2',
    },
  ];

  return (
    <div className="space-y-4">
      <Card className="bg-white/80 backdrop-blur-md border-slate-200">
        <CardHeader>
          <CardTitle>Agent Execution Pipeline</CardTitle>
          <CardDescription>
            Multi-agent architecture with parallel processing and hierarchical coordination
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {agents.map((agent, idx) => (
            <div key={idx}>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <Card className={`bg-gradient-to-br ${agent.color} text-white border-0 shadow-lg`}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 bg-white/20 backdrop-blur-md rounded-lg flex items-center justify-center">
                          <agent.icon className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <h3 className="font-semibold text-white">{agent.name}</h3>
                            {idx === 0 && <Badge variant="secondary" className="bg-white/20 text-white text-xs">Root</Badge>}
                          </div>
                          <p className="text-xs text-white/80">{agent.model}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {agent.status === 'completed' ? (
                          <CheckCircle2 className="w-5 h-5 text-white" />
                        ) : (
                          <Clock className="w-5 h-5 text-white animate-pulse" />
                        )}
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <p className="text-sm text-white/90">{agent.description}</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {agent.behavior && (
                        <div className="bg-white/10 backdrop-blur-md rounded-lg p-3">
                          <p className="text-xs text-white/70 mb-1">Key Behavior</p>
                          <p className="text-sm text-white font-medium">{agent.behavior}</p>
                        </div>
                      )}
                      {agent.tool && (
                        <div className="bg-white/10 backdrop-blur-md rounded-lg p-3">
                          <p className="text-xs text-white/70 mb-1">Tool</p>
                          <p className="text-sm text-white font-mono">{agent.tool}</p>
                        </div>
                      )}
                      {agent.output && (
                        <div className="bg-white/10 backdrop-blur-md rounded-lg p-3">
                          <p className="text-xs text-white/70 mb-1">Output</p>
                          <p className="text-sm text-white">{agent.output}</p>
                        </div>
                      )}
                      {agent.prompt && (
                        <div className="bg-white/10 backdrop-blur-md rounded-lg p-3">
                          <p className="text-xs text-white/70 mb-1">Strategy</p>
                          <p className="text-sm text-white">{agent.prompt}</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {idx < agents.length - 1 && (
                <div className="flex justify-center py-2">
                  <ArrowDown className="w-6 h-6 text-slate-400" />
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Parallel Processing Note */}
      <Card className="bg-gradient-to-br from-indigo-50 to-blue-50 border-indigo-200">
        <CardContent className="p-6">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center shrink-0">
              <Network className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h4 className="font-semibold text-slate-900 mb-2">Parallel Processing Architecture</h4>
              <p className="text-sm text-slate-700 mb-3">
                The Screening and Researcher agents run concurrently for optimal performance. 
                The Orchestrator manages coordination and implements reflection loops to resolve conflicts.
              </p>
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="border-indigo-300 text-indigo-700">
                  Hierarchical Dispatch
                </Badge>
                <Badge variant="outline" className="border-indigo-300 text-indigo-700">
                  Conflict Resolution
                </Badge>
                <Badge variant="outline" className="border-indigo-300 text-indigo-700">
                  Aggregated Results
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
