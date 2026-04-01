import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { FileText, Download, Eye, CheckCircle2, AlertTriangle, Info } from 'lucide-react';
import { Button } from './ui/button';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';

interface ReportCardProps {
  isAnalyzing: boolean;
  expanded?: boolean;
}

export function ReportCard({ isAnalyzing, expanded = false }: ReportCardProps) {
  const report = {
    executive_summary: 'Compound shows low toxicity probability (23%) with high model confidence. Literature review supports safe clinical use profile.',
    toxicity_assessment: {
      level: 'LOW',
      confidence: 'High',
      key_findings: [
        'GNN model prediction: 23% toxic probability',
        'Confidence score: 89%',
        'Passes RDKit molecular validation',
      ],
    },
    structural_alerts: [
      { type: 'Carbonyl group (C=O)', severity: 'info', description: 'Primary attribution site but not a toxicophore in this context' },
    ],
    literature_support: {
      sources: 2,
      consensus: 'Low toxicity in therapeutic use',
    },
    recommendations: [
      'Compound suitable for further development',
      'Monitor acetyl group metabolic pathways',
      'Consider clinical dose-response studies',
    ],
  };

  return (
    <Card className="bg-white/80 backdrop-blur-md border-slate-200 hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-600 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div>
              <CardTitle>Report Writer Agent</CardTitle>
              <CardDescription>Gemini 1.5 Pro - Comprehensive Analysis</CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <Eye className="w-4 h-4 mr-2" />
              View
            </Button>
            <Button variant="default" size="sm" className="bg-amber-600 hover:bg-amber-700">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Executive Summary */}
        <div className="bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Info className="w-4 h-4 text-amber-600" />
            <span className="text-sm font-semibold text-slate-900">Executive Summary</span>
          </div>
          <p className="text-sm text-slate-700 leading-relaxed">{report.executive_summary}</p>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center mb-1">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
            </div>
            <p className="text-lg font-bold text-emerald-700">{report.toxicity_assessment.level}</p>
            <p className="text-xs text-slate-600">Toxicity Level</p>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center mb-1">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <p className="text-lg font-bold text-blue-700">{report.literature_support.sources}</p>
            <p className="text-xs text-slate-600">Sources</p>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center mb-1">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
            </div>
            <p className="text-lg font-bold text-amber-700">{report.structural_alerts.length}</p>
            <p className="text-xs text-slate-600">Alerts</p>
          </div>
        </div>

        {expanded && (
          <Accordion type="single" collapsible className="w-full">
            {/* Toxicity Assessment */}
            <AccordionItem value="assessment">
              <AccordionTrigger className="text-sm font-semibold">
                Toxicity Assessment Details
              </AccordionTrigger>
              <AccordionContent className="space-y-2">
                {report.toxicity_assessment.key_findings.map((finding, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 mt-0.5 shrink-0" />
                    <span className="text-slate-700">{finding}</span>
                  </div>
                ))}
              </AccordionContent>
            </AccordionItem>

            {/* Structural Alerts */}
            <AccordionItem value="alerts">
              <AccordionTrigger className="text-sm font-semibold">
                Structural Alerts
              </AccordionTrigger>
              <AccordionContent className="space-y-2">
                {report.structural_alerts.map((alert, idx) => (
                  <div key={idx} className="bg-slate-50 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1">
                      <Badge 
                        variant={alert.severity === 'high' ? 'destructive' : 'outline'}
                        className="text-xs"
                      >
                        {alert.severity}
                      </Badge>
                      <span className="text-sm font-medium text-slate-900">{alert.type}</span>
                    </div>
                    <p className="text-xs text-slate-600">{alert.description}</p>
                  </div>
                ))}
              </AccordionContent>
            </AccordionItem>

            {/* Recommendations */}
            <AccordionItem value="recommendations">
              <AccordionTrigger className="text-sm font-semibold">
                Recommendations
              </AccordionTrigger>
              <AccordionContent className="space-y-2">
                {report.recommendations.map((rec, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-sm">
                    <div className="w-5 h-5 bg-blue-100 rounded-full flex items-center justify-center shrink-0 mt-0.5">
                      <span className="text-xs font-bold text-blue-700">{idx + 1}</span>
                    </div>
                    <span className="text-slate-700">{rec}</span>
                  </div>
                ))}
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        )}

        {/* Generation Info */}
        <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-600 space-y-1">
          <p><span className="font-semibold">Model:</span> Gemini 1.5 Pro</p>
          <p><span className="font-semibold">Temperature:</span> 0.2 (deterministic)</p>
          <p><span className="font-semibold">Format:</span> Structured JSON Schema</p>
        </div>
      </CardContent>
    </Card>
  );
}
