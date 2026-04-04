import { Navbar } from '../components/navbar';
import { Users, Target, Zap, Shield, Github, Mail } from 'lucide-react';
import { Button } from '../components/ui/button';
import logoImage from '../../assets/a654c40bdf5d3906916ebeed588d27aa413d5bd4.png';
import { Footer } from '../components/footer';

export function AboutPage() {
  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: 'var(--bg)',
      fontFamily: 'Inter, sans-serif'
    }}>
      <Navbar />
      
      <main className="max-w-5xl mx-auto px-6 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <img src={logoImage} alt="ToxAgent Logo" className="h-20 mx-auto mb-6" />
          <h1 className="text-5xl font-bold mb-4" style={{ color: 'var(--text)' }}>
            About ToxAgent
          </h1>
          <p className="text-xl max-w-3xl mx-auto" style={{ color: 'var(--text-muted)' }}>
            A cutting-edge multi-agent AI system for molecular toxicity analysis, 
            powered by Gemini 2.0 Flash models and advanced graph neural networks.
          </p>
        </div>

        {/* Mission & Vision */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <div className="rounded-xl p-8" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <Target className="w-8 h-8 mb-4" style={{ color: 'var(--accent-blue)' }} />
            <h2 className="text-2xl font-bold mb-3" style={{ color: 'var(--text)' }}>
              Our Mission
            </h2>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              To accelerate drug discovery and development by providing researchers with 
              powerful, AI-driven toxicity prediction tools that are accurate, interpretable, 
              and easy to use.
            </p>
          </div>

          <div className="rounded-xl p-8" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <Zap className="w-8 h-8 mb-4" style={{ color: 'var(--accent-blue)' }} />
            <h2 className="text-2xl font-bold mb-3" style={{ color: 'var(--text)' }}>
              Why ToxAgent?
            </h2>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Traditional toxicity testing is expensive and time-consuming. ToxAgent uses 
              state-of-the-art AI to predict molecular toxicity in seconds, helping researchers 
              make informed decisions early in the drug development pipeline.
            </p>
          </div>
        </div>

        {/* Technology Stack */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-center" style={{ color: 'var(--text)' }}>
            Technology Stack
          </h2>
          <div className="rounded-xl p-8" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text)' }}>
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--accent-blue)' }} />
                  Multi-Agent Architecture
                </h3>
                <ul className="space-y-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                  <li>• <strong>Orchestrator Agent:</strong> Google ADK LlmAgent coordination</li>
                  <li>• <strong>Screening Agent:</strong> GNN-based toxicity prediction</li>
                  <li>• <strong>Explainer Agent:</strong> Molecular attribution with GNNExplainer</li>
                  <li>• <strong>Researcher Agent:</strong> Literature search via RAG</li>
                  <li>• <strong>Report Writer Agent:</strong> Structured output generation</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold mb-3 flex items-center gap-2" style={{ color: 'var(--text)' }}>
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--accent-blue)' }} />
                  AI Models & APIs
                </h3>
                <ul className="space-y-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                  <li>• <strong>Gemini 2.0 Flash:</strong> Fast, efficient LLM inference</li>
                  <li>• <strong>Graph Neural Networks:</strong> Molecular property prediction</li>
                  <li>• <strong>FastAPI:</strong> High-performance REST API</li>
                  <li>• <strong>PubChem & PubMed:</strong> Chemical and literature databases</li>
                  <li>• <strong>Tox21 Dataset:</strong> 12 toxicity pathway endpoints</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-center" style={{ color: 'var(--text)' }}>
            Key Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="rounded-xl p-6 text-center" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <div className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center" style={{ backgroundColor: 'var(--accent-blue-muted)' }}>
                <Zap className="w-6 h-6" style={{ color: 'var(--accent-blue)' }} />
              </div>
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text)' }}>Fast Analysis</h3>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Get comprehensive toxicity predictions in seconds, not hours
              </p>
            </div>

            <div className="rounded-xl p-6 text-center" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <div className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center" style={{ backgroundColor: 'var(--accent-blue-muted)' }}>
                <Shield className="w-6 h-6" style={{ color: 'var(--accent-blue)' }} />
              </div>
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text)' }}>Interpretable Results</h3>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Understand which molecular features contribute to toxicity
              </p>
            </div>

            <div className="rounded-xl p-6 text-center" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <div className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center" style={{ backgroundColor: 'var(--accent-blue-muted)' }}>
                <Users className="w-6 h-6" style={{ color: 'var(--accent-blue)' }} />
              </div>
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text)' }}>Multi-Agent Workflow</h3>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
                Parallel processing for maximum efficiency and accuracy
              </p>
            </div>
          </div>
        </section>

        {/* Event Info */}
        <section className="mb-12">
          <div className="rounded-xl p-8 text-center" style={{ 
            backgroundColor: 'var(--surface)', 
            border: '2px solid var(--accent-blue)',
            boxShadow: '0 0 20px rgba(59, 130, 246, 0.15)'
          }}>
            <h2 className="text-2xl font-bold mb-3" style={{ color: 'var(--text)' }}>
              GDGoC Vietnam 2026 Hackathon
            </h2>
            <p className="text-base mb-4" style={{ color: 'var(--text-muted)' }}>
              ToxAgent was developed for the Google Developer Groups on Campus Vietnam 2026 Hackathon, 
              demonstrating the power of AI in accelerating scientific research and drug discovery.
            </p>
            <div className="flex items-center justify-center gap-2 text-sm" style={{ color: 'var(--text-faint)' }}>
              <span>📅 March 31, 2026</span>
              <span>•</span>
              <span>🇻🇳 Vietnam</span>
            </div>
          </div>
        </section>

        {/* Team & Contact */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold mb-6 text-center" style={{ color: 'var(--text)' }}>
            Get in Touch
          </h2>
          <div className="rounded-xl p-8 text-center" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <p className="text-base mb-6" style={{ color: 'var(--text-muted)' }}>
              Have questions or feedback? We'd love to hear from you!
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button
                variant="outline"
                className="flex items-center gap-2"
              >
                <Github className="w-4 h-4" />
                View on GitHub
              </Button>
              <Button
                className="flex items-center gap-2"
                style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
              >
                <Mail className="w-4 h-4" />
                Contact Us
              </Button>
            </div>
          </div>
        </section>

        {/* Footer */}
        <div className="text-center pt-8 border-t" style={{ borderColor: 'var(--border)', color: 'var(--text-faint)' }}>
          <p className="text-sm">
            © 2026 ToxAgent. Built for the GDGoC Vietnam Hackathon.
          </p>
          <p className="text-xs mt-2">
            Powered by Gemini 2.0 Flash • Google ADK • FastAPI • React
          </p>
        </div>
      </main>
      <Footer />
    </div>
  );
}
