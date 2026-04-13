import { useState } from 'react';
import { BookOpenText, FileText, Link as LinkIcon } from 'lucide-react';
import { Navbar } from '../components/navbar';
import { Footer } from '../components/footer';

import docImage1 from '../../assets/documents/1-interface.png';
import docImage2 from '../../assets/documents/2-analysis-pipeline.png';
import docImage3 from '../../assets/documents/3-quick-prediction.png';
import docImage4 from '../../assets/documents/4-report-part-0.png';
import docImage5 from '../../assets/documents/5-report-part-1.png';
import docImage6 from '../../assets/documents/6-report-part-2.png';
import docImage7 from '../../assets/documents/7-report-part-3.png';
import docImage8 from '../../assets/documents/8-report-part-4.png';
import docImage9 from '../../assets/documents/9-report-part-5.png';

const DOCUMENTS = [
  {
    id: 'user-guide',
    title: 'User Guide',
    description: 'ToxAgent system guide for toxicity assessment workflow.',
  },
] as const;

const SECTION_ITEMS = [
  { id: 'intro', label: '1. Introduction' },
  { id: 'objectives', label: '2. System Objectives' },
  { id: 'target-users', label: '3. Target User' },
  { id: 'how-to-use', label: '4. How To Use' },
  { id: 'result-components', label: '5. Result Components' },
  { id: 'limitations', label: '6. Notes & Limitations' },
] as const;

interface FigureCardProps {
  src: string;
  alt: string;
  caption: string;
}

function FigureCard({ src, alt, caption }: FigureCardProps) {
  return (
    <figure
      className="rounded-xl overflow-hidden"
      style={{ border: '1px solid var(--border)', backgroundColor: 'var(--surface)' }}
    >
      <img src={src} alt={alt} className="w-full h-auto" loading="lazy" />
      <figcaption className="px-4 py-3 text-xs" style={{ color: 'var(--text-muted)' }}>
        {caption}
      </figcaption>
    </figure>
  );
}

export function DocumentsPage() {
  const [activeSection, setActiveSection] = useState<string>('intro');

  const handleScrollTo = (sectionId: string) => {
    setActiveSection(sectionId);
    if (typeof window !== 'undefined') {
      document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
      <Navbar />

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] max-w-[1480px] mx-auto">
        <aside
          className="sticky top-16 h-[calc(100vh-4rem)] border-r p-6 overflow-y-auto"
          style={{ borderColor: 'var(--border)' }}
        >
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-3">
              <BookOpenText className="w-4 h-4" style={{ color: 'var(--accent-blue)' }} />
              <h2 className="text-sm font-semibold uppercase" style={{ color: 'var(--text)', letterSpacing: '0.05em' }}>
                Documents
              </h2>
            </div>

            <div className="space-y-2">
              {DOCUMENTS.map((doc) => (
                <button
                  key={doc.id}
                  type="button"
                  className="w-full text-left rounded-lg px-3 py-2 transition-colors"
                  style={{
                    backgroundColor: 'var(--accent-blue-muted)',
                    color: 'var(--accent-blue)',
                  }}
                >
                  <div className="flex items-start gap-2">
                    <FileText className="w-4 h-4 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold">{doc.title}</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                        {doc.description}
                      </p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="border-t pt-6" style={{ borderColor: 'var(--border)' }}>
            <h3 className="text-xs font-semibold uppercase mb-3" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
              On This Page
            </h3>
            <nav className="space-y-2">
              {SECTION_ITEMS.map((section) => (
                <button
                  key={section.id}
                  type="button"
                  onClick={() => handleScrollTo(section.id)}
                  className="w-full text-left px-3 py-2 rounded-lg text-sm transition-colors"
                  style={{
                    backgroundColor: activeSection === section.id ? 'var(--accent-blue-muted)' : 'transparent',
                    color: activeSection === section.id ? 'var(--accent-blue)' : 'var(--text-muted)',
                  }}
                >
                  {activeSection === section.id ? '●' : '○'} {section.label}
                </button>
              ))}
            </nav>
          </div>
        </aside>

        <main className="p-6 md:p-10 space-y-10 max-w-[960px]">
          <header className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <p className="text-xs uppercase mb-2" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
              ToxAgent Documentation
            </p>
            <h1 className="text-3xl font-bold mb-2" style={{ color: 'var(--text)' }}>
              User Guide
            </h1>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              A complete guide for the ToxAgent toxicity analysis workflow based on SMILES strings.
            </p>
          </header>

          <section id="intro" className="space-y-4">
            <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>
              1. Introduction to ToxAgent
            </h2>
            <div className="rounded-xl p-6 space-y-3" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
                ToxAgent is an intelligent support system designed to analyze molecular toxicity from SMILES string input.
                Instead of returning a single score, it combines predictive modeling, research evidence retrieval, and
                structured report generation.
              </p>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
                The system helps users interpret toxicity signals through mechanism-level results, structure-based
                explanations, and scientific context from external sources.
              </p>
            </div>
          </section>

          <section id="objectives" className="space-y-4">
            <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>
              2. System Objectives
            </h2>
            <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <ul className="list-disc pl-5 space-y-2" style={{ color: 'var(--text-muted)' }}>
                <li>Provide a complete and easy-to-understand toxicity analysis workflow.</li>
                <li>Support convenient compound screening in one integrated system.</li>
                <li>Help identify potential toxicity risks from input molecules.</li>
                <li>Allow users to review supporting evidence in the same workflow.</li>
                <li>Connect model predictions with external scientific context.</li>
                <li>Improve transparency and interpretability for downstream decisions.</li>
              </ul>
            </div>
          </section>

          <section id="target-users" className="space-y-4">
            <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>
              3. Target User
            </h2>
            <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <ul className="list-disc pl-5 space-y-2" style={{ color: 'var(--text-muted)' }}>
                <li>Students in chemistry, pharmacy, bioinformatics, and AI for biomedical applications.</li>
                <li>Researchers who need support for molecular toxicity screening and interpretation.</li>
                <li>Users needing quick risk assessment from SMILES strings.</li>
                <li>Development teams demonstrating AI-powered toxicity analysis systems.</li>
              </ul>
            </div>
          </section>

          <section id="how-to-use" className="space-y-5">
            <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>
              4. How to Use ToxAgent
            </h2>

            <div className="rounded-xl p-6 space-y-3" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <p style={{ color: 'var(--text-muted)' }}>
                Open ToxAgent at{' '}
                <a
                  href="https://tox-agent.firebaseapp.com/"
                  className="inline-flex items-center gap-1 underline"
                  style={{ color: 'var(--accent-blue)' }}
                >
                  https://tox-agent.firebaseapp.com/
                  <LinkIcon className="w-3 h-3" />
                </a>
                .
              </p>
              <p style={{ color: 'var(--text-muted)' }}>
                Enter the molecule SMILES in the input box, then click Analyze. The pipeline performs validation,
                prediction, literature retrieval, and structured report generation automatically.
              </p>
            </div>

            <FigureCard
              src={docImage1}
              alt="ToxAgent main interface with SMILES input and Analyze button"
              caption="ToxAgent main interface on the Drug Toxicity Analysis page."
            />
            <FigureCard
              src={docImage2}
              alt="Analysis pipeline with InputValidator ScreeningAgent ResearchAgent WriterAgent"
              caption="Analysis pipeline execution showing InputValidator, ScreeningAgent, ResearchAgent, and WriterAgent."
            />
            <FigureCard
              src={docImage3}
              alt="Quick prediction card after analysis"
              caption="Quick prediction view after analysis is completed."
            />
          </section>

          <section id="result-components" className="space-y-6">
            <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>
              5. Explanation of Result Components
            </h2>

            <FigureCard
              src={docImage4}
              alt="Decision Metrics Dashboard and overview report layout"
              caption="Overview screen with report sections and Decision Metrics Dashboard."
            />

            <article className="rounded-xl p-6 space-y-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <h3 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                a. Clinical Toxicity
              </h3>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
                Displays the overall toxicity prediction including label, probability, confidence, and interpretation.
                This helps users quickly understand whether the compound is predicted to be safe or potentially toxic.
              </p>
              <FigureCard
                src={docImage5}
                alt="Clinical toxicity panel with label probability confidence"
                caption="Figure 1. Clinical toxicity result with predicted label, model probability, and confidence."
              />
            </article>

            <article className="rounded-xl p-6 space-y-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <h3 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                b. Toxicity Mechanism Profile
              </h3>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
                Shows mechanism-level risk scores, allowing users to identify the highest-risk mechanism and compare
                relative signals across all endpoints.
              </p>
              <FigureCard
                src={docImage6}
                alt="Mechanism profiling bar chart for Tox21 tasks"
                caption="Figure 2. Mechanism profiling across toxicity pathways, highlighting the highest risk."
              />
            </article>

            <article className="rounded-xl p-6 space-y-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <h3 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                c. Structural Explanation
              </h3>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
                Visualizes molecular structure attribution with heatmaps plus top atoms and top bonds to indicate
                influential regions used by the model.
              </p>
              <FigureCard
                src={docImage7}
                alt="Structural explanation with molecule heatmap top atoms and top bonds"
                caption="Figure 3. Structural explanation with molecule view, attribution heatmap, top atoms, and top bonds."
              />
            </article>

            <article className="rounded-xl p-6 space-y-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <h3 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                d. Literature Context
              </h3>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
                Retrieves supporting references from PubChem and related papers so users can compare model output with
                scientific context.
              </p>
              <FigureCard
                src={docImage8}
                alt="Literature context with PubChem metadata and related studies"
                caption="Figure 4. Literature context with PubChem information and related studies."
              />
            </article>

            <article className="rounded-xl p-6 space-y-4" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <h3 className="text-xl font-semibold" style={{ color: 'var(--text)' }}>
                e. AI Recommendations
              </h3>
              <p style={{ color: 'var(--text-muted)', lineHeight: '1.8' }}>
                Provides a final synthesized recommendation including executive summary, risk level, and practical
                action points. The chatbot can be used for follow-up questions.
              </p>
              <FigureCard
                src={docImage9}
                alt="AI recommendations with executive summary risk level and chatbot"
                caption="Figure 5. AI recommendations with executive summary, risk level, recommendations, and chatbot."
              />
            </article>
          </section>

          <section id="limitations" className="space-y-4 pb-10">
            <h2 className="text-2xl font-bold" style={{ color: 'var(--text)' }}>
              6. Notes and Limitations of the System
            </h2>
            <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
              <ul className="list-disc pl-5 space-y-2" style={{ color: 'var(--text-muted)' }}>
                <li>ToxAgent is a support tool and is not a replacement for expert judgment or experimental data.</li>
                <li>Prediction quality depends on SMILES quality and training data scope.</li>
                <li>Heatmaps, top atoms, and top bonds are interpretive aids, not definitive evidence.</li>
                <li>PubChem and paper retrieval provide context but may not cover all relevant literature.</li>
                <li>Reliability may vary across compounds, especially when confidence is low.</li>
                <li>Use the system for screening and preliminary support, not as a sole decision basis.</li>
              </ul>
            </div>
          </section>
        </main>
      </div>

      <Footer />
    </div>
  );
}
